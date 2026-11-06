"""Hybrid retrieval: dense (semantic) + BM25 (keyword) rankings fused with RRF.

Pure code, no Chroma, no catalog: the caller supplies the corpus and the two search
functions. Used by the DB MCP server for schema docs and few-shots, and reused agent-side
for learned SQL pairs.

Why both: dense search handles paraphrases ("money we made" ~ "revenue") but blurs exact
terms; BM25 nails identifiers and rare words ("unit_price", "freight") but misses synonyms.
RRF fuses *ranks*, not scores, because cosine similarity and BM25 scores live on unrelated
scales.
"""

import logging
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import bm25s

logger = logging.getLogger(__name__)

RRF_K = 60
# Below this size, both branches look at every (filtered) doc, so every hit gets a dense
# similarity and the fusion sees complete rankings.
FULL_DEPTH_MAX = 50

STOPWORDS = frozenset(
    (  # noqa: SIM905 - one compact line beats 37 list items
        "a an and are as at be by did do does for from how i in is it many me much of on or "
        "our per show the to was we were what when which who with"
    ).split()
)

_IDENTIFIER = re.compile(r"[A-Za-z0-9_]+")
# CustomerID -> Customer, ID ; unitPrice -> unit, Price ; order2024 -> order, 2024
_CAMEL_PART = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")


def _singular(word: str) -> str:
    """Naive plural strip: orders -> order, categories -> category; address, status stay."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    """Lowercase keyword tokens; compound identifiers are kept whole *and* split.

    "unit_price" -> ["unit_price", "unit", "price"]; "CustomerID" -> ["customerid",
    "customer", "id"]. Used for both documents and queries, so they always agree.
    """
    tokens: list[str] = []
    for raw in _IDENTIFIER.findall(text):
        # Split camelCase before lowercasing; afterwards the case information is gone.
        parts = [
            p.lower() for chunk in raw.split("_") for p in _CAMEL_PART.findall(chunk)
        ]
        whole = raw.lower().strip("_")
        if len(parts) > 1 and whole not in STOPWORDS:
            tokens.append(whole)
        tokens.extend(_singular(p) for p in parts if p not in STOPWORDS)
    return tokens


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    return not where or all(metadata.get(key) == value for key, value in where.items())


class BM25Index:
    """In-memory BM25 over a small corpus, with Lucene IDF (never negative, G11).

    Always scores against the *full* corpus and filters by metadata afterwards, so a
    domain filter that leaves 2 docs doesn't shrink the IDF statistics to nothing.
    """

    def __init__(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict[str, Any]],
    ):
        if not len(ids) == len(documents) == len(metadatas):
            raise ValueError("ids, documents and metadatas must have the same length")
        self.ids = list(ids)
        self.metadatas = list(metadatas)
        self._bm25: bm25s.BM25 | None = None
        if self.ids:
            self._bm25 = bm25s.BM25(method="lucene")
            self._bm25.index([tokenize(d) for d in documents], show_progress=False)

    def count(self, where: dict[str, Any] | None = None) -> int:
        """How many docs pass the metadata filter."""
        return sum(_matches(m, where) for m in self.metadatas)

    def search(
        self, query: str, k: int, where: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Top-k (id, score) pairs, best first. Docs that don't match any term are dropped."""
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        hits = [
            (doc_id, float(score))
            for doc_id, score, meta in zip(self.ids, scores, self.metadatas)
            if score > 0 and _matches(meta, where)
        ]
        hits.sort(key=lambda hit: -hit[1])
        return hits[:k]


def rrf(
    ranked_lists: Sequence[Sequence[str]],
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(d) = sum_i w_i / (k + rank_i(d)), 1-based ranks.

    A list that misses the doc adds nothing. Ties go to the better rank in the first list
    (dense), then the next list, then the id, so the order is fully deterministic.
    """
    weights = list(weights) if weights is not None else [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("need one weight per ranked list")

    scores: dict[str, float] = {}
    ranks: dict[str, list[float]] = {}
    for i, (ranking, weight) in enumerate(zip(ranked_lists, weights)):
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
            ranks.setdefault(doc_id, [math.inf] * len(ranked_lists))[i] = rank
    return sorted(scores.items(), key=lambda item: (-item[1], *ranks[item[0]], item[0]))


@dataclass
class DenseHit:
    id: str
    similarity: float


@dataclass
class HybridHit:
    id: str
    rrf_score: float
    similarity: float | None  # None when the dense branch didn't return this doc
    dense_rank: int | None
    bm25_rank: int | None


@dataclass
class HybridResult:
    hits: list[HybridHit]
    degraded: bool  # True when the dense branch failed and only BM25 answered


def candidate_depth(filtered_size: int, k: int) -> int:
    """How many candidates each branch fetches before fusion."""
    return filtered_size if filtered_size <= FULL_DEPTH_MAX else max(20, 4 * k)


def hybrid_search(
    dense_fn: Callable[[int], list[DenseHit]],
    keyword_fn: Callable[[int], list[tuple[str, float]]],
    k: int,
    depth: int,
) -> HybridResult:
    """Run both branches at `depth`, fuse with RRF, keep the top k.

    The dense branch is guarded: if Chroma or Ollama fails (or the collection is empty,
    i.e. never built), the result comes from BM25 alone with degraded=True instead of an
    error. BM25 runs in-process and can't fail that way. depth=0 (the filter matches
    nothing) is a plain empty result, not an outage.
    """
    if depth <= 0:
        return HybridResult(hits=[], degraded=False)
    try:
        dense = dense_fn(depth)
        degraded = not dense
        if degraded:
            logger.warning("dense search returned nothing; is the index built?")
    except Exception as exc:  # noqa: BLE001 - failing soft is the point
        logger.warning(
            "dense search failed, using BM25 only: %s: %s", type(exc).__name__, exc
        )
        dense, degraded = [], True
    keyword = keyword_fn(depth)

    similarity = {hit.id: hit.similarity for hit in dense}
    dense_rank = {hit.id: rank for rank, hit in enumerate(dense, start=1)}
    bm25_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(keyword, start=1)}
    fused = rrf([[hit.id for hit in dense], [doc_id for doc_id, _ in keyword]])
    hits = [
        HybridHit(
            id=doc_id,
            rrf_score=round(score, 6),
            similarity=similarity.get(doc_id),
            dense_rank=dense_rank.get(doc_id),
            bm25_rank=bm25_rank.get(doc_id),
        )
        for doc_id, score in fused[:k]
    ]
    return HybridResult(hits=hits, degraded=degraded)
