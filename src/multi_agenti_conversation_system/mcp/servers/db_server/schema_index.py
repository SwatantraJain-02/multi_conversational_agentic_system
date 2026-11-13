"""Hybrid index over the schema docs and the seed question -> SQL examples.

Both branches run in Pinecone (see vectorstore.py) and are fused with RRF
(retrieval/hybrid.py):
- keyword: BM25 over the `keywords` field (our tokenizer's output);
- dense: cosine similarity on the `embedding` field.

The namespaces `schema_docs` / `sql_seed_examples` are built offline by `build_index()`
(scripts/build_schema_index.py). The catalog YAML stays the source of truth: hit metadata
and the candidate depth come from it, so search results never depend on stale index
metadata. If the embedding API is down, the keyword branch answers alone (`degraded`).

The search functions are what the MCP tools `search_schema` and `get_few_shots` call.
"""

import hashlib
from dataclasses import dataclass
from functools import cache
from typing import Any

from multi_agenti_conversation_system import vectorstore
from multi_agenti_conversation_system.llm import get_embedding_model
from multi_agenti_conversation_system.mcp.servers.db_server import catalog
from multi_agenti_conversation_system.retrieval.hybrid import (
    DenseHit,
    HybridResult,
    candidate_depth,
    hybrid_search,
    keyword_query,
)
from multi_agenti_conversation_system.vectorstore import SCHEMA_DOCS, SQL_SEED_EXAMPLES


def _few_shot_id(question: str) -> str:
    # Stable across rebuilds, so re-running the build updates entries instead of duplicating.
    return "fewshot-" + hashlib.sha1(question.encode()).hexdigest()[:12]


def _table_id(table: str) -> str:
    return f"table-{table}"


def _table_metadata(table: str) -> dict[str, Any]:
    return {
        "table": table,
        "domain": catalog.table_domain(table),
        "has_pii": table.lower() in catalog.pii_columns(),
    }


def _few_shot_metadata(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "domain": example["domain"],
        "sql": example["sql"].strip(),
        "pii": bool(example.get("pii", False)),
    }


@dataclass(frozen=True)
class _Corpus:
    """One namespace's documents as the YAML defines them."""

    ids: list[str]
    dense_texts: list[str]  # what gets embedded
    keyword_texts: list[str]  # what gets BM25-indexed
    metadatas: list[dict[str, Any]]

    def count(self, domain: str | None) -> int:
        return sum(domain is None or m["domain"] == domain for m in self.metadatas)

    def metadata(self) -> dict[str, dict[str, Any]]:
        return dict(zip(self.ids, self.metadatas))


@cache
def _schema_corpus() -> _Corpus:
    tables = list(catalog.schema_tables())
    documents = [catalog.table_document(t) for t in tables]
    return _Corpus(
        ids=[_table_id(t) for t in tables],
        dense_texts=documents,
        keyword_texts=documents,
        metadatas=[_table_metadata(t) for t in tables],
    )


@cache
def _few_shot_corpus() -> _Corpus:
    examples = catalog.few_shots()
    return _Corpus(
        ids=[_few_shot_id(e["question"]) for e in examples],
        # Dense embeds the question; BM25 indexes question + SQL, so "freight" or
        # "unit_price" also match through the SQL.
        dense_texts=[e["question"] for e in examples],
        keyword_texts=[f"{e['question']}\n{e['sql']}" for e in examples],
        metadatas=[
            {"question": e["question"], **_few_shot_metadata(e)} for e in examples
        ],
    )


# ----------------------------------------------------------------------- build
def _upsert(namespace: str, corpus: _Corpus) -> None:
    vectors = get_embedding_model().embed_documents(corpus.dense_texts)
    vectorstore.upsert(
        namespace,
        [
            {
                "_id": doc_id,
                vectorstore.KEYWORDS_FIELD: keyword_query(keyword_text),
                vectorstore.EMBEDDING_FIELD: vector,
                "text": dense_text,
                **metadata,
            }
            for doc_id, dense_text, keyword_text, vector, metadata in zip(
                corpus.ids,
                corpus.dense_texts,
                corpus.keyword_texts,
                vectors,
                corpus.metadatas,
            )
        ],
    )
    # Remove entries whose table/question no longer exists in the YAML files.
    stale = set(vectorstore.list_ids(namespace)) - set(corpus.ids)
    vectorstore.delete(namespace, sorted(stale))


def build_index() -> dict[str, int]:
    """(Re)build both namespaces from data/*.yaml and return how many entries each got."""
    sizes = {}
    for namespace, corpus in (
        (SCHEMA_DOCS, _schema_corpus()),
        (SQL_SEED_EXAMPLES, _few_shot_corpus()),
    ):
        _upsert(namespace, corpus)
        # Counted from the YAML: a fresh upsert isn't instantly searchable (G14).
        sizes[namespace] = len(corpus.ids)
    return sizes


# ---------------------------------------------------------------------- search
def _dense(namespace: str, question: str, n: int, domain: str | None) -> list[DenseHit]:
    matches = vectorstore.dense_search(
        namespace,
        get_embedding_model().embed_query(question),
        top_k=n,
        filter=vectorstore.eq_filter(domain=domain),
    )
    return [DenseHit(id=m.id, similarity=round(m.score, 4)) for m in matches]


def _keyword(
    namespace: str, question: str, n: int, domain: str | None
) -> list[tuple[str, float]]:
    matches = vectorstore.keyword_search(
        namespace,
        keyword_query(question),
        top_k=n,
        filter=vectorstore.eq_filter(domain=domain),
    )
    return [(m.id, m.score) for m in matches]


def _hybrid(
    namespace: str, corpus: _Corpus, question: str, k: int, domain: str | None
) -> HybridResult:
    return hybrid_search(
        dense_fn=lambda n: _dense(namespace, question, n, domain),
        keyword_fn=lambda n: _keyword(namespace, question, n, domain),
        k=k,
        depth=candidate_depth(corpus.count(domain), k),
    )


def _ranking(hit) -> dict[str, Any]:
    return {
        "similarity": hit.similarity,
        "rrf_score": hit.rrf_score,
        "dense_rank": hit.dense_rank,
        "bm25_rank": hit.bm25_rank,
    }


def search_schema(
    question: str, k: int = 4, domain: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """Tables most relevant to the question, and whether the dense branch was down.

    Hits: [{table, domain, has_pii, document, similarity, rrf_score, dense_rank, bm25_rank}].
    """
    corpus = _schema_corpus()
    result = _hybrid(SCHEMA_DOCS, corpus, question, k, domain)
    meta = corpus.metadata()
    hits = [
        {
            **meta[hit.id],
            "document": catalog.table_document(meta[hit.id]["table"]),
            **_ranking(hit),
        }
        for hit in result.hits
        if hit.id in meta  # ignore ids the YAML no longer has (index not rebuilt yet)
    ]
    return hits, result.degraded


def search_few_shots(
    question: str, k: int = 3, domain: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """Most similar solved examples, and whether the dense branch was down.

    Hits: [{question, sql, domain, pii, similarity, rrf_score, dense_rank, bm25_rank}].
    """
    corpus = _few_shot_corpus()
    result = _hybrid(SQL_SEED_EXAMPLES, corpus, question, k, domain)
    meta = corpus.metadata()
    hits = [{**meta[hit.id], **_ranking(hit)} for hit in result.hits if hit.id in meta]
    return hits, result.degraded
