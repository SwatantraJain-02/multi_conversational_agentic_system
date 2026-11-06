"""Hybrid index over the schema docs and the seed question -> SQL examples.

Two branches, fused with RRF (see retrieval/hybrid.py):
- dense: Chroma collections `schema_docs` / `sql_seed_examples`, built offline by
  `build_index()` (scripts/build_schema_index.py);
- keyword: in-process BM25 built from the catalog YAML, never from Chroma, so keyword
  search keeps working when Chroma or Ollama is down (the result is then `degraded`).

The search functions are what the MCP tools `search_schema` and `get_few_shots` call.
"""

import hashlib
from functools import cache
from typing import Any

from chromadb import Collection

from multi_agenti_conversation_system.llm import get_embedding_model
from multi_agenti_conversation_system.mcp.servers.db_server import catalog
from multi_agenti_conversation_system.retrieval.hybrid import (
    BM25Index,
    DenseHit,
    HybridResult,
    candidate_depth,
    hybrid_search,
)
from multi_agenti_conversation_system.vectorstore import get_collection

SCHEMA_COLLECTION = "schema_docs"
FEW_SHOT_COLLECTION = "sql_seed_examples"

# nomic-embed-text is trained with these task prefixes; retrieval is noticeably better with them.
# The stored document text stays clean, the prefix is added only when embedding.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "


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


def _upsert(collection: Collection, ids, documents, metadatas) -> None:
    vectors = get_embedding_model().embed_documents([DOC_PREFIX + d for d in documents])
    collection.upsert(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors
    )
    # Remove entries whose table/question no longer exists in the YAML files.
    stale = set(collection.get(include=[])["ids"]) - set(ids)
    if stale:
        collection.delete(ids=list(stale))


def build_index() -> dict[str, int]:
    """(Re)build both collections from data/*.yaml and return their sizes."""
    tables = list(catalog.schema_tables())
    schema = get_collection(SCHEMA_COLLECTION)
    _upsert(
        schema,
        ids=[_table_id(t) for t in tables],
        documents=[catalog.table_document(t) for t in tables],
        metadatas=[_table_metadata(t) for t in tables],
    )

    examples = catalog.few_shots()
    few_shots = get_collection(FEW_SHOT_COLLECTION)
    _upsert(
        few_shots,
        ids=[_few_shot_id(e["question"]) for e in examples],
        # The question is what gets embedded; the SQL travels along as metadata.
        documents=[e["question"] for e in examples],
        metadatas=[_few_shot_metadata(e) for e in examples],
    )
    return {SCHEMA_COLLECTION: schema.count(), FEW_SHOT_COLLECTION: few_shots.count()}


# ---------------------------------------------------------------- keyword corpora
@cache
def _schema_bm25() -> BM25Index:
    tables = list(catalog.schema_tables())
    return BM25Index(
        ids=[_table_id(t) for t in tables],
        documents=[catalog.table_document(t) for t in tables],
        metadatas=[_table_metadata(t) for t in tables],
    )


@cache
def _few_shot_bm25() -> BM25Index:
    examples = catalog.few_shots()
    return BM25Index(
        ids=[_few_shot_id(e["question"]) for e in examples],
        # Question + SQL, so "freight" or "unit_price" also match through the SQL.
        documents=[f"{e['question']}\n{e['sql']}" for e in examples],
        metadatas=[
            {"question": e["question"], **_few_shot_metadata(e)} for e in examples
        ],
    )


# ---------------------------------------------------------------------- search
def _dense(
    collection_name: str, question: str, n: int, where: dict[str, Any] | None
) -> list[DenseHit]:
    # The collection is fetched per call, so a Chroma outage at startup can't stop the server.
    collection = get_collection(collection_name)
    if n == 0 or collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[get_embedding_model().embed_query(QUERY_PREFIX + question)],
        n_results=n,
        where=where,
        include=["distances"],
    )
    return [
        DenseHit(id=doc_id, similarity=round(1 - dist, 4))
        for doc_id, dist in zip(result["ids"][0], result["distances"][0])
    ]


def _hybrid(
    collection_name: str, index: BM25Index, question: str, k: int, domain: str | None
) -> HybridResult:
    where = {"domain": domain} if domain else None
    # Depth comes from the filtered catalog: collection.count() ignores the where filter.
    depth = candidate_depth(index.count(where), k)
    return hybrid_search(
        dense_fn=lambda n: _dense(collection_name, question, n, where),
        keyword_fn=lambda n: index.search(question, n, where),
        k=k,
        depth=depth,
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
    index = _schema_bm25()
    result = _hybrid(SCHEMA_COLLECTION, index, question, k, domain)
    meta = dict(zip(index.ids, index.metadatas))
    hits = [
        {
            **meta[hit.id],
            "document": catalog.table_document(meta[hit.id]["table"]),
            **_ranking(hit),
        }
        for hit in result.hits
    ]
    return hits, result.degraded


def search_few_shots(
    question: str, k: int = 3, domain: str | None = None
) -> tuple[list[dict[str, Any]], bool]:
    """Most similar solved examples, and whether the dense branch was down.

    Hits: [{question, sql, domain, pii, similarity, rrf_score, dense_rank, bm25_rank}].
    """
    index = _few_shot_bm25()
    result = _hybrid(FEW_SHOT_COLLECTION, index, question, k, domain)
    meta = dict(zip(index.ids, index.metadatas))
    hits = [{**meta[hit.id], **_ranking(hit)} for hit in result.hits]
    return hits, result.degraded
