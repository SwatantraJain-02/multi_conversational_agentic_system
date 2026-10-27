"""Chroma index over the schema docs and the seed question -> SQL examples.

`build_index()` is run offline (scripts/build_schema_index.py). The search functions are
what the MCP tools `search_schema` and `get_few_shots` call.
"""

import hashlib
from typing import Any

from chromadb import Collection

from multi_agenti_conversation_system.catalog import load_catalog
from multi_agenti_conversation_system.config import get_path_settings
from multi_agenti_conversation_system.llm import get_embedding_model
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
    catalog = load_catalog(get_path_settings().DATA_DIR)
    pii = catalog.pii_columns()
    tables = list(catalog.tables)
    schema = get_collection(SCHEMA_COLLECTION)
    _upsert(
        schema,
        ids=[f"table-{t}" for t in tables],
        documents=[catalog.table_document(t) for t in tables],
        metadatas=[
            {
                "table": t,
                "domain": catalog.table_domain(t),
                "has_pii": t.lower() in pii,
            }
            for t in tables
        ],
    )

    examples = catalog.few_shots
    few_shots = get_collection(FEW_SHOT_COLLECTION)
    _upsert(
        few_shots,
        ids=[_few_shot_id(e["question"]) for e in examples],
        # The question is what gets embedded; the SQL travels along as metadata.
        documents=[e["question"] for e in examples],
        metadatas=[
            {
                "domain": e["domain"],
                "sql": e["sql"].strip(),
                "pii": bool(e.get("pii", False)),
            }
            for e in examples
        ],
    )
    return {SCHEMA_COLLECTION: schema.count(), FEW_SHOT_COLLECTION: few_shots.count()}


def _search(
    collection_name: str, question: str, k: int, domain: str | None
) -> list[dict[str, Any]]:
    collection = get_collection(collection_name)
    total = collection.count()
    if total == 0:
        return []
    result = collection.query(
        query_embeddings=[get_embedding_model().embed_query(QUERY_PREFIX + question)],
        n_results=max(1, min(k, total)),
        where={"domain": domain} if domain else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"document": doc, "metadata": meta, "similarity": round(1 - dist, 4)}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def search_schema(
    question: str, k: int = 4, domain: str | None = None
) -> list[dict[str, Any]]:
    """Tables most relevant to the question: [{table, domain, has_pii, document, similarity}]."""
    return [
        {
            **hit["metadata"],
            "document": hit["document"],
            "similarity": hit["similarity"],
        }
        for hit in _search(SCHEMA_COLLECTION, question, k, domain)
    ]


def search_few_shots(
    question: str, k: int = 3, domain: str | None = None
) -> list[dict[str, Any]]:
    """Most similar solved examples: [{question, sql, domain, pii, similarity}]."""
    return [
        {
            "question": hit["document"],
            "sql": hit["metadata"]["sql"],
            "domain": hit["metadata"]["domain"],
            "pii": hit["metadata"]["pii"],
            "similarity": hit["similarity"],
        }
        for hit in _search(FEW_SHOT_COLLECTION, question, k, domain)
    ]
