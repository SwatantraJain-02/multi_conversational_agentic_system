"""Chroma index over the schema docs and the seed question -> SQL examples.

`SchemaIndex.build()` is run offline (scripts/build_schema_index.py). The search methods are
what the MCP tools `search_schema` and `get_few_shots` call. The Chroma client, the embedder
and the catalog are passed in, so nothing here reads settings or builds its own clients.
"""

import hashlib
from typing import Any

import chromadb
import httpx
import ollama
from chromadb import Collection
from chromadb.errors import ChromaError
from langchain_core.embeddings import Embeddings

from multi_agenti_conversation_system.catalog import SchemaCatalog
from multi_agenti_conversation_system.observability import span
from multi_agenti_conversation_system.vectorstore import get_collection

SCHEMA_COLLECTION = "schema_docs"
FEW_SHOT_COLLECTION = "sql_seed_examples"

# What a down, slow or misbehaving Chroma / Ollama raises. httpx errors are not subclasses of
# the built-in ConnectionError, so they must be listed explicitly.
BACKEND_ERRORS = (
    httpx.HTTPError,
    ConnectionError,
    TimeoutError,
    ChromaError,
    ollama.ResponseError,
    ValueError,
)


class SearchUnavailable(Exception):
    """Chroma or the embedding model could not answer."""


def _few_shot_id(question: str) -> str:
    # Stable across rebuilds, so re-running the build updates entries instead of duplicating.
    return "fewshot-" + hashlib.sha1(question.encode()).hexdigest()[:12]


class SchemaIndex:
    def __init__(
        self, client: chromadb.ClientAPI, embeddings: Embeddings, catalog: SchemaCatalog
    ):
        self._client = client
        self._embeddings = embeddings
        self._catalog = catalog
        self._collections: dict[str, Collection] = {}

    def _collection(self, name: str) -> Collection:
        # Fetched on first use, so the server can start while Chroma is still down.
        if name not in self._collections:
            self._collections[name] = get_collection(self._client, name)
        return self._collections[name]

    # ------------------------------------------------------------------ build (offline)
    def _upsert(self, name: str, ids, documents, metadatas) -> None:
        collection = self._collection(name)
        vectors = self._embeddings.embed_documents(documents)
        collection.upsert(
            ids=ids, documents=documents, metadatas=metadatas, embeddings=vectors
        )
        # Remove entries whose table/question no longer exists in the YAML files.
        stale = set(collection.get(include=[])["ids"]) - set(ids)
        if stale:
            collection.delete(ids=list(stale))

    def build(self) -> dict[str, int]:
        """(Re)build both collections from the catalog and return their sizes."""
        catalog = self._catalog
        pii = catalog.pii_columns()
        tables = list(catalog.tables)
        self._upsert(
            SCHEMA_COLLECTION,
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
        self._upsert(
            FEW_SHOT_COLLECTION,
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
        return {
            name: self._collection(name).count()
            for name in (SCHEMA_COLLECTION, FEW_SHOT_COLLECTION)
        }

    # ------------------------------------------------------------------ search (runtime)
    def _search(
        self, collection_name: str, question: str, k: int, domain: str | None
    ) -> list[dict[str, Any]]:
        attributes = {"collection": collection_name, "k": k, "domain": domain}
        try:
            with span("index.search", attributes) as s:
                collection = self._collection(collection_name)
                total = collection.count()
                if total == 0:
                    return []
                result = collection.query(
                    query_embeddings=[self._embeddings.embed_query(question)],
                    n_results=max(1, min(k, total)),
                    where={"domain": domain} if domain else None,
                    include=["documents", "metadatas", "distances"],
                )
                s.set_attribute("hits", len(result["ids"][0]))
        except BACKEND_ERRORS as exc:
            raise SearchUnavailable(type(exc).__name__) from exc
        return [
            {"document": doc, "metadata": meta, "similarity": round(1 - dist, 4)}
            for doc, meta, dist in zip(
                result["documents"][0],  # type: ignore[index]
                result["metadatas"][0],  # type: ignore[index]
                result["distances"][0],  # type: ignore[index]
                strict=True,
            )
        ]

    def search_schema(
        self, question: str, k: int = 4, domain: str | None = None
    ) -> list[dict[str, Any]]:
        """Tables most relevant to the question: [{table, domain, has_pii, document, similarity}]."""
        return [
            {
                **hit["metadata"],
                "document": hit["document"],
                "similarity": hit["similarity"],
            }
            for hit in self._search(SCHEMA_COLLECTION, question, k, domain)
        ]

    def search_few_shots(
        self, question: str, k: int = 3, domain: str | None = None
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
            for hit in self._search(FEW_SHOT_COLLECTION, question, k, domain)
        ]

    # ------------------------------------------------------------------ readiness
    def ping(self) -> bool:
        try:
            self._client.heartbeat()
        except BACKEND_ERRORS:
            return False
        return True

    def is_built(self) -> bool:
        """Both collections exist and hold documents, i.e. the index build has been run."""
        try:
            return all(
                self._collection(name).count() > 0
                for name in (SCHEMA_COLLECTION, FEW_SHOT_COLLECTION)
            )
        except BACKEND_ERRORS:
            return False
