"""Pinecone: one document index, one namespace per corpus, hybrid by design.

The index schema declares two ranking fields:
- `keywords`: a full-text (BM25) string field. It holds `retrieval.hybrid.tokenize(text)`
  joined by spaces, so identifiers like `unit_price` are indexed as `unit_price`, `unit`
  and `price` (Pinecone's own analyzer only splits on whitespace and punctuation).
- `embedding`: a dense vector we compute ourselves (`llm.get_embedding_model()`).

Every other field on a document (`text`, `domain`, `sql`, ...) is stored metadata,
returned via `include_fields` and filterable. Keyword and dense searches are separate
requests, fused client-side with RRF (`retrieval.hybrid`). A fresh upsert is not
instantly searchable (G14), and metadata values may not be None (G15).
"""

from collections.abc import Iterator, Sequence
from typing import Any

from pinecone import Pinecone, SchemaBuilder

from multi_agenti_conversation_system.config import settings

SCHEMA_DOCS = "schema_docs"
SQL_SEED_EXAMPLES = "sql_seed_examples"
USER_FACTS = "user_facts"
EPISODES = "episodes"
SEMANTIC_CACHE = "semantic_cache"

KEYWORDS_FIELD = "keywords"
EMBEDDING_FIELD = "embedding"
UPSERT_BATCH = 50

__index = None


def _client() -> Pinecone:
    if not settings.PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set")
    return Pinecone(api_key=settings.PINECONE_API_KEY)


def _schema(dimension: int) -> dict[str, Any]:
    return (
        SchemaBuilder()
        # Our tokenizer already lowercases, drops stopwords and strips plurals.
        .add_string_field(
            KEYWORDS_FIELD,
            full_text_search={"language": "en", "stemming": False, "stop_words": False},
        )
        .add_dense_vector_field(EMBEDDING_FIELD, dimension=dimension, metric="cosine")
        .build()
    )


def get_index():
    """The project index, created on first use with the embedding model's dimension.

    Cached only once it succeeds, so an outage at startup is retried on the next call.
    """
    global __index

    if __index is None:
        pc = _client()
        if not pc.has_index(settings.PINECONE_INDEX):
            # Imported here: the embedding model is only needed to size a new index.
            from multi_agenti_conversation_system.llm import get_embedding_model

            pc.indexes.create(
                name=settings.PINECONE_INDEX,
                schema=_schema(get_embedding_model().dimension()),
                deployment={
                    "deployment_type": "managed",
                    "cloud": settings.PINECONE_CLOUD,
                    "region": settings.PINECONE_REGION,
                },
            )
        __index = pc.Index(settings.PINECONE_INDEX)

    return __index


def index_dimension() -> int:
    """Size of the index's dense field (creating the index first if needed)."""
    return get_index().describe_index_stats().dimension


def upsert(namespace: str, documents: Sequence[dict[str, Any]]) -> None:
    """Write documents shaped `{"_id", "keywords", "embedding", ...metadata}`."""
    docs = [clean_metadata(d) for d in documents]
    index = get_index()
    for start in range(0, len(docs), UPSERT_BATCH):
        index.documents.upsert(
            namespace=namespace, documents=docs[start : start + UPSERT_BATCH]
        )


def keyword_search(
    namespace: str,
    keywords: str,
    top_k: int,
    filter: dict[str, Any] | None = None,
    include_fields: Sequence[str] | None = None,
):
    """BM25 over the `keywords` field. Returns Pinecone documents, best first."""
    if not keywords:
        return []
    return (
        get_index()
        .documents.search(
            namespace=namespace,
            top_k=top_k,
            score_by=[{"type": "text", "fields": [KEYWORDS_FIELD], "query": keywords}],
            filter=filter,
            include_fields=include_fields,
        )
        .matches
    )


def dense_search(
    namespace: str,
    vector: Sequence[float],
    top_k: int,
    filter: dict[str, Any] | None = None,
    include_fields: Sequence[str] | None = None,
):
    """Cosine similarity on the `embedding` field; the score already is the similarity."""
    return (
        get_index()
        .documents.search(
            namespace=namespace,
            top_k=top_k,
            score_by=[
                {
                    "type": "dense_vector",
                    "fields": [EMBEDDING_FIELD],
                    "values": list(vector),
                }
            ],
            filter=filter,
            include_fields=include_fields,
        )
        .matches
    )


def list_ids(namespace: str, prefix: str | None = None) -> Iterator[str]:
    """Every document id in a namespace (optionally with an id prefix), across pages."""
    for entry in get_index().documents.list(namespace=namespace, prefix=prefix):
        yield entry.id


def delete(namespace: str, ids: Sequence[str]) -> None:
    if ids:
        get_index().documents.delete(namespace=namespace, ids=list(ids))


def fetch(
    namespace: str,
    ids: Sequence[str] = (),
    filter: dict[str, Any] | None = None,
    limit: int = 100,
) -> dict[str, dict[str, Any]]:
    """Documents by id, or by filter, as `{id: {field: value}}` (the embedding is not returned).

    A fetch by id sees a fresh upsert at once; only *search* lags behind (G14). An id that
    does not exist is simply absent.
    """
    index = get_index()
    if ids:
        response = index.documents.fetch(namespace=namespace, ids=list(ids))
    else:
        response = index.documents.fetch(
            namespace=namespace, filter=filter, limit=limit
        )
    return {
        doc_id: {k: v for k, v in doc.to_dict().items() if k not in (EMBEDDING_FIELD,)}
        for doc_id, doc in response.documents.items()
    }


def update(namespace: str, doc_id: str, **fields: Any) -> None:
    """Change some fields of one document (the others, and its vectors, stay)."""
    get_index().documents.update(
        namespace=namespace, documents=[{"_id": doc_id, **clean_metadata(fields)}]
    )


def delete_where(namespace: str, filter: dict[str, Any]) -> None:
    get_index().documents.delete(namespace=namespace, filter=filter)


def build_filter(
    eq: dict[str, Any] | None = None, gt: dict[str, Any] | None = None
) -> dict[str, Any]:
    """An AND of exact matches and `>` comparisons.

    Unlike `eq_filter` it NEVER drops a field: a None in `eq` is an error. A silently
    dropped clause would widen a cache lookup to every entry (the product 100 vs 150 false
    hit), so "no value" must be written as an explicit "".
    """
    clauses: dict[str, Any] = {}
    for field, value in (eq or {}).items():
        if value is None:
            raise ValueError(f"filter field {field!r} is None")
        clauses[field] = {"$eq": value}
    for field, value in (gt or {}).items():
        if value is None:
            raise ValueError(f"filter field {field!r} is None")
        clauses.setdefault(field, {})["$gt"] = value
    return clauses


def eq_filter(**fields: Any) -> dict[str, Any] | None:
    """`{"field": {"$eq": value}}` for every non-None value, or None when there are none."""
    clauses = {
        key: {"$eq": value} for key, value in fields.items() if value is not None
    }
    return clauses or None


def clean_metadata(document: dict[str, Any]) -> dict[str, Any]:
    """Drop None values: Pinecone rejects null fields (G15)."""
    return {key: value for key, value in document.items() if value is not None}
