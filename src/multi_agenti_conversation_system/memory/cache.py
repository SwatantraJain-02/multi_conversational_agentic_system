"""The semantic cache: finished answers, returned as-is under strict conditions.

Pinecone namespace `semantic_cache`. Similarity is necessary but not sufficient: "details of
product 100" and "... product 150" score ~0.97. A hit needs ALL of:

  1. cosine similarity >= `CACHE_THRESHOLD` on the standalone question (a ranking, in code);
  2. `entity_key` equal (an exact filter: different ids can never match);
  3. `profile_hash` equal (a user who wants EUR never gets a USD-formatted answer);
  4. `data_version` equal to the current setting, and `expires_at` in the future;
  5. `prompt_version` equal (a prompt promoted in Langfuse never serves old answers).

Conditions 2-5 are one exact filter built by `vectorstore.build_filter`, which raises on a
None instead of dropping the clause, and the key fields are always strings ("" for "no
entities"). A dropped clause would widen the lookup to every entry: the false hit.

Never cached (the caller decides, `cacheable()` states the rule): partial or failed answers,
anything that went through human approval, direct chit-chat, follow-ups that could not be made
standalone. The cache is never injected into a prompt: a hit is the answer.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from multi_agenti_conversation_system import vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.retrieval.hybrid import keyword_query

logger = logging.getLogger(__name__)

NAMESPACE = vectorstore.SEMANTIC_CACHE
# Prompts whose output ends up in an answer. A new version of any of them changes the
# `prompt_version` of the cache, so the old answers stop matching.
ANSWER_PROMPTS = (
    "supervisor/plan",
    "sql/generate",
    "sql/explain",
    "web/plan",
    "web/summarise",
    "synthesis/external-question",
    "synthesis/compare",
)
FIELDS = (
    "text",
    "answer",
    "citations_json",
    "sql",
    "agents_json",
    "route",
    "created_at",
)


@dataclass(frozen=True)
class CacheHit:
    id: str
    question: str
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    sql: str | None = None
    agents: list[str] = field(default_factory=list)
    route: str = ""
    similarity: float = 0.0


def cache_id(
    question: str, entity_key: str, profile_hash: str, prompt_version: str
) -> str:
    raw = "\n".join(
        (question.strip().lower(), entity_key, profile_hash, prompt_version)
    )
    return "cache-" + hashlib.sha1(raw.encode()).hexdigest()[:20]


def prompt_version_of(tags: list[str]) -> str:
    return hashlib.sha1("|".join(sorted(tags)).encode()).hexdigest()[:10]


def lookup_filter(
    entity_key: str, profile_hash: str, prompt_version: str, now: float | None = None
) -> dict[str, Any]:
    """The exact-match part of a lookup. Every key is a string; none may be missing."""
    return vectorstore.build_filter(
        eq={
            "entity_key": entity_key,
            "profile_hash": profile_hash,
            "prompt_version": prompt_version,
            "data_version": settings.DATA_VERSION,
        },
        gt={"expires_at": int(time.time() if now is None else now)},
    )


def cacheable(
    *, partial: bool, approvals: list, intent: str, standalone_ok: bool, answer: str
) -> bool:
    """Only complete, successful, standalone answers that needed no human."""
    return bool(
        answer.strip()
        and not partial
        and not approvals
        and intent in ("sql", "web", "compare", "hybrid")
        and standalone_ok
    )


def ttl_seconds(agents: list[str]) -> int:
    """Answers that involve the web age quickly; database answers wait for DATA_VERSION."""
    uses_web = "web" in agents or "synthesis" in agents
    return settings.CACHE_TTL_WEB_S if uses_web else settings.CACHE_TTL_DB_S


class SemanticCache:
    def __init__(self, backend: Any = vectorstore, embedder: Any = None) -> None:
        self.backend = backend
        self._embedder = embedder

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from multi_agenti_conversation_system.llm import get_embedding_model

            self._embedder = get_embedding_model()
        return self._embedder

    async def lookup(
        self,
        question: str,
        entity_key: str,
        profile_hash: str,
        prompt_version: str,
        threshold: float | None = None,
        now: float | None = None,
    ) -> CacheHit | None:
        try:
            return await asyncio.to_thread(
                self._lookup,
                question,
                entity_key,
                profile_hash,
                prompt_version,
                threshold,
                now,
            )
        except Exception as exc:  # noqa: BLE001 - a cache failure is a miss, never an error
            logger.warning("cache lookup failed: %s: %s", type(exc).__name__, exc)
            return None

    def _lookup(
        self,
        question: str,
        entity_key: str,
        profile_hash: str,
        prompt_version: str,
        threshold: float | None,
        now: float | None,
    ) -> CacheHit | None:
        floor = settings.CACHE_THRESHOLD if threshold is None else threshold
        matches = self.backend.dense_search(
            NAMESPACE,
            self.embedder.embed_query(question),
            3,
            filter=lookup_filter(entity_key, profile_hash, prompt_version, now),
            include_fields=FIELDS,
        )
        for m in matches:  # best first
            if m.score < floor:
                return None
            return CacheHit(
                id=m.id,
                question=m.get("text", ""),
                answer=m.get("answer", ""),
                citations=json.loads(m.get("citations_json") or "[]"),
                sql=m.get("sql") or None,
                agents=json.loads(m.get("agents_json") or "[]"),
                route=m.get("route", ""),
                similarity=round(m.score, 4),
            )
        return None

    async def store(
        self,
        question: str,
        entity_key: str,
        profile_hash: str,
        prompt_version: str,
        *,
        answer: str,
        citations: list[dict[str, Any]],
        sql: str | None,
        agents: list[str],
        route: str,
        now: float | None = None,
    ) -> str:
        return await asyncio.to_thread(
            self._store,
            question,
            entity_key,
            profile_hash,
            prompt_version,
            answer,
            citations,
            sql,
            agents,
            route,
            now,
        )

    def _store(
        self,
        question: str,
        entity_key: str,
        profile_hash: str,
        prompt_version: str,
        answer: str,
        citations: list[dict[str, Any]],
        sql: str | None,
        agents: list[str],
        route: str,
        now: float | None,
    ) -> str:
        now = int(time.time() if now is None else now)
        doc_id = cache_id(question, entity_key, profile_hash, prompt_version)
        self.backend.upsert(
            NAMESPACE,
            [
                {
                    "_id": doc_id,
                    vectorstore.KEYWORDS_FIELD: keyword_query(question),
                    vectorstore.EMBEDDING_FIELD: self.embedder.embed_documents(
                        [question]
                    )[0],
                    "text": question,
                    "answer": answer,
                    "citations_json": json.dumps(citations),
                    "sql": sql,
                    "agents_json": json.dumps(agents),
                    "route": route,
                    # Always strings, "" for none: see the module docstring.
                    "entity_key": entity_key,
                    "profile_hash": profile_hash,
                    "prompt_version": prompt_version,
                    "data_version": settings.DATA_VERSION,
                    "expires_at": now + ttl_seconds(agents),
                    "created_at": now,
                }
            ],
        )
        return doc_id

    async def delete(self, doc_ids: list[str]) -> None:
        await asyncio.to_thread(self.backend.delete, NAMESPACE, doc_ids)
