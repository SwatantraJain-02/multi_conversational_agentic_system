"""Long-term semantic memory: durable facts about a user ("manages the APAC region").

Pinecone namespace `user_facts`. Facts are keyed by ATTRIBUTE first, similarity second
(the same reason as the profile: "manages Nordic" and "manages APAC" are near-identical
sentences but a contradiction, not a duplicate). One *active* fact exists per
(user, attribute), at a deterministic id, so deciding what a new statement means is a plain
fetch by id: it works straight after a write, unlike search (G14).

    same attribute, same value   -> re-statement: refresh `last_confirmed_at`, `confidence`
    same attribute, other value  -> contradiction: the old fact is kept as `superseded`
                                    (linked both ways), the new one becomes the active fact
    new attribute                -> added

Nothing is deleted on a contradiction, so history survives; retrieval only returns
`status=active`, `expires_at > now`, and always for ONE user (`user_id` is in every filter).
A temporary fact has `expires_at = valid_until`; a stable one a far-future date, so one range
filter covers both.

Ranking: `similarity x (0.7 + 0.3 x freshness)`, `freshness = exp(-age_days / 180)`, with age
from `last_confirmed_at`. Facts are injected DATED, and tagged "(may be outdated)" past
180 days unconfirmed, so the model can hedge.

Not built (Should): similarity fallback for facts without an attribute (they dedupe on exact
normalised text only), consolidation of clusters, the per-user cap.
"""

import asyncio
import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from multi_agenti_conversation_system import vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.retrieval.hybrid import keyword_query

logger = logging.getLogger(__name__)

NAMESPACE = vectorstore.USER_FACTS
NEVER = 9_999_999_999  # `expires_at` of a stable fact
DAY = 86_400
STALE_AFTER_DAYS = 180
TEMPORARY_DEFAULT_DAYS = 30
Outcome = Literal["added", "restated", "superseded", "ignored"]


@dataclass(frozen=True)
class Fact:
    id: str
    text: str
    attribute: str
    value: str
    kind: str
    created_at: int
    last_confirmed_at: int
    score: float = 0.0

    def dated(self, now: float | None = None) -> str:
        """For a prompt: 'Manages APAC region (stated 2026-10-02)', plus a staleness tag."""
        now = time.time() if now is None else now
        stated = datetime.fromtimestamp(self.created_at, UTC).date().isoformat()
        stale = (now - self.last_confirmed_at) / DAY > STALE_AFTER_DAYS
        return f"{self.text} (stated {stated}{', may be outdated' if stale else ''})"


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40]


def user_hash(user_id: str) -> str:
    return hashlib.sha1(user_id.encode()).hexdigest()[:10]


def fact_id(user_id: str, attribute: str, text: str) -> str:
    """Id of the ACTIVE fact. Without an attribute the id comes from the text itself."""
    key = (
        _slug(attribute)
        if attribute
        else "misc-" + hashlib.sha1(_norm(text).encode()).hexdigest()[:10]
    )
    return f"fact-{user_hash(user_id)}-{key}"


def freshness(age_days: float) -> float:
    return math.exp(-max(age_days, 0.0) / 180)


def rank_score(similarity: float, last_confirmed_at: float, now: float) -> float:
    return similarity * (0.7 + 0.3 * freshness((now - last_confirmed_at) / DAY))


def _fact(doc_id: str, fields: dict[str, Any], score: float = 0.0) -> Fact:
    return Fact(
        id=doc_id,
        text=fields.get("text", ""),
        attribute=fields.get("attribute", ""),
        value=fields.get("value", ""),
        kind=fields.get("kind", "stable"),
        created_at=int(fields.get("created_at", 0)),
        last_confirmed_at=int(fields.get("last_confirmed_at", 0)),
        score=score,
    )


class FactStore:
    """Blocking Pinecone calls behind async methods. `backend` and `embedder` are injectable."""

    def __init__(self, backend: Any = vectorstore, embedder: Any = None) -> None:
        self.backend = backend
        self._embedder = embedder

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from multi_agenti_conversation_system.llm import get_embedding_model

            self._embedder = get_embedding_model()
        return self._embedder

    # ----------------------------------------------------------------------- writing
    async def remember(
        self,
        user_id: str,
        text: str,
        attribute: str = "",
        value: str = "",
        kind: str = "stable",
        valid_days: float | None = None,
        now: float | None = None,
    ) -> Outcome:
        return await asyncio.to_thread(
            self._remember, user_id, text, attribute, value, kind, valid_days, now
        )

    def _remember(
        self,
        user_id: str,
        text: str,
        attribute: str,
        value: str,
        kind: str,
        valid_days: float | None,
        now: float | None,
    ) -> Outcome:
        text = " ".join(text.split())[:300]
        if not user_id or not text:
            return "ignored"
        now = int(time.time() if now is None else now)
        attribute, value = _slug(attribute), value.strip()[:100]
        temporary = kind == "temporary"
        expires = (
            now + int((valid_days or TEMPORARY_DEFAULT_DAYS) * DAY)
            if temporary
            else NEVER
        )
        doc_id = fact_id(user_id, attribute, text)
        existing = self.backend.fetch(NAMESPACE, ids=[doc_id]).get(doc_id)
        live = existing and existing.get("status") == "active"

        if live and _norm(existing.get("value", "")) == _norm(value or text):
            self.backend.update(
                NAMESPACE,
                doc_id,
                last_confirmed_at=now,
                expires_at=expires,
                confidence=min(1.0, float(existing.get("confidence", 0.7)) + 0.1),
            )
            return "restated"

        supersedes = ""
        outcome: Outcome = "added"
        if (
            live
        ):  # a contradiction: keep the old fact as history, linked to its successor
            supersedes = f"{doc_id}-v{int(existing.get('created_at', now))}"
            self._write(
                supersedes,
                {**existing, "status": "superseded", "superseded_by": doc_id},
                existing.get("text", ""),
            )
            outcome = "superseded"
        self._write(
            doc_id,
            {
                "user_id": user_id,
                "attribute": attribute,
                "value": value or _norm(text),
                "text": text,
                "kind": "temporary" if temporary else "stable",
                "expires_at": expires,
                "status": "active",
                "supersedes": supersedes,
                "superseded_by": "",
                "created_at": now,
                "last_confirmed_at": now,
                "last_used_at": now,
                "use_count": 0,
                "confidence": 0.7,
            },
            text,
        )
        return outcome

    def _write(self, doc_id: str, fields: dict[str, Any], text: str) -> None:
        document = {
            **{k: v for k, v in fields.items() if k not in ("keywords", "embedding")},
            "_id": doc_id,
            vectorstore.KEYWORDS_FIELD: keyword_query(text),
            vectorstore.EMBEDDING_FIELD: self.embedder.embed_documents([text])[0],
        }
        self.backend.upsert(NAMESPACE, [document])

    # ----------------------------------------------------------------------- reading
    async def recall(
        self,
        user_id: str,
        question: str,
        k: int | None = None,
        min_similarity: float | None = None,
        now: float | None = None,
    ) -> list[Fact]:
        """The user's active, unexpired facts most relevant to the question."""
        try:
            return await asyncio.to_thread(
                self._recall, user_id, question, k, min_similarity, now
            )
        except Exception as exc:  # noqa: BLE001 - memory is a bonus, never a blocker
            logger.warning("fact recall failed: %s: %s", type(exc).__name__, exc)
            return []

    def _recall(
        self,
        user_id: str,
        question: str,
        k: int | None,
        min_similarity: float | None,
        now: float | None,
    ) -> list[Fact]:
        if not user_id or not question.strip():
            return []
        k = k or settings.FACTS_TOP_K
        floor = (
            settings.FACTS_MIN_SIMILARITY if min_similarity is None else min_similarity
        )
        now = time.time() if now is None else now
        matches = self.backend.dense_search(
            NAMESPACE,
            self.embedder.embed_query(question),
            max(k * 3, 10),
            filter=vectorstore.build_filter(
                eq={"user_id": user_id, "status": "active"}, gt={"expires_at": int(now)}
            ),
            include_fields=(
                "text",
                "attribute",
                "value",
                "kind",
                "created_at",
                "last_confirmed_at",
                "use_count",
            ),
        )
        ranked = sorted(
            (
                _fact(
                    m.id,
                    m.to_dict(),
                    rank_score(m.score, float(m.get("last_confirmed_at", 0)), now),
                )
                for m in matches
                if m.score >= floor
            ),
            key=lambda f: f.score,
            reverse=True,
        )[:k]
        for fact in ranked:  # best effort: a failed bump must not lose the recall
            try:
                self.backend.update(NAMESPACE, fact.id, last_used_at=int(now))
            except Exception:  # noqa: BLE001, S110
                pass
        return ranked

    async def list_active(self, user_id: str, now: float | None = None) -> list[Fact]:
        """Everything known about the user ('what do you know about me?'), newest first."""
        now = int(time.time() if now is None else now)

        def run() -> list[Fact]:
            docs = self.backend.fetch(
                NAMESPACE,
                filter=vectorstore.build_filter(
                    eq={"user_id": user_id, "status": "active"}, gt={"expires_at": now}
                ),
            )
            return sorted(
                (_fact(i, d) for i, d in docs.items()),
                key=lambda f: f.created_at,
                reverse=True,
            )

        return await asyncio.to_thread(run)

    async def history(self, user_id: str, attribute: str) -> list[Fact]:
        """The superseded versions of one attribute, oldest first."""

        def run() -> list[Fact]:
            docs = self.backend.fetch(
                NAMESPACE,
                filter=vectorstore.build_filter(
                    eq={
                        "user_id": user_id,
                        "attribute": _slug(attribute),
                        "status": "superseded",
                    }
                ),
            )
            return sorted(
                (_fact(i, d) for i, d in docs.items()), key=lambda f: f.created_at
            )

        return await asyncio.to_thread(run)

    # ---------------------------------------------------------------------- deleting
    async def forget(self, user_id: str, attribute: str) -> None:
        """Hard-delete a fact and its history ('forget X')."""
        await asyncio.to_thread(
            self.backend.delete_where,
            NAMESPACE,
            vectorstore.build_filter(
                eq={"user_id": user_id, "attribute": _slug(attribute)}
            ),
        )

    async def delete_user(self, user_id: str) -> None:
        await asyncio.to_thread(
            self.backend.delete_where,
            NAMESPACE,
            vectorstore.build_filter(eq={"user_id": user_id}),
        )
