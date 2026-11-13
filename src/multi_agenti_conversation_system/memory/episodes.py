"""Long-term episodic memory: what happened in a past conversation.

Pinecone namespace `episodes`, `kind=summary`, one document per thread (id = the thread id, so
re-writing it later in the same thread replaces it). Written every few turns and when a
thread ends (`Supervisor.end_thread`). Read by similarity for ONE user, newest threads
favoured; the current thread is excluded (the checkpoint already has it).

(The other episodic kind, learned SQL pairs, is `memory/sql_pairs.py`, in the same namespace
under `kind=sql_pair`.)  Retention (90 days) is not enforced here: a Should item.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from multi_agenti_conversation_system import vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.retrieval.hybrid import keyword_query

logger = logging.getLogger(__name__)

NAMESPACE = vectorstore.EPISODES
KIND = "summary"
DAY = 86_400


@dataclass(frozen=True)
class Episode:
    thread_id: str
    summary: str
    created_at: int
    score: float = 0.0

    def dated(self) -> str:
        day = datetime.fromtimestamp(self.created_at, UTC).date().isoformat()
        return f"{self.summary} (conversation on {day})"


def episode_id(thread_id: str) -> str:
    return f"episode-{thread_id}"


class EpisodeStore:
    def __init__(self, backend: Any = vectorstore, embedder: Any = None) -> None:
        self.backend = backend
        self._embedder = embedder

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from multi_agenti_conversation_system.llm import get_embedding_model

            self._embedder = get_embedding_model()
        return self._embedder

    async def write(
        self, thread_id: str, user_id: str, summary: str, now: float | None = None
    ) -> None:
        summary = " ".join(summary.split())[:1500]
        if not (thread_id and user_id and summary):
            return

        def run() -> None:
            self.backend.upsert(
                NAMESPACE,
                [
                    {
                        "_id": episode_id(thread_id),
                        vectorstore.KEYWORDS_FIELD: keyword_query(summary),
                        vectorstore.EMBEDDING_FIELD: self.embedder.embed_documents(
                            [summary]
                        )[0],
                        "text": summary,
                        "kind": KIND,
                        "user_id": user_id,
                        "thread_id": thread_id,
                        "created_at": int(time.time() if now is None else now),
                    }
                ],
            )

        await asyncio.to_thread(run)

    async def recall(
        self,
        user_id: str,
        question: str,
        exclude_thread: str = "",
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[Episode]:
        try:
            return await asyncio.to_thread(
                self._recall, user_id, question, exclude_thread, k, min_similarity
            )
        except Exception as exc:  # noqa: BLE001 - memory is a bonus, never a blocker
            logger.warning("episode recall failed: %s: %s", type(exc).__name__, exc)
            return []

    def _recall(
        self,
        user_id: str,
        question: str,
        exclude_thread: str,
        k: int | None,
        min_similarity: float | None,
    ) -> list[Episode]:
        if not user_id or not question.strip():
            return []
        k = k or settings.EPISODES_TOP_K
        floor = (
            settings.EPISODES_MIN_SIMILARITY
            if min_similarity is None
            else min_similarity
        )
        matches = self.backend.dense_search(
            NAMESPACE,
            self.embedder.embed_query(question),
            k + 3,
            filter=vectorstore.build_filter(eq={"user_id": user_id, "kind": KIND}),
            include_fields=("text", "thread_id", "created_at"),
        )
        return [
            Episode(
                m.get("thread_id", ""),
                m.get("text", ""),
                int(m.get("created_at", 0)),
                round(m.score, 4),
            )
            for m in matches
            if m.score >= floor and m.get("thread_id") != exclude_thread
        ][:k]

    async def delete_user(self, user_id: str) -> None:
        await asyncio.to_thread(
            self.backend.delete_where,
            NAMESPACE,
            vectorstore.build_filter(eq={"user_id": user_id, "kind": KIND}),
        )
