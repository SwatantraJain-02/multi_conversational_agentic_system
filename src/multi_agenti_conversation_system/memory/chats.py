"""Chat history: every conversation a user had, as the UI showed it, to list and reopen.

Stored in the same LangGraph store as the profile (`AsyncPostgresStore`, no vector index):
namespace ("chats", user_id), key = thread_id, value
{title, created_at, updated_at, messages: [{role, content, meta}]}.

Why not read the checkpoint instead: it is the model's working memory, not a transcript.
`summarize_history` removes old messages from it (they live on only in the summary), it has
none of the answer details the UI shows (sources, SQL, prompt versions), and it is keyed by
thread only, so it cannot say which chats belong to a user. This log is per user by
construction: one user's listing never reads another user's namespace.

A message's `meta` keeps what the UI renders and what feedback needs; the bulky per-step
results (rows, raw agent output) and timings are dropped before storing.
"""

import time
from typing import Any

from multi_agenti_conversation_system.memory.profile import store_label

TITLE_CHARS = 60
LIST_LIMIT = 100
META_DROP = ("results", "timings", "approval", "plan")


def namespace(user_id: str) -> tuple[str, str]:
    return ("chats", store_label(user_id))


def title_of(messages: list[dict[str, Any]]) -> str:
    """The first user message, shortened: what the chat was about."""
    first = next((m["content"] for m in messages if m.get("role") == "user"), "")
    first = " ".join(str(first).split())
    if len(first) <= TITLE_CHARS:
        return first or "Untitled chat"
    return first[: TITLE_CHARS - 1].rstrip() + "…"


def slim(message: dict[str, Any]) -> dict[str, Any]:
    out = {"role": message.get("role", ""), "content": message.get("content", "")}
    if meta := message.get("meta"):
        out["meta"] = {k: v for k, v in meta.items() if k not in META_DROP}
    return out


class ChatLog:
    """Async access to the chat log in a LangGraph store (tests pass an `InMemoryStore`)."""

    def __init__(self, store: Any) -> None:
        self.store = store

    async def save(
        self,
        user_id: str,
        thread_id: str,
        messages: list[dict[str, Any]],
        now: float | None = None,
    ) -> None:
        """Replace the stored transcript of one chat. An empty chat is not stored."""
        if not (user_id and thread_id and messages):
            return
        now = int(time.time() if now is None else now)
        old = await self.store.aget(namespace(user_id), thread_id)
        await self.store.aput(
            namespace(user_id),
            thread_id,
            {
                "title": title_of(messages),
                "created_at": old.value["created_at"] if old else now,
                "updated_at": now,
                "messages": [slim(m) for m in messages],
            },
        )

    async def list_chats(self, user_id: str) -> list[dict[str, Any]]:
        """The user's chats, most recently active first, without their messages."""
        if not user_id:
            return []
        items = await self.store.asearch(namespace(user_id), limit=LIST_LIMIT)
        chats = [
            {
                "thread_id": item.key,
                "title": item.value.get("title", "Untitled chat"),
                "created_at": item.value.get("created_at", 0),
                "updated_at": item.value.get("updated_at", 0),
                "messages": len(item.value.get("messages", [])),
            }
            for item in items
        ]
        return sorted(chats, key=lambda c: c["updated_at"], reverse=True)

    async def load(self, user_id: str, thread_id: str) -> list[dict[str, Any]]:
        """The messages of one of the user's chats; empty if it is not theirs or not stored."""
        if not (user_id and thread_id):
            return []
        item = await self.store.aget(namespace(user_id), thread_id)
        return list(item.value.get("messages", [])) if item else []

    async def delete(self, user_id: str, thread_id: str) -> None:
        await self.store.adelete(namespace(user_id), thread_id)
