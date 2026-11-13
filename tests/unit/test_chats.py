"""The chat log: per-user listing, full transcripts, slimmed answer details."""

import pytest
from langgraph.store.memory import InMemoryStore

from multi_agenti_conversation_system.memory.chats import ChatLog, title_of

NOW = 1_800_000_000


def chat(question: str, answer: str = "An answer", **meta) -> list[dict]:
    return [
        {"role": "user", "content": question},
        {
            "role": "assistant",
            "content": answer,
            "meta": {"status": "answered", **meta},
        },
    ]


@pytest.fixture
def log():
    return ChatLog(InMemoryStore())


async def test_a_user_sees_only_their_own_chats_newest_first(log):
    await log.save("alice", "t1", chat("How many customers?"), now=NOW)
    await log.save("alice", "t2", chat("Beverage market trends"), now=NOW + 60)
    await log.save("bob", "t3", chat("Bob's question"), now=NOW + 120)
    chats = await log.list_chats("alice")
    assert [c["thread_id"] for c in chats] == ["t2", "t1"]
    assert chats[0]["title"] == "Beverage market trends" and chats[0]["messages"] == 2
    assert [c["thread_id"] for c in await log.list_chats("bob")] == ["t3"]
    # another user's thread id does not open their chat
    assert await log.load("bob", "t1") == []
    # nor does a user whose id starts like theirs, or one with a dot in it
    await log.save("alice.smith", "t4", chat("Dotted user"), now=NOW)
    assert [c["thread_id"] for c in await log.list_chats("alice")] == ["t2", "t1"]
    assert [c["thread_id"] for c in await log.list_chats("alice.smith")] == ["t4"]


async def test_saving_again_replaces_the_transcript_and_keeps_the_start_time(log):
    await log.save("a", "t", chat("First question"), now=NOW)
    longer = chat("First question") + chat("Second question", "Second answer")
    await log.save("a", "t", longer, now=NOW + 300)
    [only] = await log.list_chats("a")
    assert only["created_at"] == NOW and only["updated_at"] == NOW + 300
    assert only["title"] == "First question"
    assert [m["content"] for m in await log.load("a", "t")] == [
        "First question",
        "An answer",
        "Second question",
        "Second answer",
    ]


async def test_bulky_details_are_dropped_and_rendered_ones_kept(log):
    messages = chat(
        "Top products?",
        sql_used="SELECT 1",
        citations=[{"n": 1, "url": "https://example.com"}],
        results={"s1": {"data": {"rows": [[1]] * 100}}},
        timings={"s1": [0.0, 1.0]},
    )
    await log.save("a", "t", messages, now=NOW)
    meta = (await log.load("a", "t"))[1]["meta"]
    assert meta["sql_used"] == "SELECT 1" and meta["citations"][0]["n"] == 1
    assert "results" not in meta and "timings" not in meta
    assert "results" in messages[1]["meta"]  # the caller's copy is not modified


async def test_an_empty_chat_is_not_stored_and_delete_removes_one(log):
    await log.save("a", "empty", [], now=NOW)
    await log.save("a", "t", chat("Q"), now=NOW)
    assert [c["thread_id"] for c in await log.list_chats("a")] == ["t"]
    await log.delete("a", "t")
    assert await log.list_chats("a") == [] and await log.list_chats("") == []


def test_the_title_is_the_first_user_message_shortened():
    assert title_of(chat("  How   many customers?  ")) == "How many customers?"
    long = title_of(chat("word " * 40))
    assert len(long) <= 60 and long.endswith("…")
    assert title_of([]) == "Untitled chat"
