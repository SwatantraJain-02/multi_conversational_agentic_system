"""Phase H live gate: long-term memory and the semantic cache, end to end.

    uv run python scripts/smoke_memory.py

Starts the DB MCP server and the SQL agent on spare ports (15555, 17001) and talks to them
through `open_supervisor` (real model, Pinecone, Aiven, Langfuse). The cache is ON here (the
other smoke scripts switch it off), so every user id is unique to the run and everything the
run writes is deleted at the end: facts, episodes, profiles, cache entries, learned SQL pair.

Checks, in the order of the Phase H gate:
 1. a fact and a preference are learned from what the user says about themselves;
 2. a NEW thread of the same user recalls both; a different user sees neither;
 3. Python -> Java replaces the preference and keeps Python in history; Nordic -> APAC
    supersedes the fact and keeps Nordic in its history;
 4. the cache: a paraphrase hits; product 150 misses where 100 was cached (and reuses the
    learned SQL pair); a profile change misses;
 5. a conversation becomes an episode, recalled for that user and no other.

A fresh write is not instantly searchable (G14): every "then read it back" step polls.
"""

import asyncio
import os
import re
import secrets
import sys
import time
import uuid

from smoke_supervisor import spawn, stop

from multi_agenti_conversation_system import tracing, vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.memory import profile as profile_mod
from multi_agenti_conversation_system.memory import sql_pairs
from multi_agenti_conversation_system.memory.entities import entity_key
from multi_agenti_conversation_system.supervisor.memory_nodes import (
    current_prompt_version,
)
from multi_agenti_conversation_system.supervisor.service import open_supervisor

PORTS = {"db": 15555, "sql": 17001}
TOKEN = secrets.token_hex(16)
RUN = uuid.uuid4().hex[:6]
ALICE, BOB, CAROL, DAVE = (
    f"smoke-mem-{n}-{RUN}" for n in ("alice", "bob", "carol", "dave")
)
POLL_S, POLL_EVERY_S = 90, 3
failures: list[str] = []
cache_questions: list[str] = []  # for cleanup: the cache has no user id, only the text
sql_used: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


async def poll(what: str, probe, deadline_s: float = POLL_S):
    """Call `probe()` (async, returns a truthy value when ready) until it does, or give up."""
    end = time.monotonic() + deadline_s
    while time.monotonic() < end:
        if value := await probe():
            return value
        await asyncio.sleep(POLL_EVERY_S)
    print(f"      (gave up waiting for {what})")
    return None


async def ask(sup, thread, user, text):
    started = time.monotonic()
    turn = await sup.ask(thread, user, text)
    print(
        f"\n[{thread} {user.split('-')[2]}] {text}\n      [{time.monotonic() - started:.1f}s] "
        f"{turn.status} intent={turn.intent} agents={turn.agents_used} cache_hit={turn.cache_hit} "
        f"learned={turn.memory_changes}\n      answer: {turn.answer[:220]!r}"
    )
    if turn.sql_used:
        sql_used.append(turn.sql_used)
    return turn


async def scenario(url: str) -> None:
    async with open_supervisor([url]) as sup:
        ctx = sup.context
        t = lambda name: f"smoke-{name}-{RUN}"

        # 1. learn ----------------------------------------------------------------
        print("\n== 1. learning from what the user says about themselves")
        turn = await ask(
            sup,
            t("a1"),
            ALICE,
            "From now on please give me code examples in Java. Also, I manage the APAC region.",
        )
        check(
            "a lasting preference went to the profile",
            any(
                c.lower().startswith("profile: code_language")
                for c in turn.memory_changes
            ),
            str(turn.memory_changes),
        )
        check(
            "a fact about the user was stored",
            any(c.startswith("fact added") for c in turn.memory_changes),
            str(turn.memory_changes),
        )

        # 2. recall in a new thread, isolation between users -----------------------
        print("\n== 2. a new thread of the same user, and another user")
        found = await poll(
            "the fact to become searchable",
            lambda: ctx.facts.recall(ALICE, "Which region does the user manage?"),
        )
        check("the fact is searchable (after polling)", bool(found), str(found))
        turn = await ask(
            sup,
            t("a2"),
            ALICE,
            "Which programming language do I prefer for code examples, and which region do I manage?",
        )
        low = turn.answer.lower()
        check("the new thread recalls the preference", "java" in low, turn.answer[:100])
        check("...and the fact", "apac" in low, turn.answer[:100])
        turn = await ask(
            sup,
            t("b1"),
            BOB,
            "Which programming language do I prefer for code examples, and which region do I manage?",
        )
        low = turn.answer.lower()
        check(
            "a different user sees neither",
            "java" not in low and "apac" not in low,
            turn.answer[:100],
        )
        mine = await sup.what_i_know(BOB)
        check(
            "...and has no stored memory",
            not mine["profile"] and not mine["facts"],
            str(mine),
        )

        # 3. change over time ------------------------------------------------------
        print("\n== 3. memory changes over time")
        await ask(sup, t("c1"), CAROL, "I prefer code examples in Python.")
        await ask(
            sup, t("c1"), CAROL, "From now on, give me code examples in Java instead."
        )
        profile = await ctx.profiles.load(CAROL)
        entry = profile.get("code_language", {})
        check(
            "Python -> Java: the value is replaced and Python is in the history",
            entry.get("value", "").lower() == "java"
            and [h.lower() for h in entry.get("history", [])] == ["python"],
            str(entry),
        )
        await ask(
            sup, t("c2"), CAROL, "Please remember that I manage the Nordic region."
        )
        await ask(
            sup,
            t("c2"),
            CAROL,
            "Update: I now manage the APAC region, not Nordic anymore.",
        )

        async def region_facts():
            active = await ctx.facts.list_active(CAROL)
            regions = [
                f
                for f in active
                if "region" in f.attribute or "region" in f.text.lower()
            ]
            history = (
                await ctx.facts.history(CAROL, regions[0].attribute) if regions else []
            )
            return (active, regions, history) if regions and history else None

        # fetch by filter lags behind a write, like search (G14): poll
        found = await poll("the facts to be listable", region_facts)
        active, regions, history = found or (await ctx.facts.list_active(CAROL), [], [])
        check(
            "Nordic -> APAC: one active region fact, APAC",
            len(regions) == 1 and "apac" in regions[0].text.lower(),
            str([f.text for f in active]),
        )
        check(
            "...and the Nordic one is kept as superseded history",
            any("nordic" in f.text.lower() for f in history),
            str([f.text for f in history]),
        )

        # 4. the cache -------------------------------------------------------------
        print("\n== 4. the semantic cache")
        q100 = "Show the details of the product with product_id 10"
        q100b = "What are the details for product_id 10?"
        q150 = "Show the details of the product with product_id 20"
        cache_questions.extend([q100, q100b, q150])
        first = await ask(sup, t("d1"), DAVE, q100)
        check(
            "first ask runs the pipeline (no hit)",
            not first.cache_hit and "sql" in first.agents_used,
        )
        version = await current_prompt_version()
        stored = await poll(
            "the cache entry to become searchable",
            lambda: ctx.cache.lookup(
                q100,
                entity_key(q100),
                profile_mod.profile_hash({}),
                version,
                threshold=0.8,
            ),
        )
        check("the entry is searchable (after polling)", stored is not None)
        again = await ask(sup, t("d2"), DAVE, q100b)
        check(
            "a paraphrase hits and calls no agent",
            again.cache_hit and not again.agents_used and again.answer == first.answer,
        )
        other = await ask(sup, t("d3"), DAVE, q150)
        check(
            "product 20 misses where 10 was cached",
            not other.cache_hit and "sql" in other.agents_used,
        )
        check(
            "...and its SQL is about 20, not 10",
            bool(other.sql_used)
            and re.search(r"product_id\s*=\s*20\b", other.sql_used) is not None
            and re.search(r"\b10\b", other.sql_used) is None,
            str(other.sql_used),
        )
        pair = (
            vectorstore.fetch(
                vectorstore.EPISODES, ids=[sql_pairs.pair_id(first.sql_used)]
            )
            if first.sql_used
            else {}
        )
        check(
            "the product 10 query was learned as a SQL pair (reused for 20)",
            bool(pair),
        )
        changed = await ask(
            sup, t("d4"), DAVE, "From now on I want every amount in EUR. Thanks."
        )
        check(
            "a preference was learned",
            any(c.startswith("profile:") for c in changed.memory_changes),
        )
        after = await ask(sup, t("d5"), DAVE, q100b)
        check(
            "after a profile change the old cached answer is not used",
            not after.cache_hit,
        )

        # 5. episodes --------------------------------------------------------------
        print("\n== 5. episodes")
        await sup.end_thread(t("a2"), ALICE)
        episode = await poll(
            "the episode to become searchable",
            lambda: ctx.episodes.recall(
                ALICE,
                "which programming language and region did we talk about",
                min_similarity=0.3,
            ),
        )
        check(
            "a finished conversation is recalled for its user",
            bool(episode),
            str(episode),
        )
        others = await ctx.episodes.recall(
            BOB,
            "which programming language and region did we talk about",
            min_similarity=0.0,
        )
        check("...and not for another user", not others)
        check(
            "the current thread is excluded from its own recall",
            not await ctx.episodes.recall(
                ALICE,
                "which programming language and region",
                exclude_thread=t("a2"),
                min_similarity=0.0,
            ),
        )

        # cleanup ------------------------------------------------------------------
        print("\n== cleanup")
        for user in (ALICE, BOB, CAROL, DAVE):
            await ctx.facts.delete_user(user)
            await ctx.episodes.delete_user(user)
            await ctx.profiles.reset(user)
        for question in cache_questions:
            vectorstore.delete_where(
                vectorstore.SEMANTIC_CACHE, {"text": {"$eq": question}}
            )
        for sql in sql_used:
            sql_pairs.delete(sql)
        print(
            "  removed this run's facts, episodes, profiles, cache entries and SQL pairs"
        )


async def main() -> None:
    env = {
        "DB_MCP_PORT": str(PORTS["db"]),
        "SQL_AGENT_PORT": str(PORTS["sql"]),
        "A2A_TOKEN": TOKEN,
        "PROMPT_LABEL": "development",
        "EVAL_MODE": "false",  # the SQL agent learns pairs here; they are deleted below
    }
    settings.A2A_TOKEN, settings.PROMPT_LABEL, settings.EVAL_MODE = (
        TOKEN,
        "development",
        False,
    )
    url = f"http://{settings.A2A_HOST}:{PORTS['sql']}"
    pkg = "multi_agenti_conversation_system"
    procs = []
    try:
        procs.append(
            spawn(
                f"{pkg}.mcp.servers.db_server.server",
                env,
                f"http://localhost:{PORTS['db']}/health",
            )
        )
        procs.append(spawn(f"{pkg}.agents.sql.app", env, f"{url}/health"))
        print(f"run {RUN}, users {ALICE}, {BOB}, {CAROL}, {DAVE}")
        await scenario(url)
    finally:
        for proc in procs:
            stop(proc)
    await asyncio.to_thread(tracing.shutdown)
    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    asyncio.run(main())
