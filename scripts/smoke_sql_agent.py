"""Phase D gate: the SQL agent answers real questions over A2A, with a real model.

    uv run python scripts/smoke_sql_agent.py

Self-contained: starts its own DB MCP server (:15555) and SQL agent (:17001) as
subprocesses on spare ports, so the current code is what gets tested and any servers you
already run are left alone. Uses OpenRouter, Aiven, Pinecone and Langfuse, and takes a few
minutes (free-tier models are slow).

Five questions: a plain one, a PII question approved, the same one declined, a request to
delete data, and a single-customer PII lookup. Plus a harder question to see whether the
model corrects itself on its own: whether it does is reported, not asserted (a live model
can't be made to fail on cue; the loop itself is proven in tests/unit/test_sql_agent.py).

The agent learns the plain question's SQL into the shared Pinecone namespace `episodes`;
this script deletes every pair it caused.
"""

import asyncio
import os
import re
import secrets
import signal
import subprocess
import sys
import time

import httpx

from multi_agenti_conversation_system.a2a.client import AgentRegistry
from multi_agenti_conversation_system.a2a.contracts import TaskConstraints, TaskRequest
from multi_agenti_conversation_system.agents.sql.db_tools import DbTools
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.mcp.client import open_mcp
from multi_agenti_conversation_system.memory import sql_pairs

DB_PORT, AGENT_PORT = 15555, 17001
AGENT_URL = f"http://{settings.A2A_HOST}:{AGENT_PORT}"
TOKEN = secrets.token_hex(16)
CALL_TIMEOUT_S = 240
failures: list[str] = []
learned_sql: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


def spawn(module: str, env: dict[str, str], health: str) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(120):
        try:
            if httpx.get(health, timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.5)
    proc.kill()
    raise SystemExit(f"{module} did not become healthy")


def stop(proc: subprocess.Popen | None) -> None:
    if proc:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def req(question: str) -> TaskRequest:
    return TaskRequest(
        sender="smoke",
        skill="answer_sql_question",
        task=question,
        constraints=TaskConstraints(timeout_s=CALL_TIMEOUT_S),
    )


async def timed(label: str, coro):
    started = time.monotonic()
    result = await coro
    print(f"      [{label}: {time.monotonic() - started:.1f}s, status={result.status}]")
    return result


def show(result) -> None:
    if result.sql:
        print("      sql:", re.sub(r"\s+", " ", result.sql)[:160])
    if result.answer:
        print("      answer:", result.answer[:160].replace("\n", " "))
    if result.error:
        print("      error:", result.error.message[:200])


async def orders_count(tools: DbTools) -> int:
    reply = await tools.run_select_query("SELECT COUNT(*) AS n FROM orders", 1)
    return int(reply["rows"][0][0])


async def main() -> None:
    settings.DB_MCP_PORT = (
        DB_PORT  # this process's MCP client talks to our spare server
    )
    db_server = agent = None
    registry = AgentRegistry([AGENT_URL], token=TOKEN)
    try:
        db_server = spawn(
            "multi_agenti_conversation_system.mcp.servers.db_server.server",
            {"DB_MCP_PORT": str(DB_PORT)},
            f"http://localhost:{DB_PORT}/health",
        )
        agent = spawn(
            "multi_agenti_conversation_system.agents.sql.app",
            {
                "DB_MCP_PORT": str(DB_PORT),
                "SQL_AGENT_PORT": str(AGENT_PORT),
                "A2A_TOKEN": TOKEN,
                "PROMPT_LABEL": "development",
                "EVAL_MODE": "false",
            },
            f"{AGENT_URL}/health",
        )
        async with open_mcp(["db"]) as mcp_tools:
            tools = DbTools({t.name: t for t in mcp_tools["db"]})
            snapshot = await tools.snapshot()
            await run_questions(registry, tools, snapshot.schema_version)
    finally:
        for sql in learned_sql:
            sql_pairs.delete(sql)
        await registry.close()
        stop(agent)
        stop(db_server)

    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


async def run_questions(
    registry: AgentRegistry, tools: DbTools, schema_version: str
) -> None:
    await registry.start()
    card_ok = (
        registry.agents()[0].up and "answer_sql_question" in registry.agents()[0].skills
    )
    check("agent card discovered with its skill", card_ok)
    orders_before = await orders_count(tools)

    print("\n1. a plain question")
    plain = "How many customers do we have in each country?"
    r1 = await timed("plain", registry.call("sql", req(plain)))
    show(r1)
    check(
        "completed with rows and the SQL used",
        r1.ok and r1.data.get("row_count", 0) > 0 and bool(r1.sql),
    )
    check("no approval needed (no PII, small result)", r1.status == "completed")
    check(
        "every prompt version is a real Langfuse version",
        bool(r1.prompt_versions)
        and all(re.search(r"@v\d+$", v) for v in r1.prompt_versions.values()),
        str(r1.prompt_versions),
    )
    print(
        f"      attempts={r1.data.get('attempts')} (a value above 1 means it corrected itself)"
    )
    pair_id = r1.data.get("learned_pair")
    check("the validated pair was learned", bool(pair_id), str(pair_id))
    if r1.sql:
        learned_sql.append(r1.sql)

    print("\n2. a PII question, approved")
    pii_question = "Show the company name and contact name of customers in Germany"
    r2 = await timed("pii ask", registry.call("sql", req(pii_question)))
    show(r2)
    check("pauses for approval", r2.status == "input_required", r2.status)
    check(
        "the payload names the SQL and the PII columns",
        bool(r2.data.get("sql")) and bool(r2.data.get("pii_columns")),
        str(r2.data.get("pii_columns")),
    )
    r2b = await timed("pii approve", registry.resume(r2, {"approved": True}))
    show(r2b)
    check(
        "runs after approval and returns rows",
        r2b.ok and r2b.data.get("row_count", 0) > 0,
    )
    check("a PII query is not learned", r2b.data.get("learned_pair") is None)
    if r2b.sql:
        learned_sql.append(r2b.sql)

    print("\n3. the same question, declined")
    r3 = await timed("pii ask", registry.call("sql", req(pii_question)))
    check("pauses for approval again", r3.status == "input_required", r3.status)
    r3b = await timed("pii reject", registry.resume(r3, {"approved": False}))
    show(r3b)
    check(
        "declined: completed (not failed), nothing returned",
        r3b.ok and r3b.data.get("rejected") is True and not r3b.data.get("rows"),
    )

    print("\n4. a request to delete data")
    r4 = await timed(
        "delete", registry.call("sql", req("Delete all orders placed in 2024"))
    )
    show(r4)
    check(
        "refused or failed, never executed",
        (r4.ok and r4.data.get("refused") is True) or r4.status == "failed",
        f"status={r4.status} refused={r4.data.get('refused')}",
    )
    check("no rows came back", not r4.data.get("rows"))
    check("the orders table is untouched", await orders_count(tools) == orders_before)

    print("\n5. a single-customer PII lookup")
    r5 = await timed(
        "lookup",
        registry.call("sql", req("What is the phone number of customer VINET?")),
    )
    show(r5)
    check("pauses for approval", r5.status == "input_required", r5.status)
    columns = " ".join(r5.data.get("pii_columns", []))
    check("flags the phone column", "phone" in columns, columns)
    if r5.status == "input_required":
        await registry.resume(r5, {"approved": False})

    print("\n6. a harder question (does the model correct itself?)")
    r6 = await timed(
        "revenue",
        registry.call(
            "sql", req("Total revenue per product category in 2025, highest first")
        ),
    )
    show(r6)
    check("answered", r6.ok and r6.data.get("row_count", 0) > 0, r6.status)
    print(
        f"      attempts={r6.data.get('attempts')}  (1 = right first time, >1 = self-corrected live)"
    )
    if r6.sql:
        learned_sql.append(r6.sql)

    print("\nlearned pairs")
    found = []
    for _ in range(20):  # a fresh upsert isn't searchable at once (G14)
        found = await asyncio.to_thread(
            sql_pairs.search, "How many customers per country?", 3, schema_version, None
        )
        if any(p.sql.strip() == (r1.sql or "").strip() for p in found):
            break
        await asyncio.sleep(3)
    check(
        "the learned pair is retrievable (searched with a paraphrase)",
        any(p.sql.strip() == (r1.sql or "").strip() for p in found),
        f"{len(found)} pair(s) found",
    )


if __name__ == "__main__":
    asyncio.run(main())
