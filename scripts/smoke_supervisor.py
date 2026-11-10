"""Phase G gate: one conversation through the real supervisor, agents, model and Aiven checkpointer.

    uv run python scripts/smoke_supervisor.py

Starts the DB and Web MCP servers and the SQL, Web and Synthesis agents as subprocesses on
spare ports (15555, 15556, 17001-17003), then talks to them through `open_supervisor` on one
thread. Real model, Tavily, Aiven (checkpoints) and Langfuse; it takes 10+ minutes.

The conversation: a web question, a SQL question, a follow-up that only makes sense with the
history, a hybrid and a compare question, a direct reply, an approval (approved, then
declined), a restart (a brand new supervisor on the same thread), history trimming, and
finally the Web agent stopped: the answers must be partial but graceful.
"""

import asyncio
import os
import re
import secrets
import signal
import subprocess
import sys
import time
import uuid

import httpx

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.supervisor.service import open_supervisor

PORTS = {"db": 15555, "web_mcp": 15556, "sql": 17001, "web": 17002, "synthesis": 17003}
TOKEN = secrets.token_hex(16)
THREAD = f"smoke-{uuid.uuid4().hex[:8]}"
USER = "smoke-user"
HISTORY_LIMIT, HISTORY_KEEP = 14, 4  # low, so trimming happens during the run
POLL_DEADLINE_S = 90
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


def spawn(
    module: str, env: dict[str, str], health: str, wait_s: int = 90
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(wait_s * 2):
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
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


async def ask(supervisor, label: str, text: str, **kw):
    print(f"\n{label}: {text}")
    started = time.monotonic()
    turn = await supervisor.ask(THREAD, USER, text, **kw)
    describe(turn, started)
    return turn


def describe(turn, started: float) -> None:
    shape = " ".join(
        f"{s['id']}:{s['agent']}"
        + (f"<-{','.join(s['depends_on'])}" if s["depends_on"] else "")
        for s in turn.plan
    )
    print(
        f"      [{time.monotonic() - started:.1f}s] {turn.status}  intent={turn.intent}  plan=[{shape}]  agents={turn.agents_used}  partial={turn.partial}"
    )
    for s in turn.plan:
        print(f"        {s['id']} {s['agent']}: {s['task'][:110]}")
    if turn.approval:
        print(
            "      approval:",
            {k: v for k, v in turn.approval.items() if k in ("sql", "pii_columns")},
        )
    print(
        "      answer:",
        (turn.answer or "(waiting for approval)")[:260].replace("\n", " "),
    )


async def conversation(urls: list[str], web_proc: dict) -> None:
    async with open_supervisor(
        urls, history_max_messages=HISTORY_LIMIT, history_keep_last=HISTORY_KEEP
    ) as supervisor:
        # 1-2: one agent each
        t1 = await ask(
            supervisor,
            "1 (web)",
            "What are the main trends in the global beverage market in 2026?",
        )
        check(
            "web question: planned as web, answered with cited URLs",
            t1.intent == "web"
            and t1.agents_used == ["web"]
            and bool(t1.citations)
            and t1.status == "answered",
            f"intent={t1.intent}",
        )
        t2 = await ask(
            supervisor, "2 (sql)", "How many customers do we have in each country?"
        )
        check(
            "SQL question: planned as sql, the SQL used is returned",
            t2.intent == "sql" and t2.agents_used == ["sql"] and bool(t2.sql_used),
            f"intent={t2.intent}",
        )
        check(
            "every prompt version in the turn is a real Langfuse version",
            bool(t2.prompt_versions)
            and all(re.search(r"@v\d+$", v) for v in t2.prompt_versions.values()),
            str(t2.prompt_versions),
        )
        # 3: needs the history, and the history cannot answer it (orders, not customers):
        # the planner must rewrite it into a standalone SQL question.
        t3 = await ask(
            supervisor,
            "3 (follow-up)",
            "And how many orders did the customers in Germany place?",
        )
        sql_tasks = " ".join(s["task"] for s in t3.plan if s["agent"] == "sql")
        check(
            "the follow-up was rewritten as a standalone question (agents never see the chat)",
            "germany" in sql_tasks.lower() and "order" in sql_tasks.lower(),
            sql_tasks[:120],
        )
        check(
            "...and answered by SQL",
            t3.status == "answered" and "sql" in t3.agents_used,
        )
        # 4-5: more than one agent
        t4 = await ask(
            supervisor,
            "4 (hybrid)",
            "Give me our number of orders per year, and separately what is happening in the global beverage market, then combine the two into one summary.",
        )
        planned = {s["agent"] for s in t4.plan}
        check(
            "hybrid question uses SQL, Web and Synthesis",
            {"sql", "synthesis"} <= planned
            and ({"web"} <= planned or "web" in t4.agents_used),
            str(sorted(planned)),
        )
        independent = [
            s for s in t4.plan if s["agent"] in ("sql", "web") and not s["depends_on"]
        ]
        if len(independent) >= 2:
            a, b = (t4.timings[s["id"]] for s in independent[:2])
            check(
                "independent steps ran in parallel (their time intervals overlap)",
                a[0] < b[1] and b[0] < a[1],
                f"{a} {b}",
            )
        else:
            print(
                "      (the planner chose the sql -> synthesis shape: Synthesis fetched the web side itself)"
            )
        t5 = await ask(
            supervisor,
            "5 (compare)",
            "How does our beverage revenue by year compare with the global beverage market trend?",
        )
        check(
            "compare question goes through Synthesis and is answered",
            "synthesis" in t5.agents_used
            and t5.status == "answered"
            and bool(t5.answer),
            str(t5.agents_used),
        )
        # 6: direct
        t6 = await ask(
            supervisor,
            "6 (direct)",
            "Thanks! What kinds of things can you help me with?",
        )
        check(
            "a direct question calls no agent",
            t6.intent == "direct" and t6.agents_used == [] and t6.status == "answered",
        )
        # 7: approval
        t7 = await ask(
            supervisor,
            "7 (approval)",
            "List the company names and contact names of our customers in Germany",
        )
        check(
            "a PII question pauses the turn for approval",
            t7.status == "needs_approval" and bool(t7.approval.get("pii_columns")),
            str(t7.approval and t7.approval.get("pii_columns")),
        )
        started = time.monotonic()
        t7b = await supervisor.resume(THREAD, approved=True, user_id=USER)
        describe(t7b, started)
        check(
            "after approval the query ran and the answer came back",
            t7b.status == "answered"
            and "sql" in t7b.agents_used
            and bool(t7b.sql_used),
        )
        t8 = await ask(
            supervisor,
            "8 (approval, declined)",
            "List the phone numbers of our customers in France",
        )
        check("pauses again", t8.status == "needs_approval")
        started = time.monotonic()
        t8b = await supervisor.resume(THREAD, approved=False, user_id=USER)
        describe(t8b, started)
        check(
            "declined: the answer says the query was not run",
            t8b.status == "answered" and "not run" in t8b.answer.lower(),
        )

        before = await supervisor.history(THREAD)
        check(
            "history was trimmed into a summary past the limit",
            bool(before["summary"]) and len(before["messages"]) <= HISTORY_LIMIT,
            f"{len(before['messages'])} messages, summary {len(before['summary'])} chars",
        )
        await check_session(t2.trace_id)

    # 9: a brand new supervisor (a restart) on the same thread
    print("\n9 (restart): a new supervisor, same thread")
    async with open_supervisor(
        urls, history_max_messages=HISTORY_LIMIT, history_keep_last=HISTORY_KEEP
    ) as supervisor:
        after = await supervisor.history(THREAD)
        check(
            "the conversation survived the restart (same messages, same summary)",
            after == before,
            f"{len(after['messages'])} messages",
        )
        t9 = await ask(
            supervisor,
            "9",
            "Which of the two countries I asked about earlier has more customers?",
        )
        check(
            "a follow-up that needs the earlier turns is answered from them",
            t9.status == "answered"
            and any(c in t9.answer for c in ("Germany", "France")),
            t9.answer[:100],
        )

        # 10: the Web agent goes away
        print("\n10 (controlled failure): the Web agent is stopped")
        stop(web_proc["proc"])
        web_proc["proc"] = None
        t10 = await ask(
            supervisor, "10a", "What is the latest news about coffee prices?"
        )
        check(
            "web question with the Web agent down: answered, partial, the gap named",
            t10.status == "answered" and t10.partial and "web" in t10.answer.lower(),
            t10.answer[:100],
        )
        t11 = await ask(
            supervisor,
            "10b",
            "How does our beverage revenue by year compare with the global beverage market trend?",
        )
        check(
            "compare with the Web agent down: still answered from our own data, flagged partial",
            t11.status == "answered" and t11.partial and bool(t11.sql_used),
            t11.answer[:100],
        )
        check(
            "...and the answer names what is missing",
            "external" in t11.answer.lower() or "web" in t11.answer.lower(),
        )


async def check_session(trace_id: str | None) -> None:
    """Turns of one conversation share a Langfuse session (CS3 demo 3: traces tied together)."""
    if not trace_id:
        check("session grouping", False, "tracing is off")
        return
    await asyncio.to_thread(tracing.flush)
    api = tracing.get_langfuse().api
    deadline = time.monotonic() + POLL_DEADLINE_S
    seen = []
    while time.monotonic() < deadline:
        seen = list(
            api.observations.get_many(
                trace_id=trace_id, limit=100, fields="core,basic"
            ).data
        )
        if any(o.name.startswith("a2a.call:") for o in seen):
            break
        await asyncio.sleep(3)
    by_id = {o.id: o for o in seen}
    turn = next((o for o in seen if o.name == "turn"), None)
    calls = [o for o in seen if o.name.startswith("a2a.call:")]
    check(
        "the turn's trace carries the conversation's session id",
        turn is not None and turn.session_id == THREAD,
        str(turn and turn.session_id),
    )
    check(
        "the agent calls hang under the turn span",
        bool(calls) and all(by_id.get(o.parent_observation_id) is turn for o in calls),
        f"{len(calls)} call(s)",
    )


async def main() -> None:
    procs: dict[str, subprocess.Popen | None] = {}
    web_proc: dict[str, subprocess.Popen | None] = {}
    ports = {
        "DB_MCP_PORT": str(PORTS["db"]),
        "WEB_MCP_PORT": str(PORTS["web_mcp"]),
        "SQL_AGENT_PORT": str(PORTS["sql"]),
        "WEB_AGENT_PORT": str(PORTS["web"]),
        "SYNTHESIS_AGENT_PORT": str(PORTS["synthesis"]),
        "A2A_TOKEN": TOKEN,
        "PROMPT_LABEL": "development",
        "EVAL_MODE": "true",  # a smoke run must not write learned SQL pairs
    }
    # This process talks to the agents with the same token and the development prompts.
    settings.A2A_TOKEN, settings.PROMPT_LABEL = TOKEN, "development"
    settings.EVAL_MODE = True  # no semantic cache: every question must reach its agents
    urls = [
        f"http://{settings.A2A_HOST}:{PORTS[k]}" for k in ("sql", "web", "synthesis")
    ]
    pkg = "multi_agenti_conversation_system"
    try:
        procs["db"] = spawn(
            f"{pkg}.mcp.servers.db_server.server",
            ports,
            f"http://localhost:{PORTS['db']}/health",
        )
        procs["web_mcp"] = spawn(
            f"{pkg}.mcp.servers.web_server.server",
            ports,
            f"http://localhost:{PORTS['web_mcp']}/health",
        )
        procs["sql"] = spawn(f"{pkg}.agents.sql.app", ports, f"{urls[0]}/health")
        web_proc["proc"] = spawn(f"{pkg}.agents.web.app", ports, f"{urls[1]}/health")
        procs["synthesis"] = spawn(
            f"{pkg}.agents.synthesis.app", ports, f"{urls[2]}/health"
        )
        print(f"thread {THREAD}")
        await conversation(urls, web_proc)
    finally:
        for proc in [*procs.values(), web_proc.get("proc")]:
            stop(proc)
    await asyncio.to_thread(tracing.shutdown)
    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
