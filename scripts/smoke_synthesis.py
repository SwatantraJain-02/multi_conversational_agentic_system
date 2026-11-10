"""Phase F gate: the Synthesis agent combines SQL and Web findings, and delegates to the Web agent.

    uv run python scripts/smoke_synthesis.py

Self-contained: starts the DB and Web MCP servers, and the SQL, Web and Synthesis agents, as
subprocesses on spare ports (15555, 15556, 17001-17003). Real model, Tavily, Aiven, Langfuse;
takes several minutes. (`smoke_a2a.py` is the echo-agent plumbing check; this is the Phase F
driver, since the supervisor only arrives in Phase G.)

  (a) Synthesis is given the SQL result AND the Web result: one combined answer, no delegation.
  (b) Synthesis is given only the SQL result: it asks the Web agent itself, over A2A.
  (c) the Web agent is stopped, and (b) is repeated: a partial answer that names what is
      missing, not a failure.

The trace is read back from Langfuse to show the delegation edge
(agent:synthesis > a2a.call:web > agent:web) in (b), and its absence in (a).
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

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.client import AgentRegistry
from multi_agenti_conversation_system.a2a.contracts import (
    SynthesisData,
    TaskConstraints,
    TaskRequest,
)
from multi_agenti_conversation_system.config import settings

PORTS = {"db": 15555, "web_mcp": 15556, "sql": 17001, "web": 17002, "synthesis": 17003}
TOKEN = secrets.token_hex(16)
CALL_TIMEOUT_S = 300
POLL_DEADLINE_S = 90
SQL_QUESTION = "What is our total revenue from the Beverages category, by year?"
WEB_QUESTION = "What is the global beverage market trend in 2025 and 2026?"
USER_QUESTION = (
    "How does our beverage revenue compare with the global beverage market trend?"
)
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


def request(agent: str, skill: str, task: str, **data) -> TaskRequest:
    return TaskRequest(
        sender="smoke",
        skill=skill,
        task=task,
        data=SynthesisData(**data).model_dump(mode="json") if data else {},
        constraints=TaskConstraints(timeout_s=CALL_TIMEOUT_S),
    )


def show(label: str, result, started: float) -> None:
    print(f"      [{label}: {time.monotonic() - started:.1f}s, status={result.status}]")
    if result.error:
        print("      error:", result.error.message[:200])
    if result.answer:
        print("      answer:", result.answer[:300].replace("\n", " "))


def read_trace(trace_id: str, want: dict[str, int]) -> list:
    api = tracing.get_langfuse().api
    deadline = time.monotonic() + POLL_DEADLINE_S
    seen: list = []
    while time.monotonic() < deadline:
        seen = list(
            api.observations.get_many(
                trace_id=trace_id, limit=100, fields="core,basic"
            ).data
        )
        counts: dict[str, int] = {}
        for obs in seen:
            counts[obs.name] = counts.get(obs.name, 0) + 1
        if all(counts.get(name, 0) >= n for name, n in want.items()):
            return seen
        time.sleep(3)
    return seen


async def main() -> None:
    procs: dict[str, subprocess.Popen | None] = {}
    urls = {
        k: f"http://{settings.A2A_HOST}:{PORTS[k]}" for k in ("sql", "web", "synthesis")
    }
    registry = AgentRegistry(list(urls.values()), token=TOKEN)
    ports = {
        "DB_MCP_PORT": str(PORTS["db"]),
        "WEB_MCP_PORT": str(PORTS["web_mcp"]),
        "SQL_AGENT_PORT": str(PORTS["sql"]),
        "WEB_AGENT_PORT": str(PORTS["web"]),
        "SYNTHESIS_AGENT_PORT": str(PORTS["synthesis"]),
        "A2A_TOKEN": TOKEN,
        "PROMPT_LABEL": "development",
        "EVAL_MODE": "true",  # the SQL agent must not write learned pairs from a smoke run
    }
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
        procs["sql"] = spawn(f"{pkg}.agents.sql.app", ports, f"{urls['sql']}/health")
        procs["web"] = spawn(f"{pkg}.agents.web.app", ports, f"{urls['web']}/health")
        procs["synthesis"] = spawn(
            f"{pkg}.agents.synthesis.app", ports, f"{urls['synthesis']}/health"
        )
        await registry.start()
        check(
            "all three agents are discovered from their cards",
            all(a.up for a in registry.agents()),
            str([a.name for a in registry.agents()]),
        )
        card = next(a.card for a in registry.agents() if a.name == "synthesis")
        extension = card.capabilities.extensions[0].params
        check(
            "the card publishes the synthesis input schema",
            "internal"
            in str(extension["skills"]["synthesise_findings"]["data_schema"]),
        )

        with tracing.observation("smoke-synthesis-root", as_type="span"):
            trace_id = tracing.current_trace_ctx()["trace_id"]
            await scenarios(registry, procs)
        print(f"      trace id {trace_id}")
        await asyncio.to_thread(tracing.flush)
    finally:
        await registry.close()
        for proc in procs.values():
            stop(proc)

    await check_trace(trace_id)
    await asyncio.to_thread(tracing.shutdown)
    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


async def scenarios(registry: AgentRegistry, procs: dict) -> None:
    print("\ninputs: the SQL agent and the Web agent each answer their half")
    started = time.monotonic()
    sql = await registry.call(
        "sql", request("sql", "answer_sql_question", SQL_QUESTION)
    )
    if sql.status == "input_required":  # a supervisor must handle this on any SQL step
        print(
            "      (the SQL agent asked for approval: approving, as a supervisor would)"
        )
        sql = await registry.resume(sql, {"approved": True})
    show("sql", sql, started)
    started = time.monotonic()
    web = await registry.call(
        "web", request("web", "answer_web_question", WEB_QUESTION)
    )
    show("web", web, started)
    check(
        "both inputs are usable",
        sql.ok and web.ok and bool(web.citations) and bool(sql.sql),
    )
    if not (sql.ok and web.ok):
        return

    print("\n(a) both results given: one combined answer, no delegation")
    started = time.monotonic()
    a = await registry.call(
        "synthesis",
        request(
            "synthesis",
            "synthesise_findings",
            USER_QUESTION,
            internal=sql,
            external=web,
        ),
    )
    show("a", a, started)
    check("completed", a.ok, a.status)
    check(
        "did NOT delegate (it had external data)",
        a.data.get("delegated") is False and not a.data.get("partial"),
        str(a.data.get("delegated")),
    )
    check("the SQL used is kept", a.sql == sql.sql)
    web_urls = {c["url"] for c in web.citations}
    check(
        "citations are the Web agent's, none invented",
        bool(a.citations) and {c["url"] for c in a.citations} <= web_urls,
    )
    in_text = {
        int(n)
        for g in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", a.answer)
        for n in re.split(r"\s*,\s*", g)
    }
    check(
        "every [n] in the answer has a citation",
        in_text <= {c["n"] for c in a.citations},
        str(sorted(in_text)),
    )
    check(
        "prompt versions of all three agents, all real versions",
        {"sql/generate", "web/summarise", "synthesis/compare"} <= set(a.prompt_versions)
        and all(re.search(r"@v\d+$", v) for v in a.prompt_versions.values()),
        str(a.prompt_versions),
    )

    print("\n(b) only the SQL result given: Synthesis asks the Web agent itself")
    started = time.monotonic()
    b = await registry.call(
        "synthesis",
        request("synthesis", "synthesise_findings", USER_QUESTION, internal=sql),
    )
    show("b", b, started)
    print("      question it sent to the Web agent:", b.data.get("external_question"))
    check(
        "completed, and it delegated",
        b.ok and b.data.get("delegated") is True,
        str(b.data.get("delegated")),
    )
    check(
        "a full answer, nothing missing",
        b.data.get("partial") is False and not b.data.get("missing"),
    )
    check(
        "web citations came back through the delegation",
        bool(b.citations)
        and all(re.match(r"https?://", c["url"]) for c in b.citations),
    )
    check(
        "the external question is about the outside world only",
        "our" not in (b.data.get("external_question") or "").lower().split(),
    )

    print("\n(c) the Web agent is stopped, then (b) again")
    stop(procs["web"])
    procs["web"] = None
    started = time.monotonic()
    c = await registry.call(
        "synthesis",
        request("synthesis", "synthesise_findings", USER_QUESTION, internal=sql),
    )
    show("c", c, started)
    check("still completed (graceful degradation)", c.ok, c.status)
    check(
        "flagged partial, with the missing piece named",
        c.data.get("partial") is True and c.data.get("missing") == ["external"],
        str(c.data.get("missing")),
    )
    check(
        "the answer itself says what is missing",
        "external" in c.answer.lower() and "could not be included" in c.answer,
    )
    check("the SQL used is still kept", c.sql == sql.sql and c.citations == [])


async def check_trace(trace_id: str) -> None:
    print("\ntrace: reading it back from Langfuse")
    want = {"agent:synthesis": 3, "a2a.call:web": 3, "agent:web": 2}
    observations = await asyncio.to_thread(read_trace, trace_id, want)
    by_id = {o.id: o for o in observations}

    def parent(o):
        return by_id.get(o.parent_observation_id)

    delegated = [
        o
        for o in observations
        if o.name == "a2a.call:web"
        and parent(o)
        and parent(o).name == "agent:synthesis"
    ]
    check(
        "Synthesis's own A2A calls to the Web agent are in the trace: (b) and (c), not (a)",
        len(delegated) == 2,
        f"found {len(delegated)}",
    )
    delegated_ids = {o.id for o in delegated}
    served = [
        o
        for o in observations
        if o.name == "agent:web" and o.parent_observation_id in delegated_ids
    ]
    check(
        "the Web agent's span nests under the delegated call: the delegation edge",
        len(served) == 1,
        f"found {len(served)} (only (b) had a live Web agent)",
    )
    synthesis_spans = [o for o in observations if o.name == "agent:synthesis"]
    check(
        "every agent:synthesis span hangs under an a2a.call:synthesis from the root",
        len(synthesis_spans) >= 3
        and all(
            parent(o) and parent(o).name == "a2a.call:synthesis"
            for o in synthesis_spans
        ),
    )


if __name__ == "__main__":
    asyncio.run(main())
