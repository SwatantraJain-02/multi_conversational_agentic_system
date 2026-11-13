"""Phase C gate: prompts from Langfuse and traces that nest across the A2A boundary.

    uv run python scripts/smoke_tracing.py

Needs the Langfuse keys in .env and the prompt pushed first (`scripts/sync_prompts.py`).
It talks to Langfuse Cloud, so it takes a minute: cloud ingestion is asynchronous.

Part A, prompts: fetched live with a version and label; a bad key falls back to the local
YAML quickly; empty keys use the local copy; a missing label is flagged as a fallback.

Part B, traces: a root span -> A2A calls to the real echo agent (a subprocess) -> the
agent's own spans. The trace is read back from Langfuse and checked parent by parent:
    smoke-root > a2a.call:echo > agent:echo > echo-reply (generation, linked to the prompt)
It also checks that the resume half keeps the trace, that the prompt version reaches the
caller in `TaskResult.prompt_versions`, and that an e-mail in a task is masked in Langfuse.
"""

import asyncio
import json
import os
import secrets
import signal
import subprocess
import sys
import time

import httpx

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.client import AgentRegistry
from multi_agenti_conversation_system.a2a.contracts import TaskRequest
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.prompts import get_prompt

PROMPT = "plumbing/echo-reply"
LABEL = "development"  # what scripts/sync_prompts.py pushes
PORT = settings.ECHO_AGENT_PORT
URL = f"http://{settings.A2A_HOST}:{PORT}"
TOKEN = secrets.token_hex(16)
EMAIL = "ana.trujillo@example.com"
POLL_DEADLINE_S = 90
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


def fetch_in_subprocess(env_overrides: dict[str, str]) -> tuple[dict, float]:
    """get_prompt in a fresh process with different Langfuse settings, and how long it took."""
    code = (
        "import json;"
        "from multi_agenti_conversation_system.prompts import get_prompt;"
        f"p=get_prompt({PROMPT!r}, label={LABEL!r});"
        "print(json.dumps({'source':p.source,'version':p.version,'text':p.compile(task='x')}))"
    )
    started = time.monotonic()
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **env_overrides},
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    elapsed = time.monotonic() - started
    lines = [line for line in out.stdout.splitlines() if line.startswith("{")]
    return (json.loads(lines[-1]) if lines else {"error": out.stderr[-300:]}), elapsed


def part_a_prompts() -> int | None:
    print("Part A: prompts")
    live = get_prompt(PROMPT, label=LABEL)
    check(
        "fetched from Langfuse with a version",
        live.source == "langfuse" and (live.version or 0) >= 1,
        f"source={live.source} version={live.version} label={live.label}",
    )
    if live.source != "langfuse":
        print("      -> run `uv run python scripts/sync_prompts.py` first")
    check("live prompt renders", live.compile(task="hello") == "echo: hello")

    bad, took = fetch_in_subprocess({"LANGFUSE_SECRET_KEY": "sk-lf-not-the-key"})
    check(
        "bad Langfuse key -> local fallback, fast",
        bad.get("source") == "fallback" and bad.get("text") == "echo: x" and took < 30,
        f"{bad.get('source')} in {took:.1f}s",
    )
    empty, _ = fetch_in_subprocess(
        {"LANGFUSE_PUBLIC_KEY": "", "LANGFUSE_SECRET_KEY": ""}
    )
    check(
        "empty keys -> local copy (tracing off)",
        empty.get("source") == "local",
        str(empty.get("source")),
    )
    missing = get_prompt(PROMPT, label="no-such-label")
    check(
        "a missing label is reported as a fallback, not as success",
        missing.source == "fallback" and missing.version is None,
        f"source={missing.source}",
    )
    return live.version


def start_agent() -> subprocess.Popen:
    env = {**os.environ, "A2A_TOKEN": TOKEN, "PROMPT_LABEL": LABEL}
    proc = subprocess.Popen(
        [sys.executable, "-m", "multi_agenti_conversation_system.agents.echo.app"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(50):
        try:
            if httpx.get(f"{URL}/health", timeout=1).status_code == 200:
                return proc
        except httpx.HTTPError:
            time.sleep(0.2)
    proc.kill()
    raise SystemExit("echo agent did not start")


def req(task: str) -> TaskRequest:
    return TaskRequest(sender="smoke", skill="echo", task=task)


def read_trace(trace_id: str, want: dict[str, int]) -> list:
    """Observations of a trace, polled until the expected names have all arrived."""
    api = tracing.get_langfuse().api
    deadline = time.monotonic() + POLL_DEADLINE_S
    seen: list = []
    while time.monotonic() < deadline:
        seen = list(
            api.observations.get_many(
                trace_id=trace_id, limit=100, fields="core,basic,io,prompt"
            ).data
        )
        counts: dict[str, int] = {}
        for obs in seen:
            counts[obs.name] = counts.get(obs.name, 0) + 1
        if all(counts.get(name, 0) >= n for name, n in want.items()):
            return seen
        time.sleep(3)
    return seen


async def part_b_traces(version: int | None) -> None:
    print("\nPart B: one trace across the A2A boundary")
    registry = AgentRegistry([URL], token=TOKEN)
    proc = start_agent()
    try:
        await registry.start()
        with tracing.observation("smoke-root", as_type="span") as root:
            ctx = tracing.current_trace_ctx()
            trace_id = ctx["trace_id"]
            hello = await registry.call("echo", req("hello"))
            waiting = await registry.call("echo", req("ask: ship it"))
            resumed = await registry.resume(waiting, {"approved": True})
            masked = await registry.call("echo", req(f"mail {EMAIL}"))
            if root:
                root.update(output="done")
        print(f"      trace id {trace_id}")
        tag = f"{PROMPT}@v{version}" if version else ""
        check(
            "prompt version reaches the caller in TaskResult.prompt_versions",
            hello.prompt_versions == {PROMPT: tag},
            str(hello.prompt_versions),
        )
        check("resumed task also reports it", resumed.prompt_versions == {PROMPT: tag})
        check("the calls themselves worked", hello.ok and resumed.ok and masked.ok)
        await asyncio.to_thread(tracing.flush)
    finally:
        await registry.close()
        # SIGTERM, not SIGKILL: the agent's shutdown hook flushes its buffered spans.
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()

    print("      reading the trace back from Langfuse (cloud ingestion is async)...")
    want = {"smoke-root": 1, "a2a.call:echo": 4, "agent:echo": 4, "echo-reply": 3}
    observations = await asyncio.to_thread(read_trace, trace_id, want)
    by_id = {o.id: o for o in observations}
    named = lambda name: [o for o in observations if o.name == name]

    for name, n in want.items():
        check(
            f"{n} x '{name}' in the trace",
            len(named(name)) >= n,
            f"found {len(named(name))}",
        )
    roots = named("smoke-root")
    root_id = roots[0].id if roots else None

    def parent_name(obs) -> str | None:
        parent = by_id.get(obs.parent_observation_id)
        return parent.name if parent else None

    def all_children_of(name: str, parent: str | None) -> bool:
        spans = named(name)
        # An empty list must not pass: that would prove nothing.
        return bool(spans) and all(parent_name(o) == parent for o in spans)

    check(
        "every a2a.call span is a child of the root",
        bool(named("a2a.call:echo"))
        and all(o.parent_observation_id == root_id for o in named("a2a.call:echo")),
    )
    check(
        "every agent span is a child of an a2a.call span (incl. the resumed one)",
        all_children_of("agent:echo", "a2a.call:echo"),
    )
    check(
        "every generation is inside an agent span (context survives LangGraph nodes)",
        all_children_of("echo-reply", "agent:echo"),
    )
    generations = named("echo-reply")
    check(
        "generations are linked to the prompt version",
        bool(generations)
        and all(
            g.prompt_name == PROMPT and g.prompt_version == version for g in generations
        ),
        str({(g.prompt_name, g.prompt_version) for g in generations}),
    )
    dumped = json.dumps([(o.input, o.output) for o in observations], default=str)
    check(
        "an e-mail in a task is masked in Langfuse (marker present, address absent)",
        EMAIL not in dumped and "[REDACTED:email]" in dumped,
    )


async def main() -> None:
    if not tracing.enabled():
        raise SystemExit(
            "tracing is off: set the LANGFUSE_* keys and TRACING_ENABLED=true"
        )
    version = part_a_prompts()
    await part_b_traces(version)
    await asyncio.to_thread(tracing.shutdown)
    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
