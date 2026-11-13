"""Phase E gate: the Web agent answers real questions over A2A, with a real model.

    uv run python scripts/smoke_web_agent.py

Self-contained: starts its own Web MCP server (:15556) and Web agent (:17002) as
subprocesses on spare ports. Uses Tavily (a few searches per question), OpenRouter and
the local re-ranker, and takes a few minutes.

Three questions; for each it checks that the citations are real URLs the agent actually
read, that every [n] in the answer has a source, and that the chunk counts before and
after re-ranking are reported. A last step stops the Web MCP server and expects a clean
`unavailable`, not a crash (the controlled-failure demo).
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
from multi_agenti_conversation_system.config import settings

MCP_PORT, AGENT_PORT = 15556, 17002
AGENT_URL = f"http://{settings.A2A_HOST}:{AGENT_PORT}"
TOKEN = secrets.token_hex(16)
CALL_TIMEOUT_S = 300
QUESTIONS = [
    "What are the main trends in the global beverage market in 2026?",
    "What is the current market outlook for plant-based dairy alternatives?",
    "What recent news is there about sparkling water sales growth?",
]
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
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()


def req(question: str) -> TaskRequest:
    return TaskRequest(
        sender="smoke",
        skill="answer_web_question",
        task=question,
        constraints=TaskConstraints(timeout_s=CALL_TIMEOUT_S),
    )


async def main() -> None:
    web_mcp = agent = None
    registry = AgentRegistry([AGENT_URL], token=TOKEN)
    try:
        web_mcp = spawn(
            "multi_agenti_conversation_system.mcp.servers.web_server.server",
            {"WEB_MCP_PORT": str(MCP_PORT)},
            f"http://localhost:{MCP_PORT}/health",
        )
        agent = spawn(
            "multi_agenti_conversation_system.agents.web.app",
            {
                "WEB_MCP_PORT": str(MCP_PORT),
                "WEB_AGENT_PORT": str(AGENT_PORT),
                "A2A_TOKEN": TOKEN,
                "PROMPT_LABEL": "development",
            },
            f"{AGENT_URL}/health",
        )
        await registry.start()
        info = registry.agents()[0]
        check(
            "agent card discovered with its skill",
            info.up and "answer_web_question" in info.skills,
        )

        for number, question in enumerate(QUESTIONS, start=1):
            print(f"\n{number}. {question}")
            started = time.monotonic()
            result = await registry.call("web", req(question))
            print(f"      [{time.monotonic() - started:.1f}s, status={result.status}]")
            if result.error:
                print("      error:", result.error.message[:200])
            check_answer(result)

        print("\ncontrolled failure: the Web MCP server goes down")
        stop(web_mcp)
        web_mcp = None
        down = await registry.call(
            "web", req("What is the latest news about coffee prices?")
        )
        print(
            f"      [status={down.status}] {down.error.message[:120] if down.error else ''}"
        )
        check(
            "a dead Web MCP server gives `unavailable`, not a crash",
            down.error is not None and down.error.code == "unavailable",
            str(down.error.code if down.error else down.status),
        )
    finally:
        await registry.close()
        stop(agent)
        stop(web_mcp)

    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


def check_answer(result) -> None:
    data = result.data
    print("      queries:", data.get("queries"))
    print(
        f"      pages read={data.get('pages_read')}  chunks before re-rank={data.get('chunks_before')}"
        f"  after={data.get('chunks_after')}  context chars={data.get('context_chars')}"
        f"  rate-limited={data.get('rate_limited')}  confidence={data.get('confidence')}"
    )
    for c in result.citations:
        print(f"      [{c['n']}] {c['url']}")
    print("      answer:", result.answer[:240].replace("\n", " "))

    check("completed", result.ok, result.status)
    check("cites at least one source", bool(result.citations))
    urls = [c["url"] for c in result.citations]
    check(
        "every citation is a real http(s) URL",
        bool(urls) and all(re.match(r"https?://\S+\.\S+", u) for u in urls),
    )
    read = {s["url"] for s in data.get("sources_consulted", [])}
    check("every cited URL is one the agent actually used", set(urls) <= read)
    in_text = {
        int(n)
        for group in re.findall(r"\[(\d+(?:\s*,\s*\d+)*)\]", result.answer)
        for n in re.split(r"\s*,\s*", group)
    }
    check(
        "every [n] in the answer has a citation",
        in_text <= {c["n"] for c in result.citations},
        str(sorted(in_text)),
    )
    before, after = data.get("chunks_before", 0), data.get("chunks_after", 0)
    check(
        "chunk counts are reported and re-ranking narrowed them",
        before >= after >= 1,
        f"{before} -> {after}",
    )
    check(
        "read at least one page",
        data.get("pages_read", 0) >= 1 and not data.get("snippet_only"),
    )
    check(
        "confidence is high",
        data.get("confidence") == "high",
        str(data.get("confidence")),
    )
    check(
        "every prompt version is a real Langfuse version",
        bool(result.prompt_versions)
        and all(re.search(r"@v\d+$", v) for v in result.prompt_versions.values()),
        str(result.prompt_versions),
    )


if __name__ == "__main__":
    asyncio.run(main())
