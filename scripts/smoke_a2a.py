"""Check the A2A plumbing (Phase C step 5) against a real echo agent subprocess.

    uv run python scripts/smoke_a2a.py

Starts the echo agent with a random bearer token, then checks: card discovery, a call,
an input-required -> resume round trip (approve and reject), clean failures (crash, bad
skill, timeout), auth, and that a killed agent gives `unavailable` (and recovers).
"""

import asyncio
import os
import secrets
import signal
import subprocess
import sys
import time

import httpx

from multi_agenti_conversation_system.a2a.client import AgentRegistry
from multi_agenti_conversation_system.a2a.contracts import TaskConstraints, TaskRequest
from multi_agenti_conversation_system.config import settings

PORT = settings.ECHO_AGENT_PORT
URL = f"http://{settings.A2A_HOST}:{PORT}"
TOKEN = secrets.token_hex(16)
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


def start_agent() -> subprocess.Popen:
    env = {**os.environ, "A2A_TOKEN": TOKEN}
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


def req(task: str, skill: str = "echo", timeout_s: float | None = None) -> TaskRequest:
    return TaskRequest(
        sender="smoke",
        skill=skill,
        task=task,
        constraints=TaskConstraints(timeout_s=timeout_s),
    )


async def main() -> None:
    # The registry starts BEFORE the agent: an agent that isn't up yet must be found later.
    registry = AgentRegistry([URL], token=TOKEN)
    proc = None
    try:
        print("late start")
        await registry.start()
        check("agent down at startup is marked down", not registry.agents()[0].up)
        proc = start_agent()
        late = await registry.call("echo", req("hello"))
        check(
            "agent started after the registry is found and called", late.ok, late.answer
        )

        print("\ndiscovery and auth")
        await registry.refresh()
        info = registry.agents()[0]
        check("agent card discovered", info.up and info.name == "echo", str(info.name))
        check(
            "skills come from the card", registry.capabilities()[0]["skill"] == "echo"
        )
        async with httpx.AsyncClient() as anonymous:
            card = await anonymous.get(f"{URL}/.well-known/agent-card.json")
            check("card is public (no token)", card.status_code == 200)
            rpc = await anonymous.post(f"{URL}/", json={})
            check("JSON-RPC without a token -> 401", rpc.status_code == 401)

        print("\ncall")
        progress: list[str] = []
        result = await registry.call(
            "echo",
            req("hello"),
            trace_ctx={"trace_id": "t-1"},
            on_progress=lambda agent, text: progress.append(text),
        )
        check(
            "completed with the echoed answer",
            result.ok and result.answer == "echo: hello",
            result.answer,
        )
        check(
            "node progress streamed",
            progress == ["node: listen", "node: confirm", "node: reply"],
            str(progress),
        )

        print("\ninput-required -> resume")
        first = req("ask: ship it")
        waiting = await registry.call("echo", first, trace_ctx={"trace_id": "t-call"})
        check(
            "pauses in input_required",
            waiting.status == "input_required",
            waiting.status,
        )
        check("interrupt payload carried", waiting.data.get("task") == "ask: ship it")
        done = await registry.resume(
            waiting, {"approved": True}, trace_ctx={"trace_id": "t-resume"}
        )
        check(
            "resume approve completes",
            done.ok and done.answer == "echo: ask: ship it",
            done.answer,
        )
        check(
            "trace_ctx reaches the graph on the call",
            result.data["trace_ctx"] == {"trace_id": "t-1"},
            str(result.data["trace_ctx"]),
        )
        check(
            "the resume carries its own trace_ctx",
            done.data["trace_ctx"] == {"trace_id": "t-resume"},
            str(done.data["trace_ctx"]),
        )
        check(
            "the resumed run sees the original TaskRequest",
            done.data["task_id"] == first.task_id,
            done.data["task_id"],
        )
        waiting = await registry.call("echo", req("ask: again"))
        rejected = await registry.resume(waiting, {"approved": False})
        check(
            "resume reject is honoured",
            rejected.ok and rejected.answer == "rejected",
            rejected.answer,
        )
        again = await registry.resume(done, {"approved": True})
        check(
            "resuming a finished task -> bad_request",
            again.error and again.error.code == "bad_request",
        )

        print("\nclean failures")
        crashed = await registry.call("echo", req("crash"))
        check(
            "crash -> failed/internal",
            crashed.error and crashed.error.code == "internal",
            str(crashed.error),
        )
        check("no internal detail leaked", "secret" not in str(crashed.model_dump()))
        bad = await registry.call("echo", req("x", skill="nope"))
        check(
            "unknown skill -> bad_request",
            bad.error and bad.error.code == "bad_request",
            str(bad.error),
        )
        # The HTTP read timeout must not cut a call short that its own deadline allows: an
        # agent is silent for as long as its slowest node (Synthesis waits on the Web agent).
        patient = AgentRegistry([URL], token=TOKEN, timeout_s=2)
        try:
            quiet = await patient.call("echo", req("slow", timeout_s=20))
        finally:
            await patient.close()
        check(
            "5 s of silence is fine under a 2 s default when the call's own deadline is 20 s",
            quiet.ok,
            str(quiet.error.code if quiet.error else quiet.status),
        )
        slow = await registry.call("echo", req("slow", timeout_s=1))
        check(
            "slow agent -> timeout",
            slow.error and slow.error.code == "timeout",
            str(slow.error),
        )
        unknown = await registry.call("ghost", req("x"))
        check(
            "unknown agent -> unavailable",
            unknown.error and unknown.error.code == "unavailable",
        )

        print("\nkilled agent")
        proc.send_signal(signal.SIGKILL)
        proc.wait()
        down = await registry.call("echo", req("hello"))
        check(
            "killed agent -> unavailable (no crash)",
            down.error and down.error.code == "unavailable",
            str(down.error),
        )
        check("agent is marked down", not registry.agents()[0].up)
        proc = start_agent()
        back = await registry.call("echo", req("hello"))
        check("restarted agent is picked up again", back.ok, back.answer)
    finally:
        await registry.close()
        if proc:
            proc.kill()
            proc.wait()

    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
