"""Start everything the chat UI needs, wait until it is healthy, and stop it all on Ctrl-C.

    uv run python scripts/run_all.py            # MCP servers + the three agents
    uv run python scripts/run_all.py --ui       # ... and the Streamlit UI on :8501

Order: the two MCP servers (DB :5555, Web :5556), then the SQL (:7001), Web (:7002) and
Synthesis (:7003) agents, each waited for on its `/health`. Anything already answering on its
port is reused, not started a second time (and not stopped on exit). Ports and tokens come
from `.env`. Agent logs go to `logs/<name>.log`.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from multi_agenti_conversation_system.config import settings

PKG = "multi_agenti_conversation_system"
LOGS = Path("logs")
HEALTH_WAIT_S = 120


def services() -> list[tuple[str, str, int]]:
    """(name, python module, port), in start order."""
    return [
        ("db-mcp", f"{PKG}.mcp.servers.db_server.server", settings.DB_MCP_PORT),
        ("web-mcp", f"{PKG}.mcp.servers.web_server.server", settings.WEB_MCP_PORT),
        ("sql-agent", f"{PKG}.agents.sql.app", settings.SQL_AGENT_PORT),
        ("web-agent", f"{PKG}.agents.web.app", settings.WEB_AGENT_PORT),
        (
            "synthesis-agent",
            f"{PKG}.agents.synthesis.app",
            settings.SYNTHESIS_AGENT_PORT,
        ),
    ]


def healthy(port: int) -> bool:
    try:
        return (
            httpx.get(f"http://localhost:{port}/health", timeout=1).status_code == 200
        )
    except httpx.HTTPError:
        return False


def start(name: str, module: str, port: int) -> subprocess.Popen | None:
    if healthy(port):
        print(f"  {name:16} :{port}  already running, reusing it")
        return None
    LOGS.mkdir(exist_ok=True)
    log = (LOGS / f"{name}.log").open("ab")
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        stdout=log,
        stderr=subprocess.STDOUT,
        env=os.environ,
    )
    deadline = time.monotonic() + HEALTH_WAIT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"{name} exited early (see logs/{name}.log)")
        if healthy(port):
            print(f"  {name:16} :{port}  healthy")
            return proc
        time.sleep(0.5)
    proc.kill()
    raise SystemExit(
        f"{name} did not become healthy in {HEALTH_WAIT_S}s (see logs/{name}.log)"
    )


def stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--ui", action="store_true", help="also run the Streamlit UI")
    args = parser.parse_args()

    started: list[subprocess.Popen] = []
    try:
        print("starting services")
        for name, module, port in services():
            if proc := start(name, module, port):
                started.append(proc)
        if args.ui:
            ui = subprocess.Popen(
                [sys.executable, "-m", "streamlit", "run", "src/ui/streamlit_app.py"]
            )
            started.append(ui)
            print("  ui               :8501  starting (http://localhost:8501)")
        print("\nall up. Ctrl-C stops what this script started.")
        while True:
            for proc in started:
                if proc.poll() is not None:
                    raise SystemExit(
                        f"a service exited (code {proc.returncode}); stopping"
                    )
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        for proc in reversed(started):
            stop(proc)


if __name__ == "__main__":
    main()
