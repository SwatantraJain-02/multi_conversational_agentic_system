"""The services an eval run needs, started on spare ports and stopped afterwards.

DB MCP, Web MCP and the SQL, Web and Synthesis agents, as subprocesses with `EVAL_MODE=true`
(the SQL agent writes no learned pairs, G23) and the run's prompt label, so every process
uses the same prompt versions. Your own servers on the normal ports are not touched.
"""

import os
import secrets
import signal
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import httpx

PKG = "multi_agenti_conversation_system"
PORTS = {"db": 25555, "web_mcp": 25556, "sql": 27001, "web": 27002, "synthesis": 27003}


def _spawn(
    module: str, env: dict[str, str], port: int, wait_s: int = 120
) -> subprocess.Popen:
    proc = subprocess.Popen(
        [sys.executable, "-m", module],
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            if (
                httpx.get(f"http://localhost:{port}/health", timeout=1).status_code
                == 200
            ):
                return proc
        except httpx.HTTPError:
            time.sleep(0.5)
    proc.kill()
    raise RuntimeError(f"{module} did not become healthy")


def _stop(proc: subprocess.Popen) -> None:
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def eval_stack(prompt_label: str) -> Iterator[dict[str, str]]:
    """Start the services; yields `{"token": ..., "urls": [agent urls]}`-style settings."""
    token = secrets.token_hex(16)
    env = {
        "DB_MCP_PORT": str(PORTS["db"]),
        "WEB_MCP_PORT": str(PORTS["web_mcp"]),
        "SQL_AGENT_PORT": str(PORTS["sql"]),
        "WEB_AGENT_PORT": str(PORTS["web"]),
        "SYNTHESIS_AGENT_PORT": str(PORTS["synthesis"]),
        "A2A_TOKEN": token,
        "PROMPT_LABEL": prompt_label,
        "EVAL_MODE": "true",
    }
    procs: list[subprocess.Popen] = []
    try:
        for module, port in (
            (f"{PKG}.mcp.servers.db_server.server", PORTS["db"]),
            (f"{PKG}.mcp.servers.web_server.server", PORTS["web_mcp"]),
            (f"{PKG}.agents.sql.app", PORTS["sql"]),
            (f"{PKG}.agents.web.app", PORTS["web"]),
            (f"{PKG}.agents.synthesis.app", PORTS["synthesis"]),
        ):
            procs.append(_spawn(module, env, port))
        yield {
            "token": token,
            "urls": ",".join(
                f"http://localhost:{PORTS[k]}" for k in ("sql", "web", "synthesis")
            ),
        }
    finally:
        for proc in reversed(procs):
            _stop(proc)
