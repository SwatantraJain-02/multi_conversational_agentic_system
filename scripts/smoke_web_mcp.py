"""End-to-end check of the Web MCP server over HTTP. This is the Phase B gate evidence.

Start the server first (separate terminal):
    uv run python -m multi_agenti_conversation_system.mcp.servers.web_server.server
Then run:
    uv run python scripts/smoke_web_mcp.py

Uses a few Tavily credits (two searches, maybe one extract).

Redirects towards private addresses are covered by unit tests (tests/unit/test_web_fetch.py)
with a mock transport: public redirect services such as httpbin refuse to redirect to
loopback themselves, so they can't demonstrate it. Live, the script checks a public DNS
name that resolves to 127.0.0.1 (nip.io), which the guard must also block.
"""

import asyncio
import json
import sys

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from multi_agenti_conversation_system.config import settings

BASE_URL = f"http://{settings.WEB_MCP_HOST}:{settings.WEB_MCP_PORT}"
MCP_URL = f"{BASE_URL}/mcp"
EXPECTED_TOOLS = {"search_web", "fetch_page"}
BLOCKED_URLS = [
    "http://127.0.0.1:5555/health",  # our own DB server
    "http://169.254.169.254/latest/meta-data/",  # cloud metadata
    "http://localhost/",
    "http://127.0.0.1.nip.io:5555/health",  # public DNS name -> 127.0.0.1
    "file:///etc/passwd",
]

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


async def call(session: ClientSession, tool: str, **arguments) -> dict:
    result = await session.call_tool(tool, arguments)
    return json.loads(result.content[0].text)


async def run_checks(session: ClientSession) -> None:
    tools = {t.name for t in (await session.list_tools()).tools}
    check("two tools are listed", tools == EXPECTED_TOOLS, str(sorted(tools)))

    print("\nsearch_web")
    data = await call(
        session, "search_web", query="global beverage market trends 2026", max_results=5
    )
    results = data.get("results", [])
    check(
        "returns results with URLs",
        data["ok"] and results and all(r["url"] for r in results),
        f"{len(results)} results",
    )
    for r in results[:3]:
        print(f"      {r['url']}  ({r['published_date'] or 'no date'})")
    news = await call(
        session, "search_web", query="beverage industry", recency="month", topic="news"
    )
    check(
        "recency + news topic works",
        news["ok"],
        f"{len(news.get('results', []))} results",
    )
    bad = await call(session, "search_web", query="x", max_results=50)
    check("max_results above 10 is rejected", not bad["ok"], bad.get("error", ""))

    print("\nfetch_page")
    fetched = None
    for r in results:
        page = await call(session, "fetch_page", url=r["url"], max_chars=3000)
        if page["ok"]:
            fetched = page
            break
        print(f"      skipped {r['url']}: {page['error']}")
    check(
        "fetches and extracts a search result",
        fetched is not None and len(fetched["text"]) > 200,
        f"{fetched['extracted_by']}, {len(fetched['text'])} chars, truncated={fetched['truncated']}"
        if fetched
        else "no result could be fetched",
    )

    for url in BLOCKED_URLS:
        page = await call(session, "fetch_page", url=url)
        check(
            f"blocked: {url}",
            not page["ok"] and "blocked" in page["error"],
            page.get("error", ""),
        )

    print("\nrate limit")
    statuses = []
    for _ in range(settings.WEB_FETCH_BURST + 5):
        page = await call(session, "fetch_page", url="http://127.0.0.1/")
        statuses.append(page.get("error"))
    limited = [s for s in statuses if s == "rate_limited"]
    check(
        "a burst of fetches trips the limit",
        bool(limited),
        f"{len(limited)} of {len(statuses)} calls rate_limited",
    )
    if limited:
        page = await call(session, "fetch_page", url="http://127.0.0.1/")
        print(f"      retry_after_s={page.get('retry_after_s')}")


async def main() -> None:
    token = settings.WEB_MCP_TOKEN
    async with httpx.AsyncClient() as anonymous:
        try:
            health = await anonymous.get(f"{BASE_URL}/health", timeout=5)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach {BASE_URL}. Is the Web MCP server running?")
        check("/health is open and returns 200", health.status_code == 200)
        if token:
            no_token = await anonymous.post(MCP_URL, json={})
            check("no token -> 401", no_token.status_code == 401)
        else:
            print("  (WEB_MCP_TOKEN is empty: auth is off, allowed only on loopback)")

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    http = httpx.AsyncClient(headers=headers, timeout=60)
    async with (
        http,
        streamable_http_client(MCP_URL, http_client=http) as (read, write, *_),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        await run_checks(session)

    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
