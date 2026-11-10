"""Check the MCP client (Phase C step 1) against the running DB and Web servers.

Start both servers first (see smoke_db_mcp.py / smoke_web_mcp.py), then:
    uv run python scripts/smoke_mcp_client.py
"""

import asyncio
import sys

from multi_agenti_conversation_system.mcp.client import open_mcp, servers

failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


async def main() -> None:
    async with open_mcp(["db", "web"]) as tools:
        names = {server: sorted(t.name for t in ts) for server, ts in tools.items()}
        check("db tools discovered", len(names["db"]) >= 5, str(names["db"]))
        check(
            "web tools discovered, kept separate from db",
            names["web"] == ["fetch_page", "search_web"],
            str(names["web"]),
        )

        by_name = {t.name: t for ts in tools.values() for t in ts}
        result = await by_name["list_tables"].ainvoke({})
        check("a DB tool call works through the adapter", "orders" in str(result))
        result = await by_name["fetch_page"].ainvoke({"url": "http://127.0.0.1/"})
        if "rate_limited" in str(result):
            # A smoke script that ran just before may have emptied the fetch bucket.
            await asyncio.sleep(3)
            result = await by_name["fetch_page"].ainvoke({"url": "http://127.0.0.1/"})
        check(
            "a web tool call works (SSRF block returned as data)",
            "blocked" in str(result),
        )

    print("\nwrong token is rejected")
    original = servers()["db"]
    bad = type(original)("db", original.url, "not-the-token")
    try:
        from fastmcp.client.transports import StreamableHttpTransport
        from langchain.mcp import MCPAdapter

        async with MCPAdapter(
            StreamableHttpTransport(bad.url, headers=bad.headers)
        ) as a:
            await a.list_tools()
        check("bad token -> error", False, "connected anyway")
    except Exception as exc:  # noqa: BLE001
        check("bad token -> error", True, type(exc).__name__)

    print("\nunknown server name")
    try:
        async with open_mcp(["nope"]):
            pass
        check("unknown server -> ValueError", False)
    except ValueError:
        check("unknown server -> ValueError", True)

    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    asyncio.run(main())
