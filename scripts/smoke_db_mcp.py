"""End-to-end check of the DB MCP server over HTTP. This is the Phase 2 gate evidence.

Start the server first (separate terminal):
    uv run python -m multi_agenti_conversation_system.mcp.servers.db_server.server
Then run:
    uv run python scripts/smoke_db_mcp.py
"""

import asyncio
import json
import sys

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

from multi_agenti_conversation_system.config import get_db_mcp_settings

BASE_URL = get_db_mcp_settings().base_url
MCP_URL = get_db_mcp_settings().url
EXPECTED_TOOLS = {
    "list_tables",
    "get_schema",
    "search_schema",
    "get_few_shots",
    "explain_query",
    "run_select_query",
}

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


async def count_orders(session: ClientSession) -> int:
    data = await call(
        session, "run_select_query", sql="SELECT COUNT(*) AS n FROM orders"
    )
    return data["rows"][0][0]


async def run_checks(session: ClientSession) -> None:
    tools = {t.name for t in (await session.list_tools()).tools}
    check("six tools are listed", tools == EXPECTED_TOOLS, str(sorted(tools)))

    orders_before = await count_orders(session)
    print(f"\norders before: {orders_before}")

    print("\nread tools")
    data = await call(session, "list_tables")
    check("list_tables returns 14 tables", data["ok"] and len(data["tables"]) == 14)

    data = await call(session, "get_schema", tables=["Orders", "shippers"])
    check(
        "get_schema (case-insensitive)",
        data["ok"] and set(data["schemas"]) == {"orders", "shippers"},
    )
    data = await call(session, "get_schema", tables=["nope"])
    check("get_schema rejects an unknown table", not data["ok"])

    data = await call(session, "search_schema", question="top customers by spend", k=4)
    check(
        "search_schema finds customers",
        data["ok"] and "customers" in [t["table"] for t in data["tables"]],
    )
    data = await call(session, "search_schema", question="x", domain="bogus")
    check("search_schema rejects an unknown domain", not data["ok"])

    data = await call(session, "get_few_shots", question="revenue by country", k=2)
    check(
        "get_few_shots returns examples with SQL",
        data["ok"] and all(e["sql"] for e in data["examples"]),
    )

    print("\nexplain_query")
    data = await call(session, "explain_query", sql="SELECT * FROM customers")
    check(
        "estimates rows",
        data["ok"] and data["estimated_rows"] > 0,
        str(data.get("estimated_rows")),
    )
    check(
        "flags PII for SELECT *",
        data["ok"] and data["touches_pii"] and "customers.phone" in data["pii_columns"],
    )
    data = await call(session, "explain_query", sql="SELECT order_id FROM orders")
    check("no PII for order_id", data["ok"] and not data["touches_pii"])

    print("\nrun_select_query")
    data = await call(
        session, "run_select_query", sql="SELECT * FROM orders", max_rows=5
    )
    check(
        "max_rows=5 truncates",
        data["ok"] and data["row_count"] == 5 and data["truncated"],
        data.get("sql", ""),
    )
    data = await call(
        session, "run_select_query", sql="SELECT order_id FROM orders LIMIT 3"
    )
    check(
        "small LIMIT is kept, not truncated",
        data["ok"] and data["row_count"] == 3 and not data["truncated"],
    )
    data = await call(session, "run_select_query", sql="SELECT 1", max_rows=100000)
    check("max_rows above the server cap is rejected", not data["ok"])

    print("\nblocked queries")
    for sql in [
        "DELETE FROM orders",
        "SELECT 1; DROP TABLE orders",
        "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",
        "SELECT pg_sleep(10)",
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * INTO stolen FROM orders",
        "",
    ]:
        data = await call(session, "run_select_query", sql=sql)
        check(f"blocked: {sql[:50]!r}", not data["ok"], data.get("error", ""))

    orders_after = await count_orders(session)
    print(f"\norders after: {orders_after}")
    check("orders table is untouched", orders_after == orders_before)


async def main() -> None:
    token = get_db_mcp_settings().DB_MCP_TOKEN.get_secret_value()
    async with httpx.AsyncClient() as anonymous:
        try:
            health = await anonymous.get(f"{BASE_URL}/health", timeout=5)
        except httpx.ConnectError:
            sys.exit(f"Cannot reach {BASE_URL}. Is the DB MCP server running?")
        check("/health is open and returns 200", health.status_code == 200)
        ready = await anonymous.get(f"{BASE_URL}/ready", timeout=10)
        check("/ready is open and returns 200", ready.status_code == 200, ready.text)

        no_token = await anonymous.post(MCP_URL, json={})
        check("no token -> 401", no_token.status_code == 401)
        wrong = await anonymous.post(
            MCP_URL, json={}, headers={"Authorization": "Bearer wrong"}
        )
        check("wrong token -> 401", wrong.status_code == 401)

    http = httpx.AsyncClient(headers={"Authorization": f"Bearer {token}"})
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
