"""DB MCP server (Streamable HTTP): read-only access to the Northwind database.

Run from the project root:
    uv run python -m multi_agenti_conversation_system.mcp.servers.db_server.server

Every tool returns a dict. Failures come back as {"ok": False, "error": "..."} (never as an
exception), so the calling agent can read the message and correct itself.
"""

import asyncio
import ipaddress
import secrets
from typing import Any

import uvicorn
from mcp.server import MCPServer
from psycopg_pool import PoolTimeout
from starlette.requests import Request
from starlette.responses import JSONResponse

from multi_agenti_conversation_system.catalog import SchemaCatalog, load_catalog
from multi_agenti_conversation_system.config import (
    get_db_mcp_settings,
    get_db_settings,
    get_path_settings,
)
from multi_agenti_conversation_system.guardrails.sql_validator import (
    find_pii_columns,
    validate_select,
)
from multi_agenti_conversation_system.mcp.servers.db_server import db, schema_index

MAX_SQL_CHARS = 5000
MAX_QUESTION_CHARS = 500
MAX_K = 10

server = MCPServer(
    "northwind-db",
    instructions=(
        "Read-only access to the Northwind sales database (PostgreSQL). Typical flow: "
        "search_schema and get_few_shots to find relevant tables and example queries, "
        "explain_query to check cost and personal data, then run_select_query. "
        "Only a single SELECT is accepted; anything else is rejected."
    ),
)


# --------------------------------------------------------------------------- helpers
def _catalog() -> SchemaCatalog:
    return load_catalog(get_path_settings().DATA_DIR)


def _error(message: str) -> dict[str, Any]:
    return {"ok": False, "error": message}


def _check_text(value: str, name: str, limit: int) -> str | None:
    if not value or not value.strip():
        return f"{name} must not be empty"
    if len(value) > limit:
        return f"{name} is too long (max {limit} characters)"
    return None


def _check_search_args(question: str, k: int, domain: str | None) -> str | None:
    if problem := _check_text(question, "question", MAX_QUESTION_CHARS):
        return problem
    if not 1 <= k <= MAX_K:
        return f"k must be between 1 and {MAX_K}"
    domains = _catalog().domains()
    if domain is not None and domain not in domains:
        return f"Unknown domain '{domain}'. Valid domains: {domains}"
    return None


async def _in_thread(fn, *args):
    """Run blocking code (psycopg, Chroma, Ollama) off the event loop."""
    return await asyncio.to_thread(fn, *args)


# ---------------------------------------------------------------------------- tools
@server.tool()
async def list_tables() -> dict[str, Any]:
    """List every table in the database with its business domain and a short description.

    Use this to get an overview. Use search_schema to find the tables relevant to a question.
    """
    catalog = _catalog()
    pii = catalog.pii_columns()
    tables = [
        {
            "table": name,
            "domain": catalog.table_domain(name),
            "description": catalog.table_description(name),
            "has_pii": name.lower() in pii,
        }
        for name in catalog.tables
    ]
    return {"ok": True, "tables": tables}


@server.tool()
async def get_schema(tables: list[str]) -> dict[str, Any]:
    """Get the full description of specific tables: purpose, joins, and every column.

    Columns marked [PII] hold personal data; queries touching them need human approval.

    Args:
        tables: Table names, e.g. ["orders", "customers"] (case-insensitive).
    """
    catalog = _catalog()
    if not tables or len(tables) > len(catalog):
        return _error(f"Provide between 1 and {len(catalog)} table names")
    known = {name.lower(): name for name in catalog.tables}
    unknown = [t for t in tables if t.lower() not in known]
    if unknown:
        return _error(f"Unknown tables: {unknown}. Known tables: {sorted(known)}")
    docs = {known[t.lower()]: catalog.table_document(known[t.lower()]) for t in tables}
    return {"ok": True, "schemas": docs}


@server.tool()
async def search_schema(
    question: str, k: int = 5, domain: str | None = None
) -> dict[str, Any]:
    """Find the tables most relevant to a natural-language question (semantic search).

    Joins usually need more than one table, so keep k at 4 or more.

    Args:
        question: The user's question in plain English.
        k: How many tables to return (1-10).
        domain: Optional filter: sales, customers, products, hr or logistics.
    """
    if problem := _check_search_args(question, k, domain):
        return _error(problem)
    try:
        hits = await _in_thread(schema_index.search_schema, question, k, domain)
    except (ValueError, ConnectionError) as exc:  # Chroma down / Ollama down
        return _error(f"Schema search is unavailable: {type(exc).__name__}")
    return {"ok": True, "tables": hits}


@server.tool()
async def get_few_shots(
    question: str, k: int = 3, domain: str | None = None
) -> dict[str, Any]:
    """Find solved example questions with their SQL that resemble the given question.

    Use them as a style and join guide when writing a new query.

    Args:
        question: The user's question in plain English.
        k: How many examples to return (1-10).
        domain: Optional filter: sales, customers, products, hr or logistics.
    """
    if problem := _check_search_args(question, k, domain):
        return _error(problem)
    try:
        hits = await _in_thread(schema_index.search_few_shots, question, k, domain)
    except (ValueError, ConnectionError) as exc:
        return _error(f"Example search is unavailable: {type(exc).__name__}")
    return {"ok": True, "examples": hits}


@server.tool()
async def explain_query(sql: str) -> dict[str, Any]:
    """Check a SELECT query without running it.

    Returns the planner's estimated row count and which personal-data (PII) columns the
    query touches. Use it to decide whether a query needs human approval.

    Args:
        sql: A single PostgreSQL SELECT statement.
    """
    if problem := _check_text(sql, "sql", MAX_SQL_CHARS):
        return _error(problem)
    catalog = _catalog()
    checked = validate_select(sql, get_db_settings().MAX_ROWS, catalog.table_names())
    if not checked.ok:
        return _error(checked.error or "Query rejected")
    try:
        estimated_rows = await _in_thread(db.explain, checked.sql)
    except (db.QueryError, PoolTimeout) as exc:
        return _error(str(exc) or "Database unavailable")
    pii = find_pii_columns(checked.sql, catalog.pii_columns())
    return {
        "ok": True,
        "sql": checked.sql,
        "tables": sorted(checked.tables),
        "estimated_rows": estimated_rows,
        "touches_pii": bool(pii),
        "pii_columns": pii,
    }


@server.tool()
async def run_select_query(sql: str, max_rows: int | None = None) -> dict[str, Any]:
    """Run one read-only SELECT query and return its rows.

    The query is validated first (single SELECT, known tables only, no dangerous functions)
    and a row limit is enforced. If more rows exist than were returned, truncated is true.

    Args:
        sql: A single PostgreSQL SELECT statement.
        max_rows: Maximum rows to return. Defaults to, and cannot exceed, the server limit.
    """
    if problem := _check_text(sql, "sql", MAX_SQL_CHARS):
        return _error(problem)
    server_max = get_db_settings().MAX_ROWS
    limit = server_max if max_rows is None else max_rows
    if not 1 <= limit <= server_max:
        return _error(f"max_rows must be between 1 and {server_max}")

    # Validate with one extra row so db.fetch_rows can tell if the result was cut off.
    catalog = _catalog()
    checked = validate_select(sql, limit + 1, catalog.table_names())
    if not checked.ok:
        return _error(checked.error or "Query rejected")
    try:
        result = await _in_thread(db.fetch_rows, checked.sql, limit)
    except (db.QueryError, PoolTimeout) as exc:
        return _error(str(exc) or "Database unavailable")
    return {
        "ok": True,
        "sql": checked.sql,
        "pii_columns": find_pii_columns(checked.sql, catalog.pii_columns()),
        **result,
    }


@server.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


# ------------------------------------------------------------------------ HTTP layer
class BearerAuth:
    """Pure ASGI middleware: rejects requests without the right token (except /health)."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "http" and scope["path"] != "/health":
            got = dict(scope["headers"]).get(b"authorization", b"").decode()
            if not secrets.compare_digest(got, f"Bearer {self.token}"):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    cfg = get_db_mcp_settings()
    token = cfg.DB_MCP_TOKEN.get_secret_value()
    if not token and not _is_loopback(cfg.DB_MCP_HOST):
        raise SystemExit(
            "Refusing to listen on a non-local address without DB_MCP_TOKEN set."
        )
    app = BearerAuth(server.streamable_http_app(), token=token)
    try:
        uvicorn.run(app, host=cfg.DB_MCP_HOST, port=cfg.DB_MCP_PORT)
    finally:
        db.close_pool()


if __name__ == "__main__":
    main()
