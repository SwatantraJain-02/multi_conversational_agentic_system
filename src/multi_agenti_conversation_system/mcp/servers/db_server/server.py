"""DB MCP server (Streamable HTTP): read-only access to the Northwind database.

Run from the project root:
    uv run db-mcp-server
or  uv run python -m multi_agenti_conversation_system.mcp.servers.db_server.server

This module is only the transport. create_server() registers thin tool adapters over a
DbToolService; build_service() and main() are the composition root that reads settings and
wires the adapters together. Every tool returns one typed model from mcp.contracts.db.
Failures come back as that model with ok=False and an error message (never as an
exception), so the calling agent can read the message and correct itself.

HTTP routes besides /mcp: /health (liveness, always 200) and /ready (200 only when Postgres,
Chroma, the schema index and Ollama all answer; 503 otherwise). Neither needs the token.
"""

from collections.abc import Awaitable
from functools import partial
from textwrap import dedent

import uvicorn
from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

from multi_agenti_conversation_system.catalog import load_catalog
from multi_agenti_conversation_system.config import (
    get_chroma_settings,
    get_db_mcp_settings,
    get_db_settings,
    get_db_tool_limits,
    get_embedding_settings,
    get_path_settings,
)
from multi_agenti_conversation_system.llm import build_embeddings, ping_ollama
from multi_agenti_conversation_system.mcp.contracts.db import (
    ExplainResponse,
    FewShotsResponse,
    GetSchemaResponse,
    ListTablesResponse,
    SearchSchemaResponse,
    SelectResponse,
    ToolResponse,
)
from multi_agenti_conversation_system.mcp.servers.db_server.db import NorthwindReader
from multi_agenti_conversation_system.mcp.servers.db_server.schema_index import (
    SchemaIndex,
)
from multi_agenti_conversation_system.mcp.servers.db_server.service import (
    DbToolService,
    ToolError,
    ToolLimits,
)
from multi_agenti_conversation_system.mcp.servers.http_auth import (
    BearerAuth,
    is_loopback,
)
from multi_agenti_conversation_system.vectorstore import build_chroma_client

INSTRUCTIONS = (
    "Read-only access to the Northwind sales database (PostgreSQL). Typical flow: "
    "search_schema and get_few_shots to find relevant tables and example queries, "
    "explain_query to check cost and personal data, then run_select_query. "
    "Only a single SELECT is accepted; anything else is rejected."
)

# What the LLM reads about each tool. {max_k} and {domains} are filled in from the
# configured limits and the catalog, so the text can never drift from what is enforced.
DESCRIPTIONS = {
    "list_tables": """\
        List every table in the database with its business domain and a short description.

        Use this to get an overview. Use search_schema to find the tables relevant to a question.
        """,
    "get_schema": """\
        Get the full description of specific tables: purpose, joins, and every column.

        Columns marked [PII] hold personal data; queries touching them need human approval.

        Args:
            tables: Table names, e.g. ["orders", "customers"] (case-insensitive).
        """,
    "search_schema": """\
        Find the tables most relevant to a natural-language question (semantic search).

        Joins usually need more than one table, so keep k at 4 or more.

        Args:
            question: The user's question in plain English.
            k: How many tables to return (1-{max_k}).
            domain: Optional filter: {domains}.
        """,
    "get_few_shots": """\
        Find solved example questions with their SQL that resemble the given question.

        Use them as a style and join guide when writing a new query.

        Args:
            question: The user's question in plain English.
            k: How many examples to return (1-{max_k}).
            domain: Optional filter: {domains}.
        """,
    "explain_query": """\
        Check a SELECT query without running it.

        Returns the planner's estimated row count and which personal-data (PII) columns the
        query touches. Use it to decide whether a query needs human approval.

        Args:
            sql: A single PostgreSQL SELECT statement.
        """,
    "run_select_query": """\
        Run one read-only SELECT query and return its rows.

        The query is validated first (single SELECT, known tables only, no dangerous functions)
        and a row limit is enforced. If more rows exist than were returned, truncated is true.

        Args:
            sql: A single PostgreSQL SELECT statement.
            max_rows: Maximum rows to return. Defaults to, and cannot exceed, the server limit.
        """,
}


def _describe(tool: str, service: DbToolService) -> str:
    # In the catalog's own order, e.g. "sales, customers, products, hr or logistics".
    domains = list(
        dict.fromkeys(service.catalog.table_domain(t) for t in service.catalog.tables)
    )
    joined = (
        ", ".join(domains[:-1]) + " or " + domains[-1]
        if len(domains) > 1
        else "".join(domains)
    )
    return dedent(DESCRIPTIONS[tool]).format(max_k=service.limits.max_k, domains=joined)


async def _respond[R: ToolResponse](model: type[R], call: Awaitable[R]) -> R:
    try:
        return await call
    except ToolError as exc:
        return model(ok=False, error=str(exc))


# ---------------------------------------------------------------------------- tools
def create_server(service: DbToolService) -> MCPServer:
    server = MCPServer("northwind-db", instructions=INSTRUCTIONS)

    def tool(name: str):
        return server.tool(name=name, description=_describe(name, service))

    @tool("list_tables")
    async def list_tables() -> ListTablesResponse:
        return await _respond(ListTablesResponse, service.list_tables())

    @tool("get_schema")
    async def get_schema(tables: list[str]) -> GetSchemaResponse:
        return await _respond(GetSchemaResponse, service.get_schema(tables))

    @tool("search_schema")
    async def search_schema(
        question: str, k: int = 5, domain: str | None = None
    ) -> SearchSchemaResponse:
        return await _respond(
            SearchSchemaResponse, service.search_schema(question, k, domain)
        )

    @tool("get_few_shots")
    async def get_few_shots(
        question: str, k: int = 3, domain: str | None = None
    ) -> FewShotsResponse:
        return await _respond(
            FewShotsResponse, service.get_few_shots(question, k, domain)
        )

    @tool("explain_query")
    async def explain_query(sql: str) -> ExplainResponse:
        return await _respond(ExplainResponse, service.explain_query(sql))

    @tool("run_select_query")
    async def run_select_query(sql: str, max_rows: int | None = None) -> SelectResponse:
        return await _respond(SelectResponse, service.run_select_query(sql, max_rows))

    @server.custom_route("/health", methods=["GET"])
    async def health(request: Request) -> JSONResponse:
        """Liveness: the process is up. Checks no dependencies."""
        return JSONResponse({"status": "ok"})

    @server.custom_route("/ready", methods=["GET"])
    async def ready(request: Request) -> JSONResponse:
        """Readiness: every dependency answers. Booleans only, never error details."""
        checks = await service.readiness()
        ok = all(checks.values())
        return JSONResponse(
            {"status": "ready" if ok else "not_ready", "checks": checks},
            status_code=200 if ok else 503,
        )

    return server


# ------------------------------------------------------------------ composition root
def build_service() -> DbToolService:
    """Build the catalog, the adapters and the service from settings, once per process."""
    db_cfg = get_db_settings()
    embed_cfg = get_embedding_settings()
    limits = get_db_tool_limits()
    catalog = load_catalog(get_path_settings().DATA_DIR)
    reader = NorthwindReader(
        db_cfg.conninfo,
        min_size=db_cfg.DB_POOL_MIN_SIZE,
        max_size=db_cfg.DB_POOL_MAX_SIZE,
        checkout_timeout_s=db_cfg.DB_POOL_CHECKOUT_TIMEOUT_S,
    )
    index = SchemaIndex(
        build_chroma_client(get_chroma_settings()),
        build_embeddings(embed_cfg),
        catalog,
    )
    check_timeout_s = 2.0
    return DbToolService(
        catalog,
        reader,
        index,
        ToolLimits(
            max_rows=db_cfg.MAX_ROWS,
            max_sql_chars=limits.MCP_MAX_SQL_CHARS,
            max_question_chars=limits.MCP_MAX_QUESTION_CHARS,
            max_k=limits.MCP_MAX_K,
        ),
        extra_checks={"ollama": partial(ping_ollama, embed_cfg, check_timeout_s)},
        check_timeout_s=check_timeout_s,
    )


def create_app(service: DbToolService, token: str) -> ASGIApp:
    return BearerAuth(create_server(service).streamable_http_app(), token=token)


def main() -> None:
    cfg = get_db_mcp_settings()
    token = cfg.DB_MCP_TOKEN.get_secret_value()
    if not token and not is_loopback(cfg.DB_MCP_HOST):
        raise SystemExit(
            "Refusing to listen on a non-local address without DB_MCP_TOKEN set."
        )
    service = build_service()
    try:
        uvicorn.run(
            create_app(service, token), host=cfg.DB_MCP_HOST, port=cfg.DB_MCP_PORT
        )
    finally:
        service.close()


if __name__ == "__main__":
    main()
