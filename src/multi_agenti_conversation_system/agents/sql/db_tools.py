"""The SQL agent's only door to the database: the DB MCP server's tools.

Each method returns the tool's reply dict ({"ok": ..., ...}). A transport failure (server
down, token rejected) raises `AgentError("unavailable")`, so it is never mistaken for a
mistake in the SQL: the graph retries it, and the caller sees `unavailable`, not a prompt
for the model to "fix" its query.
"""

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import psycopg
import psycopg.errors as pg_errors
from langchain_core.tools import BaseTool

from multi_agenti_conversation_system.a2a.contracts import AgentError
from multi_agenti_conversation_system.mcp.client import tool_result

DB_DOWN = (
    "Database unavailable"  # what the server says when it has no message of its own
)

# The server reports a database failure as "ClassName: first line". Connection and
# infrastructure failures (OperationalError / InterfaceError and their many subclasses:
# dropped connection, admin shutdown, too many connections...) are NOT the model's fault.
# These operational errors ARE: a cheaper or simpler query fixes them.
_FIXABLE_BY_THE_MODEL = {
    "QueryCanceled",  # statement_timeout: write a cheaper query
    "ProgramLimitExceeded",
    "StatementTooComplex",
    "TooManyColumns",
    "TooManyArguments",
}
_DB_DOWN_ERRORS = {
    name
    for name, cls in vars(pg_errors).items()
    if isinstance(cls, type)
    and issubclass(cls, (psycopg.OperationalError, psycopg.InterfaceError))
} | {"OperationalError", "InterfaceError"}
_DB_DOWN_ERRORS -= _FIXABLE_BY_THE_MODEL


def is_db_down(error: str) -> bool:
    """Is this error text from the DB server about the database being unreachable?

    True: hand it to the caller as `unavailable`. False: it's about the SQL, so the
    model gets it back to correct the query.
    """
    name = error.split(":", 1)[0].strip()
    return (
        error == DB_DOWN
        or error.startswith("couldn't get a connection")  # psycopg_pool.PoolTimeout
        or name in _DB_DOWN_ERRORS
    )


@dataclass(frozen=True)
class SchemaSnapshot:
    table_names: set[str]
    table_domains: dict[str, str]
    schema_version: (
        str  # hash of every table's description: learned pairs expire with it
    )


class DbTools:
    def __init__(self, tools: dict[str, BaseTool]) -> None:
        self._tools = tools

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = await self._tools[name].ainvoke(args)
        except Exception as exc:
            raise AgentError(
                "unavailable",
                f"database tools unavailable ({type(exc).__name__})",
                retryable=True,
            ) from exc
        return tool_result(raw)

    async def list_tables(self) -> dict[str, Any]:
        return await self._call("list_tables", {})

    async def get_schema(self, tables: list[str]) -> dict[str, Any]:
        return await self._call("get_schema", {"tables": tables})

    async def search_schema(
        self, question: str, k: int, domain: str | None
    ) -> dict[str, Any]:
        return await self._call(
            "search_schema", {"question": question, "k": k, "domain": domain}
        )

    async def get_few_shots(
        self, question: str, k: int, domain: str | None
    ) -> dict[str, Any]:
        return await self._call(
            "get_few_shots", {"question": question, "k": k, "domain": domain}
        )

    async def explain_query(self, sql: str) -> dict[str, Any]:
        return await self._call("explain_query", {"sql": sql})

    async def run_select_query(self, sql: str, max_rows: int) -> dict[str, Any]:
        return await self._call("run_select_query", {"sql": sql, "max_rows": max_rows})

    async def snapshot(self) -> SchemaSnapshot:
        """Table names, their domains and a schema version, read once at startup."""
        listing = await self.list_tables()
        if not listing.get("ok"):
            raise AgentError(
                "unavailable", f"cannot list tables: {listing.get('error')}"
            )
        names = [t["table"] for t in listing["tables"]]
        schemas = await self.get_schema(names)
        if not schemas.get("ok"):
            raise AgentError(
                "unavailable", f"cannot read schemas: {schemas.get('error')}"
            )
        digest = hashlib.sha1(
            json.dumps(schemas["schemas"], sort_keys=True).encode()
        ).hexdigest()[:12]
        return SchemaSnapshot(
            table_names={n.lower() for n in names},
            table_domains={t["table"].lower(): t["domain"] for t in listing["tables"]},
            schema_version=digest,
        )
