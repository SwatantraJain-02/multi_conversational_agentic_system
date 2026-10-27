"""All DB MCP tool logic, independent of the MCP transport.

The service checks what the agent sent, runs the SQL guardrail, and calls the reader and the
index it was given, and returns the typed responses from mcp.contracts.db. Failures are
raised as ToolError subclasses (ToolInputError, QueryFailed, BackendUnavailable); server.py
turns them into the same response model with ok=False. Tests drive it with fakes, no server
or database needed.
"""

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from multi_agenti_conversation_system.catalog import SchemaCatalog
from multi_agenti_conversation_system.guardrails.sql_validator import (
    find_pii_columns,
    validate_select,
)
from multi_agenti_conversation_system.mcp.contracts.db import (
    ExplainResponse,
    FewShot,
    FewShotsResponse,
    GetSchemaResponse,
    ListTablesResponse,
    SchemaHit,
    SearchSchemaResponse,
    SelectResponse,
    TableSummary,
)
from multi_agenti_conversation_system.mcp.servers.db_server.db import (
    DatabaseUnavailable,
    QueryError,
)
from multi_agenti_conversation_system.mcp.servers.db_server.schema_index import (
    SearchUnavailable,
)


class ToolError(Exception):
    """A failure with a message the calling agent should read and act on."""


class ToolInputError(ToolError):
    """The agent sent something invalid (empty text, unknown table, rejected SQL, ...)."""


class QueryFailed(ToolError):
    """Postgres rejected a validated query (unknown column, timeout, ...)."""


class BackendUnavailable(ToolError):
    """Postgres, Chroma or the embedding model could not answer."""


class DbReader(Protocol):
    def fetch_rows(self, sql: str, max_rows: int) -> dict[str, Any]: ...
    def explain(self, sql: str) -> int: ...
    def ping(self, timeout_s: float) -> bool: ...
    def close(self) -> None: ...


class SchemaSearch(Protocol):
    def search_schema(
        self, question: str, k: int, domain: str | None
    ) -> list[dict[str, Any]]: ...
    def search_few_shots(
        self, question: str, k: int, domain: str | None
    ) -> list[dict[str, Any]]: ...
    def ping(self) -> bool: ...
    def is_built(self) -> bool: ...


@dataclass(frozen=True)
class ToolLimits:
    max_rows: int = 100
    max_sql_chars: int = 5000
    max_question_chars: int = 500
    max_k: int = 10


async def _in_thread[T](fn: Callable[..., T], *args: Any) -> T:
    """Run blocking code (psycopg, Chroma, Ollama) off the event loop."""
    return await asyncio.to_thread(fn, *args)


class DbToolService:
    def __init__(
        self,
        catalog: SchemaCatalog,
        reader: DbReader,
        index: SchemaSearch,
        limits: ToolLimits,
        extra_checks: Mapping[str, Callable[[], bool]] | None = None,
        check_timeout_s: float = 2.0,
    ):
        self.catalog = catalog
        self.reader = reader
        self.index = index
        self.limits = limits
        self.extra_checks = dict(extra_checks or {})
        self.check_timeout_s = check_timeout_s

    def close(self) -> None:
        self.reader.close()

    # ------------------------------------------------------------------ readiness
    async def readiness(self) -> dict[str, bool]:
        """Can every dependency answer right now? Checks run concurrently, each time-boxed."""
        checks: dict[str, Callable[[], bool]] = {
            "postgres": lambda: self.reader.ping(self.check_timeout_s),
            "chroma": self.index.ping,
            "schema_index": self.index.is_built,
            **self.extra_checks,
        }

        async def run(check: Callable[[], bool]) -> bool:
            try:
                return await asyncio.wait_for(_in_thread(check), self.check_timeout_s)
            except Exception:  # noqa: BLE001 - any failure means "not ready"
                return False

        results = await asyncio.gather(*(run(c) for c in checks.values()))
        return dict(zip(checks, results, strict=True))

    # ------------------------------------------------------------------ input checks
    @staticmethod
    def _check_text(value: str, name: str, limit: int) -> None:
        if not value or not value.strip():
            raise ToolInputError(f"{name} must not be empty")
        if len(value) > limit:
            raise ToolInputError(f"{name} is too long (max {limit} characters)")

    def _check_search_args(self, question: str, k: int, domain: str | None) -> None:
        self._check_text(question, "question", self.limits.max_question_chars)
        if not 1 <= k <= self.limits.max_k:
            raise ToolInputError(f"k must be between 1 and {self.limits.max_k}")
        domains = self.catalog.domains()
        if domain is not None and domain not in domains:
            raise ToolInputError(f"Unknown domain '{domain}'. Valid domains: {domains}")

    def _validated(self, sql: str, max_rows: int):
        self._check_text(sql, "sql", self.limits.max_sql_chars)
        checked = validate_select(sql, max_rows, self.catalog.table_names())
        if not checked.ok:
            raise ToolInputError(checked.error or "Query rejected")
        return checked

    # ------------------------------------------------------------------ tools
    async def list_tables(self) -> ListTablesResponse:
        catalog = self.catalog
        pii = catalog.pii_columns()
        tables = [
            TableSummary(
                table=name,
                domain=catalog.table_domain(name),
                description=catalog.table_description(name),
                has_pii=name.lower() in pii,
            )
            for name in catalog.tables
        ]
        return ListTablesResponse(ok=True, tables=tables)

    async def get_schema(self, tables: list[str]) -> GetSchemaResponse:
        catalog = self.catalog
        if not tables or len(tables) > len(catalog):
            raise ToolInputError(f"Provide between 1 and {len(catalog)} table names")
        known = {name.lower(): name for name in catalog.tables}
        unknown = [t for t in tables if t.lower() not in known]
        if unknown:
            raise ToolInputError(
                f"Unknown tables: {unknown}. Known tables: {sorted(known)}"
            )
        docs = {
            known[t.lower()]: catalog.table_document(known[t.lower()]) for t in tables
        }
        return GetSchemaResponse(ok=True, schemas=docs)

    async def search_schema(
        self, question: str, k: int, domain: str | None
    ) -> SearchSchemaResponse:
        self._check_search_args(question, k, domain)
        try:
            hits = await _in_thread(self.index.search_schema, question, k, domain)
        except SearchUnavailable as exc:
            raise BackendUnavailable(f"Schema search is unavailable: {exc}") from exc
        return SearchSchemaResponse(ok=True, tables=[SchemaHit(**h) for h in hits])

    async def get_few_shots(
        self, question: str, k: int, domain: str | None
    ) -> FewShotsResponse:
        self._check_search_args(question, k, domain)
        try:
            hits = await _in_thread(self.index.search_few_shots, question, k, domain)
        except SearchUnavailable as exc:
            raise BackendUnavailable(f"Example search is unavailable: {exc}") from exc
        return FewShotsResponse(ok=True, examples=[FewShot(**h) for h in hits])

    async def explain_query(self, sql: str) -> ExplainResponse:
        checked = self._validated(sql, self.limits.max_rows)
        try:
            estimated_rows = await _in_thread(self.reader.explain, checked.sql)
        except DatabaseUnavailable as exc:
            raise BackendUnavailable(str(exc)) from exc
        except QueryError as exc:
            raise QueryFailed(str(exc) or "Query failed") from exc
        pii = find_pii_columns(checked.sql, self.catalog.pii_columns())
        return ExplainResponse(
            ok=True,
            sql=checked.sql,
            tables=sorted(checked.tables),
            estimated_rows=estimated_rows,
            touches_pii=bool(pii),
            pii_columns=pii,
        )

    async def run_select_query(self, sql: str, max_rows: int | None) -> SelectResponse:
        self._check_text(sql, "sql", self.limits.max_sql_chars)
        server_max = self.limits.max_rows
        limit = server_max if max_rows is None else max_rows
        if not 1 <= limit <= server_max:
            raise ToolInputError(f"max_rows must be between 1 and {server_max}")

        # Validate with one extra row so the reader can tell if the result was cut off.
        checked = self._validated(sql, limit + 1)
        try:
            result = await _in_thread(self.reader.fetch_rows, checked.sql, limit)
        except DatabaseUnavailable as exc:
            raise BackendUnavailable(str(exc)) from exc
        except QueryError as exc:
            raise QueryFailed(str(exc) or "Query failed") from exc
        return SelectResponse(
            ok=True,
            sql=checked.sql,
            pii_columns=find_pii_columns(checked.sql, self.catalog.pii_columns()),
            **result,
        )
