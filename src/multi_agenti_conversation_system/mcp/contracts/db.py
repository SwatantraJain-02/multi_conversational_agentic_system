"""Response models of the DB MCP server's tools: the contract between the server and agents.

Pure (pydantic only), so the server builds these and the agent can parse tool results into
the same classes. Each tool returns exactly one model: `ok` and `error`, plus payload fields
that are None when ok is False. One model per tool (instead of a success/error union) keeps
the published outputSchema a plain object.
"""

from typing import Any

from pydantic import BaseModel


class ToolResponse(BaseModel):
    ok: bool
    error: str | None = None
    """Set when ok is False: a short message the agent can act on."""


class TableSummary(BaseModel):
    table: str
    domain: str
    description: str
    has_pii: bool


class ListTablesResponse(ToolResponse):
    tables: list[TableSummary] | None = None


class GetSchemaResponse(ToolResponse):
    schemas: dict[str, str] | None = None
    """{table: plain-text description with every column; [PII] marks personal data}."""


class SchemaHit(BaseModel):
    table: str
    domain: str
    has_pii: bool
    document: str
    similarity: float
    """Cosine similarity to the question, 0..1 (higher is closer)."""


class SearchSchemaResponse(ToolResponse):
    tables: list[SchemaHit] | None = None


class FewShot(BaseModel):
    question: str
    sql: str
    domain: str
    pii: bool
    similarity: float


class FewShotsResponse(ToolResponse):
    examples: list[FewShot] | None = None


class ExplainResponse(ToolResponse):
    sql: str | None = None
    """The validated query, with the row limit applied."""
    tables: list[str] | None = None
    estimated_rows: int | None = None
    touches_pii: bool | None = None
    pii_columns: list[str] | None = None
    """'table.column' entries for every personal-data column the query reads."""


class SelectResponse(ToolResponse):
    sql: str | None = None
    pii_columns: list[str] | None = None
    columns: list[str] | None = None
    rows: list[list[Any]] | None = None
    row_count: int | None = None
    truncated: bool | None = None
    """True when more rows exist than were returned."""
