"""State of one SQL agent run, and the model's structured output."""

from typing import Any, TypedDict

from pydantic import BaseModel, Field


class SQLGeneration(BaseModel):
    """What the model returns for one attempt (structured output)."""

    sql: str = Field(
        default="", description="One PostgreSQL SELECT, or empty when refusing"
    )
    explanation: str = Field(
        default="",
        description="One or two sentences on how the query answers the question",
    )
    tables_used: list[str] = Field(default_factory=list)
    refusal: str | None = Field(
        default=None,
        description="Set (and sql left empty) when the request needs a write or can't be answered from this database",
    )


class SqlState(TypedDict, total=False):
    # --- input, reset on every task by `graph.build_input`
    question: str
    domain: str | None
    max_rows: int
    max_retries: int
    # --- retrieval
    schema_docs: list[str]
    examples: list[dict[str, str]]
    degraded: bool  # semantic search was down, keyword search answered alone
    # --- generation and self-correction
    sql: str
    explanation: str
    tables_used: list[str]
    attempts: int  # how many times the model was asked
    error: str  # why the last attempt failed; "" when it didn't
    failed_sql: str
    refused: str  # the model's refusal (writes, off-topic); "" otherwise
    # --- approval (HITL)
    est_rows: int
    pii_columns: list[str]
    requires_approval: bool  # this SQL needs a human yes (PII or a big result)
    needs_approval: bool  # ... and doesn't have one yet
    approval_reason: str
    approved_sql: str  # the exact string a human approved; only this may run
    rejected: bool
    # --- result
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool
    answer: str
    failure: dict[str, str]  # {"code", "message"} when the run failed for good
    prompt_tags: dict[str, str]
    learned_pair: str | None
