"""The only module that talks to the Northwind database, always as the read-only role.

Callers must run the SQL through guardrails.sql_validator first. The read-only role
(default_transaction_read_only, statement_timeout=5s) is the last line of defence if they don't.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any
from uuid import UUID

import psycopg
from psycopg_pool import ConnectionPool

from multi_agenti_conversation_system.config import settings

__pool: ConnectionPool | None = None


class QueryError(Exception):
    """A database failure with a short message that is safe to show to the agent."""


def get_pool() -> ConnectionPool:
    global __pool

    if __pool is None:
        __pool = ConnectionPool(
            settings.northwind_conninfo, min_size=1, max_size=5, open=False
        )
        __pool.open(wait=True, timeout=10)

    return __pool


def close_pool() -> None:
    global __pool

    if __pool is not None:
        __pool.close()
        __pool = None


def to_jsonable(value: Any) -> Any:
    """Convert a value coming out of Postgres into something json.dumps can handle."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (timedelta, UUID)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"<binary {len(bytes(value))} bytes>"
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)


def _short_message(exc: psycopg.Error) -> str:
    # First line only: the full text can carry server internals.
    return f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"


def fetch_rows(sql: str, max_rows: int) -> dict[str, Any]:
    """Run a query and return {columns, rows, row_count, truncated}.

    Fetches one row more than max_rows so we can tell whether the result was cut off.
    """
    try:
        with get_pool().connection() as conn, conn.cursor() as cur:
            # No second argument: a '%' in the SQL (e.g. LIKE 'V%') is then plain text.
            cur.execute(sql)
            if cur.description is None:
                raise QueryError("The statement did not return rows")
            columns = [col.name for col in cur.description]
            fetched = cur.fetchmany(max_rows + 1)
    except psycopg.Error as exc:
        raise QueryError(_short_message(exc)) from exc

    truncated = len(fetched) > max_rows
    rows = [[to_jsonable(v) for v in row] for row in fetched[:max_rows]]
    return {
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": truncated,
    }


def explain(sql: str) -> int:
    """Return the planner's row estimate. EXPLAIN without ANALYZE does not run the query."""
    try:
        with get_pool().connection() as conn, conn.cursor() as cur:
            cur.execute("EXPLAIN (FORMAT JSON) " + sql)
            plan = cur.fetchone()[0]  # psycopg already parsed the JSON into a list
    except psycopg.Error as exc:
        raise QueryError(_short_message(exc)) from exc
    return int(plan[0]["Plan"]["Plan Rows"])
