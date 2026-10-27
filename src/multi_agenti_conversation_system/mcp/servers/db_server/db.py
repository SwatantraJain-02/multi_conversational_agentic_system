"""The only module that talks to the Northwind database, always as the read-only role.

Callers must run the SQL through guardrails.sql_validator first. The read-only role
(default_transaction_read_only, statement_timeout=5s) is the last line of defence if they don't.
"""

import threading
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Any, Self
from uuid import UUID

import psycopg
from psycopg_pool import ConnectionPool, PoolTimeout

from multi_agenti_conversation_system.observability import span


class QueryError(Exception):
    """A database failure with a short message that is safe to show to the agent."""


class DatabaseUnavailable(QueryError):
    """No connection could be obtained: Postgres is down or the pool is exhausted."""


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


class NorthwindReader:
    """Read-only queries over a connection pool that is opened on first use."""

    def __init__(
        self,
        conninfo: str,
        *,
        min_size: int = 1,
        max_size: int = 5,
        checkout_timeout_s: float = 10,
    ):
        self._pool = ConnectionPool(
            conninfo,
            min_size=min_size,
            max_size=max_size,
            timeout=checkout_timeout_s,
            open=False,
        )
        self._opened = False
        self._lock = threading.Lock()

    def _ensure_open(self) -> None:
        # Lazy, and without wait=True: a timed-out wait() closes the pool for good, which
        # would leave every later query failing with PoolClosed after one Postgres outage.
        # Opened this way the pool keeps reconnecting in the background, and each query
        # waits at most checkout_timeout_s for a connection.
        with self._lock:
            if not self._opened:
                self._pool.open(wait=False)
                self._opened = True

    def _run(self, sql: str, fetch, timeout_s: float | None = None) -> Any:
        try:
            self._ensure_open()
            with self._pool.connection(timeout=timeout_s) as conn, conn.cursor() as cur:
                # No second argument: a '%' in the SQL (e.g. LIKE 'V%') is then plain text.
                cur.execute(sql)  # type: ignore[arg-type]  # validated SQL, not a literal
                return fetch(cur)
        except PoolTimeout as exc:
            raise DatabaseUnavailable(
                "Database unavailable: no connection within the timeout"
            ) from exc
        except psycopg.Error as exc:
            raise QueryError(_short_message(exc)) from exc

    def fetch_rows(self, sql: str, max_rows: int) -> dict[str, Any]:
        """Run a query and return {columns, rows, row_count, truncated}.

        Fetches one row more than max_rows so we can tell whether the result was cut off.
        """

        def fetch(cur: psycopg.Cursor) -> tuple[list[str], list[tuple[Any, ...]]]:
            if cur.description is None:
                raise QueryError("The statement did not return rows")
            return [col.name for col in cur.description], cur.fetchmany(max_rows + 1)

        with span("db.fetch_rows", {"db.system": "postgresql"}) as s:
            columns, fetched = self._run(sql, fetch)
            truncated = len(fetched) > max_rows
            rows = [[to_jsonable(v) for v in row] for row in fetched[:max_rows]]
            s.set_attributes({"row_count": len(rows), "truncated": truncated})
        return {
            "columns": columns,
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
        }

    def explain(self, sql: str) -> int:
        """Return the planner's row estimate. EXPLAIN without ANALYZE does not run the query."""

        def fetch(cur: psycopg.Cursor) -> Any:
            row = cur.fetchone()
            if row is None:
                raise QueryError("EXPLAIN returned no plan")
            return row[0]  # psycopg already parsed the JSON into a list

        with span("db.explain", {"db.system": "postgresql"}) as s:
            plan = self._run("EXPLAIN (FORMAT JSON) " + sql, fetch)
            estimated = int(plan[0]["Plan"]["Plan Rows"])
            s.set_attribute("estimated_rows", estimated)
        return estimated

    def ping(self, timeout_s: float) -> bool:
        try:
            self._run("SELECT 1", lambda cur: cur.fetchone(), timeout_s=timeout_s)
        except QueryError:
            return False
        return True

    def close(self) -> None:
        self._pool.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
