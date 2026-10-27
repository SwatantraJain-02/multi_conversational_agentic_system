"""Tests for db_server/db.py against the real Northwind database (needs `docker compose up`).

They skip automatically when Postgres is not reachable.
"""

import time

import pytest
from pydantic import ValidationError

from multi_agenti_conversation_system.config import get_db_settings
from multi_agenti_conversation_system.mcp.servers.db_server.db import (
    NorthwindReader,
    QueryError,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def db():
    try:
        conninfo = get_db_settings().conninfo
    except ValidationError:
        pytest.skip("No database settings (is .env set up?)")
    reader = NorthwindReader(conninfo, checkout_timeout_s=5)
    try:
        reader.fetch_rows("SELECT 1", 1)
    except QueryError:
        reader.close()
        pytest.skip("Postgres not reachable (is `docker compose up` running?)")
    yield reader
    reader.close()


def test_fetch_rows_truncates_and_reports_it(db):
    result = db.fetch_rows("SELECT * FROM orders", 5)
    assert result["row_count"] == 5
    assert result["truncated"] is True
    assert len(result["rows"]) == 5
    assert len(result["rows"][0]) == len(result["columns"])


def test_fetch_rows_not_truncated_when_everything_fits(db):
    result = db.fetch_rows("SELECT 1 AS x", 5)
    assert result == {
        "columns": ["x"],
        "rows": [[1]],
        "row_count": 1,
        "truncated": False,
    }


def test_percent_in_sql_is_literal(db):
    result = db.fetch_rows(
        "SELECT customer_id FROM customers WHERE customer_id LIKE 'V%'", 10
    )
    assert result["rows"] == [["VAFFE"], ["VICTE"], ["VINET"]]


def test_values_are_json_safe(db):
    import json

    result = db.fetch_rows("SELECT order_date, freight FROM orders", 3)
    json.dumps(result)  # would raise on date / Decimal
    assert isinstance(result["rows"][0][0], str)


def test_explain_returns_the_row_estimate(db):
    assert db.explain("SELECT * FROM orders") == 830


def test_role_cannot_write(db):
    with pytest.raises(QueryError, match="read-only"):
        db.fetch_rows("DELETE FROM orders WHERE order_id = -1", 5)


def test_unknown_table_is_a_query_error(db):
    with pytest.raises(QueryError, match="UndefinedTable"):
        db.fetch_rows("SELECT * FROM nope", 5)


def test_statement_timeout_cancels_slow_queries(db):
    started = time.monotonic()
    with pytest.raises(QueryError, match="timeout"):
        db.fetch_rows("SELECT pg_sleep(8)", 5)
    assert time.monotonic() - started < 7
