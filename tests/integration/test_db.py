"""Tests for db_server/db.py against the real Northwind database (needs `docker compose up`).

They skip automatically when Postgres is not reachable.
"""

import time

import pytest
from psycopg_pool import PoolTimeout

from multi_agenti_conversation_system.mcp.servers.db_server import db

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def pool():
    try:
        db.get_pool()
    except PoolTimeout:  # the pool gives up after 10 s when Postgres is not running
        pytest.skip("Postgres not reachable (is `docker compose up` running?)")
    yield
    db.close_pool()


def test_fetch_rows_truncates_and_reports_it():
    result = db.fetch_rows("SELECT * FROM orders", 5)
    assert result["row_count"] == 5
    assert result["truncated"] is True
    assert len(result["rows"]) == 5
    assert len(result["rows"][0]) == len(result["columns"])


def test_fetch_rows_not_truncated_when_everything_fits():
    result = db.fetch_rows("SELECT 1 AS x", 5)
    assert result == {
        "columns": ["x"],
        "rows": [[1]],
        "row_count": 1,
        "truncated": False,
    }


def test_percent_in_sql_is_literal():
    result = db.fetch_rows(
        "SELECT customer_id FROM customers WHERE customer_id LIKE 'V%'", 10
    )
    assert result["rows"] == [["VAFFE"], ["VICTE"], ["VINET"]]


def test_values_are_json_safe():
    import json

    result = db.fetch_rows("SELECT order_date, freight FROM orders", 3)
    json.dumps(result)  # would raise on date / Decimal
    assert isinstance(result["rows"][0][0], str)


def test_explain_returns_the_row_estimate():
    assert db.explain("SELECT * FROM orders") == 830


def test_role_cannot_write():
    with pytest.raises(db.QueryError, match="read-only"):
        db.fetch_rows("DELETE FROM orders WHERE order_id = -1", 5)


def test_unknown_table_is_a_query_error():
    with pytest.raises(db.QueryError, match="UndefinedTable"):
        db.fetch_rows("SELECT * FROM nope", 5)


def test_statement_timeout_cancels_slow_queries():
    started = time.monotonic()
    with pytest.raises(db.QueryError, match="timeout"):
        db.fetch_rows("SELECT pg_sleep(8)", 5)
    assert time.monotonic() - started < 7
