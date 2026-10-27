"""DbToolService with fake adapters: no database, Chroma, Ollama or MCP transport needed."""

import pytest

from multi_agenti_conversation_system.catalog import SchemaCatalog
from multi_agenti_conversation_system.mcp.contracts.db import (
    ExplainResponse,
    FewShotsResponse,
    ListTablesResponse,
    SearchSchemaResponse,
    SelectResponse,
)
from multi_agenti_conversation_system.mcp.servers.db_server.db import (
    DatabaseUnavailable,
    QueryError,
)
from multi_agenti_conversation_system.mcp.servers.db_server.schema_index import (
    SearchUnavailable,
)
from multi_agenti_conversation_system.mcp.servers.db_server.service import (
    BackendUnavailable,
    DbToolService,
    QueryFailed,
    ToolInputError,
    ToolLimits,
)
from tests.unit.fakes import FakeIndex, FakeReader

LIMITS = ToolLimits(max_rows=10, max_sql_chars=200, max_question_chars=50, max_k=5)


@pytest.fixture
def make(catalog: SchemaCatalog):
    def _make(reader=None, index=None) -> DbToolService:
        return DbToolService(
            catalog, reader or FakeReader(), index or FakeIndex(), LIMITS
        )

    return _make


# ------------------------------------------------------------------ catalog tools
async def test_list_tables_covers_the_catalog(make, catalog):
    result = await make().list_tables()
    assert isinstance(result, ListTablesResponse) and result.ok
    assert len(result.tables) == len(catalog)
    orders = next(t for t in result.tables if t.table == "orders")
    assert orders.has_pii is True and orders.domain == "sales"
    assert "\n" not in orders.description


async def test_get_schema_is_case_insensitive(make):
    result = await make().get_schema(["Orders", "shippers"])
    assert set(result.schemas) == {"orders", "shippers"}


@pytest.mark.parametrize("tables", [[], ["orders"] * 15])
async def test_get_schema_bounds_come_from_the_catalog(make, catalog, tables):
    with pytest.raises(ToolInputError, match=f"between 1 and {len(catalog)}"):
        await make().get_schema(tables)


async def test_get_schema_rejects_unknown_tables(make):
    with pytest.raises(ToolInputError, match="Unknown tables: \\['nope'\\]"):
        await make().get_schema(["nope"])


# ------------------------------------------------------------------ search tools
@pytest.mark.parametrize(
    ("question", "k", "domain", "message"),
    [
        ("", 3, None, "must not be empty"),
        ("   ", 3, None, "must not be empty"),
        ("x" * 51, 3, None, "too long \\(max 50"),
        ("q", 0, None, "between 1 and 5"),
        ("q", 6, None, "between 1 and 5"),
        ("q", 3, "bogus", "Unknown domain 'bogus'"),
    ],
)
async def test_search_input_checks(make, question, k, domain, message):
    service = make()
    for tool in (service.search_schema, service.get_few_shots):
        with pytest.raises(ToolInputError, match=message):
            await tool(question, k, domain)


async def test_search_returns_hits(make):
    service = make()
    schema = await service.search_schema("q", 3, "sales")
    assert isinstance(schema, SearchSchemaResponse)
    assert schema.tables[0].table == "orders"
    examples = await service.get_few_shots("q", 3, None)
    assert isinstance(examples, FewShotsResponse)
    assert examples.examples[0].sql == "SELECT 1"


async def test_search_outage_becomes_backend_unavailable(make):
    service = make(index=FakeIndex(SearchUnavailable("ConnectError")))
    with pytest.raises(
        BackendUnavailable, match="Schema search is unavailable: ConnectError"
    ):
        await service.search_schema("q", 3, None)
    with pytest.raises(
        BackendUnavailable, match="Example search is unavailable: ConnectError"
    ):
        await service.get_few_shots("q", 3, None)


# ------------------------------------------------------------------ SQL tools
async def test_explain_reports_estimate_and_pii(make):
    result = await make().explain_query("SELECT * FROM customers")
    assert isinstance(result, ExplainResponse)
    assert result.estimated_rows == 42
    assert result.touches_pii is True
    assert "customers.phone" in result.pii_columns
    assert result.tables == ["customers"]
    assert "LIMIT 10" in result.sql


async def test_explain_without_pii(make):
    result = await make().explain_query("SELECT order_id FROM orders")
    assert result.touches_pii is False and result.pii_columns == []


@pytest.mark.parametrize(
    "sql", ["DELETE FROM orders", "SELECT 1; DROP TABLE orders", "SELECT * FROM nope"]
)
async def test_rejected_sql_never_reaches_the_database(make, sql):
    reader = FakeReader()
    service = make(reader=reader)
    with pytest.raises(ToolInputError):
        await service.run_select_query(sql, None)
    with pytest.raises(ToolInputError):
        await service.explain_query(sql)
    assert reader.calls == []


async def test_sql_length_limit(make):
    with pytest.raises(ToolInputError, match="too long \\(max 200"):
        await make().run_select_query("SELECT 1 " + "x" * 200, None)


@pytest.mark.parametrize("max_rows", [0, 11, -1])
async def test_max_rows_is_capped_by_the_server_limit(make, max_rows):
    with pytest.raises(ToolInputError, match="max_rows must be between 1 and 10"):
        await make().run_select_query("SELECT 1", max_rows)


async def test_query_is_limited_to_one_extra_row_and_reader_gets_the_limit(make):
    """The SQL asks for limit+1 rows, so the reader can tell whether the result was cut off."""
    reader = FakeReader(rows=50)
    result = await make(reader=reader).run_select_query("SELECT * FROM orders", 5)
    assert isinstance(result, SelectResponse)
    sql, max_rows = reader.calls[0][1]
    assert "LIMIT 6" in sql and max_rows == 5
    assert result.row_count == 5 and result.truncated is True


async def test_default_limit_is_the_server_max(make):
    reader = FakeReader()
    await make(reader=reader).run_select_query("SELECT * FROM orders", None)
    assert reader.calls[0][1][1] == 10


async def test_run_select_flags_pii(make):
    result = await make().run_select_query("SELECT phone FROM customers", None)
    assert result.pii_columns == ["customers.phone"]


async def test_database_outage_is_backend_unavailable(make):
    service = make(
        reader=FakeReader(DatabaseUnavailable("Database unavailable: no connection"))
    )
    with pytest.raises(BackendUnavailable, match="Database unavailable"):
        await service.run_select_query("SELECT 1", None)
    with pytest.raises(BackendUnavailable, match="Database unavailable"):
        await service.explain_query("SELECT 1")


async def test_query_error_is_query_failed_with_the_database_message(make):
    service = make(
        reader=FakeReader(QueryError('UndefinedColumn: column "x" does not exist'))
    )
    with pytest.raises(QueryFailed, match="UndefinedColumn"):
        await service.run_select_query("SELECT x FROM orders", None)


# ------------------------------------------------------------------ readiness
async def test_ready_when_every_check_passes(make):
    service = make()
    service.extra_checks["ollama"] = lambda: True
    assert await service.readiness() == {
        "postgres": True,
        "chroma": True,
        "schema_index": True,
        "ollama": True,
    }


async def test_one_failing_dependency_is_reported(make):
    service = make(reader=FakeReader(DatabaseUnavailable("down")))
    checks = await service.readiness()
    assert checks["postgres"] is False and checks["chroma"] is True


async def test_a_raising_or_hanging_check_counts_as_not_ready(make, catalog):
    import time

    def hangs() -> bool:
        time.sleep(0.3)
        return True

    def raises() -> bool:
        raise RuntimeError("boom")

    service = DbToolService(
        catalog,
        FakeReader(),
        FakeIndex(),
        LIMITS,
        extra_checks={"slow": hangs, "broken": raises},
        check_timeout_s=0.1,
    )
    checks = await service.readiness()
    assert checks["slow"] is False and checks["broken"] is False
    assert checks["postgres"] is True
