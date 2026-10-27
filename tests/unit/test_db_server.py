"""The MCP transport layer over fake adapters: tool registration, typed outputs, HTTP routes."""

import json

import pytest
from starlette.testclient import TestClient

from multi_agenti_conversation_system.mcp.servers.db_server.db import (
    DatabaseUnavailable,
)
from multi_agenti_conversation_system.mcp.servers.db_server.server import (
    create_app,
    create_server,
)
from multi_agenti_conversation_system.mcp.servers.db_server.service import (
    DbToolService,
    ToolLimits,
)
from tests.unit.fakes import FakeIndex, FakeReader

TOOLS = {
    "list_tables",
    "get_schema",
    "search_schema",
    "get_few_shots",
    "explain_query",
    "run_select_query",
}


@pytest.fixture
def service(catalog) -> DbToolService:
    return DbToolService(catalog, FakeReader(), FakeIndex(), ToolLimits(max_k=7))


async def test_every_tool_publishes_an_output_schema(service):
    tools = {t.name: t for t in await create_server(service).list_tools()}
    assert set(tools) == TOOLS
    for tool in tools.values():
        schema = tool.output_schema
        assert schema and schema["type"] == "object", tool.name
        assert {"ok", "error"} <= set(schema["properties"]), tool.name


async def test_descriptions_are_filled_from_limits_and_catalog(service):
    tools = {t.name: t for t in await create_server(service).list_tools()}
    description = tools["search_schema"].description
    assert "k: How many tables to return (1-7)." in description
    assert (
        "domain: Optional filter: sales, customers, products, hr or logistics."
        in description
    )
    assert "{" not in description


async def test_text_content_and_structured_content_carry_the_same_json(service):
    result = await create_server(service).call_tool(
        "get_schema", {"tables": ["orders"]}
    )
    assert json.loads(result.content[0].text) == result.structured_content
    assert result.structured_content["ok"] is True


async def test_tool_errors_come_back_as_ok_false(service):
    result = await create_server(service).call_tool("get_schema", {"tables": ["nope"]})
    data = json.loads(result.content[0].text)
    assert data["ok"] is False and "Unknown tables" in data["error"]
    assert not result.is_error  # an answer the agent reads, not a protocol error


def test_health_and_ready_need_no_token(service):
    client = TestClient(create_app(service, token="secret"))
    assert client.get("/health").json() == {"status": "ok"}
    ready = client.get("/ready")
    assert ready.status_code == 200 and ready.json()["status"] == "ready"
    assert client.post("/mcp", json={}).status_code == 401


def test_ready_is_503_when_a_dependency_is_down(catalog):
    service = DbToolService(
        catalog, FakeReader(DatabaseUnavailable("down")), FakeIndex(), ToolLimits()
    )
    response = TestClient(create_app(service, token="")).get("/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "not_ready",
        "checks": {"postgres": False, "chroma": True, "schema_index": True},
    }
