"""A2A contracts, result extraction and the auth allowlist (no network)."""

import asyncio

import pytest
from a2a.helpers.proto_helpers import new_data_part
from a2a.types import Part, TaskState
from pydantic import BaseModel

from multi_agenti_conversation_system.a2a.client import (
    AgentRegistry,
    _Exchange,
    _result_from,
)
from multi_agenti_conversation_system.a2a.contracts import (
    CONTRACT_MEDIA_TYPE,
    MAX_PART_CHARS,
    ContractError,
    ResumeRequest,
    TaskRequest,
    TaskResult,
    contract_extension,
    from_parts,
    to_part,
)
from multi_agenti_conversation_system.mcp.common import BearerAuth


def test_integers_survive_the_wire():
    """The reason contracts use a JSON text part, not a protobuf DataPart."""
    rows = {"rows": [[100, "Chai", 18.5]], "product_id": 100}
    result = TaskResult(status="completed", agent="sql", data=rows)
    back = from_parts([to_part(result)], TaskResult)
    assert back.data["product_id"] == 100
    assert isinstance(back.data["product_id"], int)
    assert isinstance(back.data["rows"][0][0], int)


def test_datapart_would_have_lost_the_integer():
    # Documents the SDK behaviour that drove the encoding choice.
    from a2a.helpers.proto_helpers import get_data_parts

    [value] = get_data_parts([new_data_part({"product_id": 100})])
    assert isinstance(value["product_id"], float)


def test_request_roundtrip_and_defaults():
    request = TaskRequest(sender="supervisor", skill="echo", task="hi")
    back = from_parts([to_part(request)], TaskRequest)
    assert back == request
    assert back.task_id  # generated


@pytest.mark.parametrize(
    "bad",
    [
        {"sender": "s", "skill": "echo", "task": ""},  # empty task
        {"sender": "s", "skill": "echo", "task": "x" * 5000},  # over the size cap
        {"sender": "s", "skill": "echo", "task": "x", "surprise": 1},  # extra field
        {"skill": "echo", "task": "x"},  # missing sender
    ],
)
def test_invalid_requests_are_rejected(bad):
    import json

    part = Part(text=json.dumps(bad), media_type=CONTRACT_MEDIA_TYPE)
    with pytest.raises(ContractError):
        from_parts([part], TaskRequest)


def test_missing_oversized_and_wrong_media_type_parts():
    with pytest.raises(ContractError, match="no application/json part"):
        from_parts([], TaskRequest)
    with pytest.raises(ContractError, match="no application/json part"):
        from_parts([Part(text="{}", media_type="text/plain")], TaskRequest)
    with pytest.raises(ContractError, match="too large"):
        from_parts(
            [Part(text=" " * (MAX_PART_CHARS + 1), media_type=CONTRACT_MEDIA_TYPE)],
            TaskRequest,
        )


def test_contract_extension_publishes_schemas():
    class SqlData(BaseModel):
        max_rows: int

    ext = contract_extension({"sql": SqlData, "echo": None})
    params = ext.params
    assert "task" in params["request_schema"]["properties"]
    assert "status" in params["result_schema"]["properties"]
    assert "value" in params["resume_schema"]["properties"]
    assert "max_rows" in params["skills"]["sql"]["data_schema"]["properties"]


def _exchange(state, *, parts=(), status_parts=()):
    ex = _Exchange(task_id="t1", context_id="c1", state=state)
    ex.parts, ex.status_parts = list(parts), list(status_parts)
    return ex


def test_result_from_completed_artifact_fills_ids():
    part = to_part(TaskResult(status="completed", agent="echo", answer="ok"))
    result = _result_from(
        _exchange(TaskState.TASK_STATE_COMPLETED, parts=[part]), "echo"
    )
    assert result.ok and (result.a2a_task_id, result.context_id) == ("t1", "c1")


def test_result_from_input_required_status_message():
    part = to_part(
        TaskResult(status="input_required", agent="sql", data={"sql": "select 1"})
    )
    result = _result_from(
        _exchange(TaskState.TASK_STATE_INPUT_REQUIRED, status_parts=[part]), "sql"
    )
    assert result.status == "input_required" and result.a2a_task_id == "t1"


def test_result_from_garbage_is_invalid_output_and_retryable():
    junk = Part(text="not json", media_type=CONTRACT_MEDIA_TYPE)
    result = _result_from(
        _exchange(TaskState.TASK_STATE_COMPLETED, parts=[junk]), "echo"
    )
    assert result.error.code == "invalid_output" and result.error.retryable


def test_result_from_failed_task_without_detail():
    result = _result_from(_exchange(TaskState.TASK_STATE_FAILED), "echo")
    assert result.error.code == "failed"


def test_result_from_stream_that_stopped_mid_work():
    result = _result_from(_exchange(TaskState.TASK_STATE_WORKING), "echo")
    assert result.error.code == "invalid_output"


async def test_registry_marks_unreachable_agent_down_without_raising():
    registry = AgentRegistry(["http://127.0.0.1:1"], token="")
    try:
        await asyncio.wait_for(registry.start(), 10)
        assert [a.up for a in registry.agents()] == [False]
        result = await registry.call(
            "anything", TaskRequest(sender="s", skill="x", task="t")
        )
        assert result.error.code == "unavailable"
        waiting = await registry.resume(
            TaskResult(status="completed", agent="a"), {"approved": True}
        )
        assert waiting.error.code == "bad_request"
    finally:
        await registry.close()


async def _asgi_call(app, path):
    sent = []

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(message):
        sent.append(message)

    scope = {"type": "http", "path": path, "headers": []}
    await app(scope, receive, send)
    return sent[0]["status"]


async def test_bearer_auth_public_paths():
    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})

    app = BearerAuth(inner, token="secret", public_paths=("/health", "/card"))
    assert await _asgi_call(app, "/health") == 200
    assert await _asgi_call(app, "/card") == 200
    assert await _asgi_call(app, "/") == 401
    # default allowlist is still just /health
    default = BearerAuth(inner, token="secret")
    assert await _asgi_call(default, "/card") == 401


def test_resume_request_roundtrip():
    back = from_parts(
        [to_part(ResumeRequest(sender="s", value={"approved": True}))], ResumeRequest
    )
    assert back.value == {"approved": True}


async def test_the_http_client_has_no_read_timeout():
    """A silent gap between events must be bounded by the call's own deadline, not by the
    HTTP read timeout (see smoke_a2a.py for the live check)."""
    registry = AgentRegistry(["http://127.0.0.1:1"], token="", timeout_s=2)
    try:
        timeout = registry._http.timeout
        assert timeout.read is None and timeout.connect <= 5
    finally:
        await registry.close()
