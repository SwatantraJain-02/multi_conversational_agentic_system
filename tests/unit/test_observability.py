import logging

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from multi_agenti_conversation_system import observability
from multi_agenti_conversation_system.config import ObservabilitySettings
from multi_agenti_conversation_system.mcp.servers.db_server.db import NorthwindReader
from multi_agenti_conversation_system.mcp.servers.db_server.server import create_server
from multi_agenti_conversation_system.mcp.servers.db_server.service import (
    DbToolService,
    ToolLimits,
)
from tests.unit.fakes import FakeIndex, FakeReader


@pytest.fixture
def spans(monkeypatch) -> InMemorySpanExporter:
    """Route observability.span() to a private provider, leaving the global one alone."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(observability.trace, "get_tracer", provider.get_tracer)
    return exporter


def test_log_event_formats_key_value_pairs_and_skips_none(caplog):
    logger = logging.getLogger("t")
    with caplog.at_level(logging.INFO, logger="t"):
        observability.log_event(
            logger, "tool_call", tool="x", ok=True, row_count=None, msg="a b"
        )
    assert caplog.messages == ['tool_call tool=x ok=True msg="a b"']


def test_span_records_attributes_and_skips_none(spans):
    with observability.span("work", {"k": 3, "domain": None}) as s:
        s.set_attribute("hits", 2)
    (finished,) = spans.get_finished_spans()
    assert finished.name == "work"
    assert dict(finished.attributes) == {"k": 3, "hits": 2}


def test_span_marks_errors_without_recording_the_message(spans):
    with pytest.raises(ValueError), observability.span("work"):
        raise ValueError("secret SELECT text")
    (finished,) = spans.get_finished_spans()
    assert finished.status.status_code.name == "ERROR"
    assert finished.attributes["error.type"] == "ValueError"
    assert "secret" not in str(finished.attributes) and not finished.events


def test_tracing_stays_off_without_an_exporter():
    assert observability.configure_tracing(ObservabilitySettings()) is None


def test_trace_id_is_none_outside_a_span():
    assert observability.current_trace_id() is None


async def test_each_tool_call_logs_one_line_without_sql(catalog, caplog):
    service = DbToolService(catalog, FakeReader(rows=2), FakeIndex(), ToolLimits())
    server = create_server(service)
    with caplog.at_level(logging.INFO):
        await server.call_tool(
            "run_select_query", {"sql": "SELECT order_id FROM orders"}
        )
        await server.call_tool(
            "run_select_query", {"sql": "SELECT secret_marker FROM nope"}
        )
    lines = [m for m in caplog.messages if m.startswith("tool_call")]
    assert len(lines) == 2
    assert "tool=run_select_query ok=True" in lines[0] and "row_count=2" in lines[0]
    assert "ok=False" in lines[1] and "error_type=ToolInputError" in lines[1]
    assert "duration_ms=" in lines[0]
    everything = "\n".join(caplog.messages)
    assert "secret_marker" not in everything and "SELECT" not in everything


def test_reader_spans_carry_counts_not_sql(spans, monkeypatch):
    reader = NorthwindReader("host=unused")
    monkeypatch.setattr(reader, "_ensure_open", lambda: None)

    # Drive fetch_rows' span through a stubbed _run so no database is needed.
    monkeypatch.setattr(
        reader,
        "_run",
        lambda sql, fetch, timeout_s=None: (["x"], [(1,), (2,), (3,)]),
    )
    reader.fetch_rows("SELECT secret_marker FROM orders", 2)
    (finished,) = spans.get_finished_spans()
    assert finished.name == "db.fetch_rows"
    assert dict(finished.attributes) == {
        "db.system": "postgresql",
        "row_count": 2,
        "truncated": True,
    }
    assert "secret_marker" not in str(finished.attributes)
