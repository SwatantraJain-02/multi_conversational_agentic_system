"""Trace-context validation and the off switch (no Langfuse involved)."""

import pytest

from multi_agenti_conversation_system import tracing

TRACE = "0123456789abcdef0123456789abcdef"
SPAN = "0123456789abcdef"


@pytest.mark.parametrize(
    "ctx,expected",
    [
        (
            {"trace_id": TRACE, "parent_span_id": SPAN},
            {"trace_id": TRACE, "parent_span_id": SPAN},
        ),
        ({"trace_id": TRACE}, {"trace_id": TRACE}),
        ({"trace_id": "t-1"}, None),  # a test value, not 32 hex
        ({"trace_id": TRACE.upper()}, None),
        ({"trace_id": TRACE, "parent_span_id": "short"}, None),
        ({"trace_id": 5}, None),
        ({}, None),
        (None, None),
        ("not a dict", None),
    ],
)
def test_valid_trace_ctx(ctx, expected):
    assert tracing.valid_trace_ctx(ctx) == expected


def test_everything_is_a_no_op_when_tracing_is_off():
    assert tracing.enabled() is False  # tests/conftest.py
    assert tracing.get_langfuse() is None
    assert tracing.current_trace_ctx() == {}
    with tracing.observation(
        "x", as_type="generation", trace_ctx={"trace_id": TRACE}
    ) as obs:
        assert obs is None
    tracing.flush()
    tracing.shutdown()
