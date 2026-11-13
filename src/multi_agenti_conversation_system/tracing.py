"""Langfuse tracing: one lazily built client, a mask, and trace context across processes.

    with observation("plan", as_type="span") as span: ...   # no-op when tracing is off

Across the A2A boundary (G20) the caller injects `current_trace_ctx()` into the message
metadata, and the agent opens its own span with `observation(..., trace_ctx=ctx)`, so the
agent's spans become children of the caller's span in the same Langfuse trace instead of
separate traces.

Tracing is off when `TRACING_ENABLED` is false or the Langfuse keys are empty: every helper
then does nothing, so the system runs (and unit tests run) without Langfuse. The client is
created on first use, not on import, so importing this module never talks to the network.

Everything sent to Langfuse passes through `guardrails.pii.mask` (secrets and PII).
"""

import logging
import re
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from typing import Any

from langfuse import Langfuse, propagate_attributes

from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.guardrails.pii import mask

logger = logging.getLogger(__name__)

_TRACE_ID = re.compile(r"[0-9a-f]{32}")
_SPAN_ID = re.compile(r"[0-9a-f]{16}")

_lock = threading.Lock()
_client: Langfuse | None = None
_client_built = False


def enabled() -> bool:
    return bool(
        settings.TRACING_ENABLED
        and settings.LANGFUSE_PUBLIC_KEY
        and settings.LANGFUSE_SECRET_KEY
    )


def get_langfuse() -> Langfuse | None:
    """The shared client, or None when tracing is off. Built once, on first use."""
    global _client, _client_built

    if not enabled():
        return None
    with _lock:
        if not _client_built:
            _client = Langfuse(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                base_url=settings.LANGFUSE_BASE_URL,
                mask=mask,
            )
            _client_built = True
    return _client


def flush() -> None:
    """Send buffered spans now. Blocking: call it with `asyncio.to_thread` from async code."""
    if _client is not None:
        _client.flush()


def shutdown() -> None:
    """Flush and stop the exporter (call once when a process ends)."""
    if _client is not None:
        _client.shutdown()


def valid_trace_ctx(ctx: Any) -> dict[str, str] | None:
    """`ctx` if it is a usable remote parent (32-hex trace id, optional 16-hex span id).

    Anything else (a typo, a test value, a hostile caller) is ignored, never fatal: the
    agent then simply starts a new trace.
    """
    if not isinstance(ctx, dict):
        return None
    trace_id, span_id = ctx.get("trace_id"), ctx.get("parent_span_id")
    if not isinstance(trace_id, str) or not _TRACE_ID.fullmatch(trace_id):
        return None
    if span_id is None:
        return {"trace_id": trace_id}
    if not isinstance(span_id, str) or not _SPAN_ID.fullmatch(span_id):
        return None
    return {"trace_id": trace_id, "parent_span_id": span_id}


def current_trace_ctx() -> dict[str, str]:
    """The trace and span we are inside right now, ready to send to another process."""
    client = get_langfuse()
    if client is None:
        return {}
    trace_id, span_id = (
        client.get_current_trace_id(),
        client.get_current_observation_id(),
    )
    if not trace_id:
        return {}
    return {"trace_id": trace_id, **({"parent_span_id": span_id} if span_id else {})}


@contextmanager
def observation(
    name: str,
    *,
    as_type: str = "span",
    trace_ctx: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Iterator[Any]:
    """Open a Langfuse observation (span, generation, agent, ...) around a block.

    Yields the observation (call `.update(output=...)` on it) or None when tracing is off.
    With `trace_ctx` the observation becomes a child of that remote span; without it, a
    child of the current one, or a new trace if there is none.
    """
    client = get_langfuse()
    if client is None:
        yield None
        return
    with client.start_as_current_observation(
        name=name,
        as_type=as_type,
        trace_context=valid_trace_ctx(trace_ctx),
        **kwargs,
    ) as obs:
        yield obs


# What replaces the input and output of an LLM call whose data touched PII columns.
OMITTED = "[omitted: the result touches personal data]"


async def generation(
    call: Callable[[Any], Awaitable[Any]],
    messages: Any,
    name: str,
    *,
    prompt_link: Any = None,
    model: str | None = None,
    sensitive: bool = False,
) -> Any:
    """Run an LLM call inside a Langfuse generation, linked to the prompt version.

    `sensitive` keeps the input and output out of the trace entirely: the mask can't
    recognise names and addresses, so data from PII columns must not be sent there.
    """
    with observation(
        name,
        as_type="generation",
        prompt=prompt_link,
        model=model,
        input=OMITTED if sensitive else messages,
    ) as span:
        out = await call(messages)
        if span:
            shown = out.model_dump() if hasattr(out, "model_dump") else out
            span.update(output=OMITTED if sensitive else shown)
    return out


@contextmanager
def session(session_id: str, user_id: str | None = None) -> Iterator[None]:
    """Tag every observation opened inside with a Langfuse session (the conversation's
    thread id) and user, so a conversation's turns are grouped in the UI. No-op when off."""
    if get_langfuse() is None:
        yield
        return
    with propagate_attributes(session_id=session_id, user_id=user_id):
        yield
