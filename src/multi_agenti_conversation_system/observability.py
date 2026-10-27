"""Logging and tracing setup, plus the helpers adapters use to emit spans and events.

Nothing here runs at import. Composition roots call configure_logging() / configure_tracing();
until then spans are no-ops, so library code can create them unconditionally.

Privacy rule for everything emitted from here: no row values, and no SQL text in span
attributes or INFO logs. Attributes are counts, names and flags.
"""

import logging
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Span, Status, StatusCode

from multi_agenti_conversation_system.config import ObservabilitySettings

_TRACER_NAME = "multi_agenti_conversation_system"


# ------------------------------------------------------------------------------ logging
def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def _format_value(value: Any) -> str:
    text = str(value)
    return f'"{text}"' if not text or any(c in text for c in ' ="') else text


def log_event(
    logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any
) -> None:
    """Log `event key=value ...`, adding trace_id when a span is active. Skips None values."""
    if trace_id := current_trace_id():
        fields["trace_id"] = trace_id
    pairs = " ".join(
        f"{k}={_format_value(v)}" for k, v in fields.items() if v is not None
    )
    logger.log(level, "%s %s", event, pairs)


# ------------------------------------------------------------------------------ tracing
def configure_tracing(cfg: ObservabilitySettings) -> TracerProvider | None:
    """Install a TracerProvider when an exporter is configured; otherwise leave tracing off.

    The MCP SDK already wraps every request in a span and reads incoming trace context, so
    once a provider exists, tool spans (and spans from an agent calling over MCP) show up
    without further code.
    """
    if not cfg.OTEL_EXPORTER_OTLP_ENDPOINT and not cfg.OTEL_CONSOLE_EXPORT:
        return None

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: cfg.OTEL_SERVICE_NAME})
    )
    if cfg.OTEL_EXPORTER_OTLP_ENDPOINT:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )

        endpoint = cfg.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/") + "/v1/traces"
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
        )
    if cfg.OTEL_CONSOLE_EXPORT:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)

    # httpx carries the Chroma and Ollama calls: gives client spans with URL and status, never
    # bodies. psycopg is deliberately not instrumented: it records the SQL text on each span.
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

    HTTPXClientInstrumentor().instrument()
    return provider


@contextmanager
def span(name: str, attributes: Mapping[str, Any] | None = None) -> Iterator[Span]:
    """A child span with the given attributes (None values skipped). Errors mark it failed."""
    tracer = trace.get_tracer(_TRACER_NAME)
    with tracer.start_as_current_span(
        name, record_exception=False, set_status_on_exception=False
    ) as s:
        s.set_attributes({k: v for k, v in (attributes or {}).items() if v is not None})
        try:
            yield s
        except Exception as exc:
            s.set_attribute("error.type", type(exc).__qualname__)
            s.set_status(Status(StatusCode.ERROR))
            raise
