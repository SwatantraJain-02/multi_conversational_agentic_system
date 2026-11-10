"""Chat-model calls for agents: structured output and plain text.

Any model failure (network, rate limit, unparseable JSON) after the chat client's own
retries becomes a retryable `unavailable` error: the graph's RetryPolicy tries once more,
then the caller sees `unavailable`, never a stack trace.
"""

from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel

from multi_agenti_conversation_system.a2a.contracts import AgentError

Messages = list[dict[str, str]]


def _unavailable(exc: Exception) -> AgentError:
    return AgentError(
        "unavailable",
        f"language model unavailable ({type(exc).__name__})",
        retryable=True,
    )


def structured[M: BaseModel](
    llm: Any, schema: type[M]
) -> Callable[[Messages], Awaitable[M]]:
    """`messages -> schema instance` (JSON-schema structured output, G1)."""
    bound = llm.with_structured_output(schema, method="json_schema")

    async def call(messages: Messages) -> M:
        try:
            out = await bound.ainvoke(messages)
            return out if isinstance(out, schema) else schema(**out)
        except Exception as exc:
            raise _unavailable(exc) from exc

    return call


def text(llm: Any) -> Callable[[Messages], Awaitable[str]]:
    """`messages -> reply text`."""

    async def call(messages: Messages) -> str:
        try:
            reply = await llm.ainvoke(messages)
        except Exception as exc:
            raise _unavailable(exc) from exc
        return reply.content if isinstance(reply.content, str) else str(reply.content)

    return call
