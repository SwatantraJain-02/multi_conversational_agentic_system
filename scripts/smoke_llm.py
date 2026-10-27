"""Phase 0 check: can the configured chat model do what the agents need?

    uv run python scripts/smoke_llm.py

Checks: (a) tool calling, (b) structured output, (c) no <think> leakage, (d) embeddings.
The prompts are tiny dummies; no project data is sent to the chat model.
"""

import math
import sys
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel

from multi_agenti_conversation_system.config import (
    get_chat_settings,
    get_embedding_settings,
)
from multi_agenti_conversation_system.llm import get_chat_model, get_embedding_model

failures: list[str] = []


def report(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(name)


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny in {city}"


class Route(BaseModel):
    intent: Literal["web", "sql", "hybrid", "direct"]
    reason: str


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.hypot(*a) * math.hypot(*b))


def check_tool_calling() -> None:
    reply = (
        get_chat_model()
        .bind_tools([get_weather])
        .invoke("What is the weather in Paris?")
    )
    calls = reply.tool_calls
    report(
        "(a) emits a tool call",
        bool(calls) and calls[0]["name"] == "get_weather",
        f"{calls[0]['name']}({calls[0]['args']})"
        if calls
        else f"no tool call; text={reply.content!r:.80}",
    )


def check_structured_output() -> None:
    question = "Classify: 'What was our revenue last month?' (company sales data lives in a SQL database)"
    for method in ("json_schema", "function_calling"):
        try:
            result = (
                get_chat_model()
                .with_structured_output(Route, method=method)
                .invoke(question)
            )
        except Exception as exc:  # noqa: BLE001 - a smoke test reports any failure
            report(
                f"(b) structured output via {method}",
                False,
                f"{type(exc).__name__}: {str(exc)[:80]}",
            )
            continue
        report(
            f"(b) structured output via {method}",
            isinstance(result, Route),
            f"intent={getattr(result, 'intent', None)}",
        )


def check_no_think_leak() -> None:
    text = str(get_chat_model().invoke("Reply with one word: hello").content)
    report(
        "(c) no <think> text in the answer",
        "<think>" not in text.lower(),
        repr(text[:60]),
    )


def check_embeddings() -> None:
    embedder = get_embedding_model()
    a, b, c = embedder.embed_documents(
        [
            "The cat sat on the mat.",
            "A kitten is resting on the rug.",
            "Quarterly revenue grew by 8 percent.",
        ]
    )
    similar, different = cosine(a, b), cosine(a, c)
    report(
        "(d) embeddings: similar > unrelated",
        similar > different,
        f"similar={similar:.3f} unrelated={different:.3f} dims={len(a)}",
    )


def main() -> None:
    print(
        f"chat model: {get_chat_settings().CHAT_MODEL}   "
        f"embeddings: {get_embedding_settings().EMBED_MODEL}\n"
    )
    for check in (
        check_tool_calling,
        check_structured_output,
        check_no_think_leak,
        check_embeddings,
    ):
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - a smoke test reports any failure
            report(check.__name__, False, f"{type(exc).__name__}: {str(exc)[:100]}")
    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
