"""The Synthesis agent's graph nodes.

    check_inputs -> [derive_external_question -> delegate_web] -> compose

Peer delegation is deterministic: if there is no usable external result, the agent ALWAYS
asks the Web agent itself, over A2A, with its own task id as `parent_task_id` (an LLM-decided
trigger would be too unreliable for a live demo). The delegated call runs inside this agent's
trace span, so Langfuse shows the edge supervisor > synthesis > web.

If the delegation fails (Web agent down, timeout), the task still succeeds: the answer covers
what is available and says what is missing, and `data.partial` is set. Only when there is
nothing at all to write about does it fail.

External text comes from the internet: it only enters the prompt in the EXTERNAL block,
which the prompt marks as untrusted data.
"""

import json
import logging
from typing import Any

from langgraph.runtime import Runtime

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import (
    AgentError,
    TaskConstraints,
    TaskRequest,
    TaskResult,
)
from multi_agenti_conversation_system.agents.synthesis.context import SynthesisContext
from multi_agenti_conversation_system.agents.synthesis.state import SynthesisState
from multi_agenti_conversation_system.agents.web.text import strip_invalid_citations
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.prompts import aget_prompt

logger = logging.getLogger(__name__)

ROWS_IN_PROMPT = 20
Rt = Runtime[SynthesisContext]


def usable(result: dict[str, Any] | None) -> bool:
    """A result we can build on: it completed and said something."""
    return (
        bool(result)
        and result["status"] == "completed"
        and bool(result["answer"].strip())
    )


def _why_unusable(result: dict[str, Any] | None) -> str:
    if not result:
        return "not provided"
    if result["status"] == "failed" and result.get("error"):
        return f"{result['error']['code']}: {result['error']['message']}"
    return f"status {result['status']}, no answer"


# ------------------------------------------------------------------------------- nodes
def check_inputs(state: SynthesisState) -> dict[str, Any]:
    """Decide what is missing. `delegate` is the (deterministic) trigger for asking Web."""
    gaps = []
    if not usable(state.get("internal")):
        gaps.append(f"internal data ({_why_unusable(state.get('internal'))})")
    external_ok = usable(state.get("external"))
    return {"delegate": not external_ok, "gaps": gaps}


def route_after_check(state: SynthesisState) -> str:
    return "derive_external_question" if state["delegate"] else "compose"


async def derive_external_question(
    state: SynthesisState, runtime: Rt
) -> dict[str, Any]:
    """The external side of the question, for the Web agent. Falls back to the question."""
    if state.get("external_question"):
        return {}
    prompt = await aget_prompt("synthesis/external-question")
    try:
        derived = await tracing.generation(
            runtime.context.derive_question,
            prompt.compile(question=state["question"]),
            "derive-external-question",
            prompt_link=prompt.link,
            model=settings.CHAT_MODEL,
        )
        question = " ".join(str(derived).split())[:500]
    except Exception as exc:  # noqa: BLE001 - the original question is a fine fallback
        logger.warning(
            "could not derive the external question (%s), using the original", exc
        )
        question = ""
    return {
        "external_question": question or state["question"],
        "prompt_tags": {**state.get("prompt_tags", {}), prompt.name: prompt.tag},
    }


async def delegate_web(state: SynthesisState, runtime: Rt) -> dict[str, Any]:
    """Ask the Web agent over A2A. A failure is recorded as a gap, never raised."""
    ctx = runtime.context
    result: TaskResult = await ctx.registry.call(
        ctx.web_agent,
        TaskRequest(
            sender="synthesis",
            parent_task_id=state["task_id"],
            skill=ctx.web_skill,
            task=state["external_question"],
            constraints=TaskConstraints(timeout_s=ctx.delegate_timeout_s),
        ),
    )
    if usable(result.model_dump(mode="json")):
        return {"external": result.model_dump(mode="json"), "delegated": True}
    reason = (
        f"{result.error.code}: {result.error.message}"
        if result.error
        else result.status
    )
    return {
        "delegated": True,
        "delegation_error": reason,
        "gaps": [
            *state.get("gaps", []),
            f"external market data (the Web agent: {reason})",
        ],
    }


def _internal_block(internal: dict[str, Any] | None) -> str:
    if not usable(internal):
        return "(not available)"
    data = internal["data"]
    lines = [f"Answer: {internal['answer']}"]
    if internal.get("sql"):
        lines.append(f"SQL used: {internal['sql']}")
    if data.get("rows"):
        lines.append(f"Columns: {', '.join(data.get('columns', []))}")
        lines.append(
            f"Rows (first {ROWS_IN_PROMPT}): {json.dumps(data['rows'][:ROWS_IN_PROMPT], default=str)}"
        )
    return "\n".join(lines)


def _external_block(external: dict[str, Any] | None) -> str:
    if not usable(external):
        return "(not available)"
    sources = "\n".join(
        f"[{c['n']}] {c.get('title') or c['url']} ({c['url']})"
        for c in external["citations"]
    )
    return f"Answer: {external['answer']}\nSources:\n{sources or '(none)'}"


async def compose(state: SynthesisState, runtime: Rt) -> dict[str, Any]:
    internal, external = state.get("internal"), state.get("external")
    internal_ok, external_ok = usable(internal), usable(external)
    if not internal_ok and not external_ok:
        raise AgentError(
            "unavailable",
            "nothing to synthesise: " + "; ".join(state["gaps"] or ["no inputs"]),
            retryable=True,
        )
    gaps = list(state.get("gaps", []))
    prompt = await aget_prompt("synthesis/compare")
    messages = prompt.compile(
        question=state["question"],
        internal=_internal_block(internal),
        external=_external_block(external),
        gaps="; ".join(gaps) or "none",
    )
    # Rows from PII columns must not reach a trace: the mask can't recognise names.
    sensitive = bool(internal_ok and internal["data"].get("pii_columns"))
    raw = await tracing.generation(
        runtime.context.compose,
        messages,
        "compose",
        prompt_link=prompt.link,
        model=settings.CHAT_MODEL,
        sensitive=sensitive,
    )
    valid = {c["n"] for c in external["citations"]} if external_ok else set()
    answer, cited = strip_invalid_citations(str(raw), valid)
    if gaps:
        answer = f"Note: {'; '.join(gaps)} could not be included.\n\n{answer}"
    return {
        "answer": answer,
        "citations": [
            c for c in (external["citations"] if external_ok else []) if c["n"] in cited
        ],
        "prompt_tags": {**state.get("prompt_tags", {}), prompt.name: prompt.tag},
    }
