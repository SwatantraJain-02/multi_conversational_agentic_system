"""Wiring of the Synthesis agent graph, and its two translations to and from A2A."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy
from pydantic import ValidationError

from multi_agenti_conversation_system.a2a.contracts import (
    AgentError,
    ContractError,
    SynthesisData,
    TaskRequest,
    TaskResult,
)
from multi_agenti_conversation_system.agents.synthesis import nodes
from multi_agenti_conversation_system.agents.synthesis.context import SynthesisContext
from multi_agenti_conversation_system.agents.synthesis.state import SynthesisState

AGENT = "synthesis"
SKILL = "synthesise_findings"


def _transient(exc: Exception) -> bool:
    return isinstance(exc, AgentError) and exc.retryable


RETRY = RetryPolicy(max_attempts=2, initial_interval=0.5, retry_on=_transient)


def build_graph(checkpointer):
    graph = StateGraph(SynthesisState, context_schema=SynthesisContext)
    graph.add_node("check_inputs", nodes.check_inputs)
    graph.add_node("derive_external_question", nodes.derive_external_question)
    graph.add_node("delegate_web", nodes.delegate_web)
    graph.add_node("compose", nodes.compose, retry_policy=RETRY)

    graph.add_edge(START, "check_inputs")
    graph.add_conditional_edges(
        "check_inputs", nodes.route_after_check, ["derive_external_question", "compose"]
    )
    graph.add_edge("derive_external_question", "delegate_web")
    graph.add_edge("delegate_web", "compose")
    graph.add_edge("compose", END)
    return graph.compile(checkpointer=checkpointer)


def build_input(request: TaskRequest) -> dict[str, Any]:
    """Validate `request.data` as `SynthesisData` and reset every per-task field."""
    try:
        data = SynthesisData.model_validate(request.data)
    except ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "(root)"
        raise ContractError(f"invalid SynthesisData: {where}: {first['msg']}") from exc
    if data.internal is None and data.external is None and not data.external_question:
        raise ContractError(
            "synthesise_findings needs `internal`, `external` or an `external_question`"
        )
    return {
        "question": request.task,
        "task_id": request.task_id,
        "internal": data.internal.model_dump(mode="json") if data.internal else None,
        "external": data.external.model_dump(mode="json") if data.external else None,
        "external_question": data.external_question,
        "delegate": False,
        "delegated": False,
        "delegation_error": "",
        "gaps": [],
        "answer": "",
        "citations": [],
        "prompt_tags": {},
    }


def to_result(state: dict[str, Any], request: TaskRequest) -> TaskResult:
    internal = state.get("internal")
    internal_ok = nodes.usable(internal)
    external_ok = nodes.usable(state.get("external"))
    missing = [
        name
        for name, ok in (("internal", internal_ok), ("external", external_ok))
        if not ok
    ]
    return TaskResult(
        status="completed",
        agent=AGENT,
        answer=state["answer"],
        citations=state["citations"],
        sql=internal.get("sql") if internal_ok else None,
        data={
            "delegated": state["delegated"],
            "partial": bool(missing),
            "missing": missing,
            "delegation_error": state["delegation_error"] or None,
            "external_question": state.get("external_question"),
            "external_confidence": (state["external"] or {})
            .get("data", {})
            .get("confidence")
            if external_ok
            else None,
        },
        prompt_versions={
            **(internal.get("prompt_versions", {}) if internal_ok else {}),
            **(state["external"].get("prompt_versions", {}) if external_ok else {}),
            **state["prompt_tags"],
        },
    )
