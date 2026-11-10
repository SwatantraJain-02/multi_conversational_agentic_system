"""Wiring of the SQL agent graph, and its two translations to and from A2A."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from multi_agenti_conversation_system.a2a.contracts import (
    AgentError,
    TaskRequest,
    TaskResult,
)
from multi_agenti_conversation_system.agents.sql import nodes
from multi_agenti_conversation_system.agents.sql.context import SqlContext
from multi_agenti_conversation_system.agents.sql.state import SqlState
from multi_agenti_conversation_system.config import settings

AGENT = "sql"
SKILL = "answer_sql_question"


def _transient(exc: Exception) -> bool:
    return isinstance(exc, AgentError) and exc.retryable


# Only transport-level failures are retried, and only once: the chat client already retries
# on its own, so more attempts here would multiply the waiting.
RETRY = RetryPolicy(max_attempts=2, initial_interval=0.5, retry_on=_transient)


def build_graph(checkpointer):
    graph = StateGraph(SqlState, context_schema=SqlContext)
    graph.add_node("retrieve_schema", nodes.retrieve_schema, retry_policy=RETRY)
    graph.add_node("generate_sql", nodes.generate_sql, retry_policy=RETRY)
    graph.add_node("validate", nodes.validate)
    graph.add_node("approval_check", nodes.approval_check, retry_policy=RETRY)
    graph.add_node("hitl_gate", nodes.hitl_gate)
    graph.add_node("execute", nodes.execute, retry_policy=RETRY)
    graph.add_node("explain_result", nodes.explain_result)
    graph.add_node("write_sql_pair", nodes.write_sql_pair)
    graph.add_node("fail_end", nodes.fail_end)

    graph.add_edge(START, "retrieve_schema")
    graph.add_edge("retrieve_schema", "generate_sql")
    graph.add_conditional_edges("generate_sql", nodes.after_generate, ["validate", END])
    graph.add_conditional_edges(
        "validate", nodes.after_validate, ["approval_check", "generate_sql", "fail_end"]
    )
    graph.add_conditional_edges(
        "approval_check",
        nodes.after_approval_check,
        ["hitl_gate", "execute", "generate_sql", "fail_end"],
    )
    graph.add_conditional_edges("hitl_gate", nodes.after_hitl, ["execute", END])
    graph.add_conditional_edges(
        "execute", nodes.after_execute, ["explain_result", "generate_sql", "fail_end"]
    )
    graph.add_edge("explain_result", "write_sql_pair")
    graph.add_edge("write_sql_pair", END)
    graph.add_edge("fail_end", END)
    return graph.compile(checkpointer=checkpointer)


def build_input(request: TaskRequest) -> dict[str, Any]:
    """Graph input for a new task. Resets EVERY per-task field: all tasks in one A2A
    context share one checkpoint, so anything left out would leak from the previous task."""
    asked = request.constraints.max_rows
    return {
        "question": request.task,
        "domain": request.context.session.get("selected_domain"),
        "max_rows": min(asked, settings.MAX_ROWS) if asked else settings.MAX_ROWS,
        "max_retries": settings.SQL_MAX_RETRIES,
        "schema_docs": [],
        "examples": [],
        "degraded": False,
        "sql": "",
        "explanation": "",
        "tables_used": [],
        "attempts": 0,
        "error": "",
        "failed_sql": "",
        "refused": "",
        "est_rows": 0,
        "pii_columns": [],
        "requires_approval": False,
        "needs_approval": False,
        "approval_reason": "",
        "approved_sql": "",
        "rejected": False,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "answer": "",
        "failure": {},
        "prompt_tags": {},
        "learned_pair": None,
    }


def to_result(state: dict[str, Any], request: TaskRequest) -> TaskResult:
    """The A2A result. A refusal or a declined approval is a *completed* task that says so
    (`data.refused` / `data.rejected`): the caller must not re-plan around a decision."""
    tags = state.get("prompt_tags", {})
    if failure := state.get("failure"):
        return TaskResult.failure(AGENT, failure["code"], failure["message"])
    if state.get("refused"):
        return TaskResult(
            status="completed",
            agent=AGENT,
            answer=f"I can't do that: {state['refused']}",
            data={"refused": True, "attempts": state["attempts"]},
            prompt_versions=tags,
        )
    if state.get("rejected"):
        return TaskResult(
            status="completed",
            agent=AGENT,
            answer="The query was not run: approval was declined.",
            sql=state["sql"],
            data={"rejected": True, "attempts": state["attempts"]},
            prompt_versions=tags,
        )
    return TaskResult(
        status="completed",
        agent=AGENT,
        answer=state["answer"],
        sql=state["sql"],
        data={
            "columns": state["columns"],
            "rows": state["rows"],
            "row_count": state["row_count"],
            "truncated": state["truncated"],
            "tables_used": state["tables_used"],
            "pii_columns": state["pii_columns"],
            "attempts": state["attempts"],
            "degraded": state["degraded"],
            "learned_pair": state.get("learned_pair"),
        },
        prompt_versions=tags,
    )
