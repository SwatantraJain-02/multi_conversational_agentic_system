"""Wiring of the supervisor graph and the per-turn input."""

from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from multi_agenti_conversation_system.supervisor import guard_nodes, memory_nodes, nodes
from multi_agenti_conversation_system.supervisor.context import SupervisorContext
from multi_agenti_conversation_system.supervisor.state import AgentState


def build_graph(checkpointer):
    graph = StateGraph(AgentState, context_schema=SupervisorContext)
    graph.add_node("input_guard", guard_nodes.input_guard)
    graph.add_node("refuse", guard_nodes.refuse)
    graph.add_node("contextualize", memory_nodes.contextualize)
    graph.add_node("load_profile", memory_nodes.load_profile)
    graph.add_node("check_cache", memory_nodes.check_cache)
    graph.add_node("load_context", memory_nodes.load_context)
    graph.add_node("write_memory", memory_nodes.write_memory)
    graph.add_node("planner", nodes.planner)
    graph.add_node("supervisor", nodes.supervisor)
    graph.add_node("run_step", nodes.run_step)
    graph.add_node("hitl_gate", nodes.hitl_gate)
    graph.add_node("resume_dispatch", nodes.resume_dispatch)
    graph.add_node("finalize", nodes.finalize)
    graph.add_node("direct_answer", nodes.direct_answer)
    graph.add_node("summarize_history", nodes.summarize_history)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard", guard_nodes.route_after_guard, ["refuse", "contextualize"]
    )
    graph.add_edge("refuse", END)
    graph.add_edge("contextualize", "load_profile")
    graph.add_edge("load_profile", "check_cache")
    graph.add_conditional_edges(
        "check_cache", memory_nodes.route_after_cache, ["load_context", "write_memory"]
    )
    graph.add_edge("load_context", "planner")
    graph.add_edge("planner", "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        nodes.route_after_supervisor,
        ["run_step", "hitl_gate", "direct_answer", "finalize"],
    )
    graph.add_edge("run_step", "supervisor")
    graph.add_edge("hitl_gate", "resume_dispatch")
    graph.add_edge("resume_dispatch", "supervisor")
    graph.add_edge("finalize", "write_memory")
    graph.add_edge("direct_answer", "write_memory")
    graph.add_edge("write_memory", "summarize_history")
    graph.add_edge("summarize_history", END)
    return graph.compile(checkpointer=checkpointer)


def turn_input(
    text: str, user_id: str, domain: str | None = None, thread_id: str = ""
) -> dict[str, Any]:
    """Input for a new turn. `messages` and `summary` are NOT here: the checkpointer restores
    the conversation. Everything about the previous turn is reset, and the accumulating
    fields need `Overwrite` (a plain value would be merged into the previous turn's)."""
    return {
        "messages": [HumanMessage(content=text)],
        "question": text,
        "user_id": user_id,
        "thread_id": thread_id,
        "selected_domain": domain,
        "standalone_question": text,
        "entity_key": "",
        "standalone_ok": True,
        "guard_blocked": False,
        "guard_flagged": False,
        "guard_reasons": [],
        "input_pii": [],
        "memory_ok": True,
        "profile": {},
        "profile_hash": "",
        "facts": [],
        "episodes": [],
        "cache_hit": False,
        "memory_changes": [],
        "intent": "",
        "plan": [],
        "plan_error": "",
        "ready": [],
        "dispatched": [],
        "dispatch_count": 0,
        "skipped": [],
        "results": Overwrite({}),
        "timings": Overwrite({}),
        "hitl_decision": None,
        "approvals": Overwrite([]),
        "answer": "",
        "citations": [],
        "sql_used": None,
        "agents_used": [],
        "prompt_versions": {},
        "partial": False,
    }
