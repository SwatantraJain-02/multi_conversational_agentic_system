"""Wiring of the Web agent graph, and its two translations to and from A2A."""

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite, RetryPolicy

from multi_agenti_conversation_system.a2a.contracts import (
    AgentError,
    TaskRequest,
    TaskResult,
)
from multi_agenti_conversation_system.agents.web import nodes
from multi_agenti_conversation_system.agents.web.context import WebContext
from multi_agenti_conversation_system.agents.web.state import WebState

AGENT = "web"
SKILL = "answer_web_question"


def _transient(exc: Exception) -> bool:
    return isinstance(exc, AgentError) and exc.retryable


# Transport-level failures only, once: the chat client already retries by itself.
RETRY = RetryPolicy(max_attempts=2, initial_interval=0.5, retry_on=_transient)


def build_graph(checkpointer):
    graph = StateGraph(WebState, context_schema=WebContext)
    graph.add_node("plan_queries", nodes.plan_queries)
    graph.add_node("search_one", nodes.search_one, retry_policy=RETRY)
    graph.add_node("merge_dedupe", nodes.merge_dedupe)
    graph.add_node("fetch_one", nodes.fetch_one, retry_policy=RETRY)
    graph.add_node("chunk", nodes.chunk)
    graph.add_node("rerank", nodes.rerank)
    graph.add_node("compress", nodes.compress)
    graph.add_node("summarise", nodes.summarise, retry_policy=RETRY)
    graph.add_node("no_results", nodes.no_results)

    graph.add_edge(START, "plan_queries")
    graph.add_conditional_edges("plan_queries", nodes.fan_out_search, ["search_one"])
    graph.add_edge("search_one", "merge_dedupe")
    graph.add_conditional_edges(
        "merge_dedupe", nodes.route_after_merge, ["fetch_one", "no_results"]
    )
    graph.add_edge("fetch_one", "chunk")
    graph.add_conditional_edges(
        "chunk", nodes.route_after_chunk, ["rerank", "no_results"]
    )
    graph.add_edge("rerank", "compress")
    graph.add_edge("compress", "summarise")
    graph.add_edge("summarise", END)
    graph.add_edge("no_results", END)
    return graph.compile(checkpointer=checkpointer)


def build_input(request: TaskRequest) -> dict[str, Any]:
    """Graph input for a new task. Resets EVERY per-task field. The accumulating lists
    need `Overwrite`: a plain `[]` would be *added* to what an earlier task in the same
    A2A context left in the checkpoint."""
    return {
        "question": request.task,
        "queries": [],
        "recency": "any",
        "topic": "general",
        "hits": Overwrite([]),
        "search_skips": Overwrite([]),
        "search_errors": Overwrite([]),
        "urls": [],
        "pages": Overwrite([]),
        "fetch_skips": Overwrite([]),
        "fetch_failures": Overwrite([]),
        "chunks": [],
        "chunks_before": 0,
        "chunks_after": 0,
        "snippet_mode": False,
        "sources": [],
        "context_chars": 0,
        "answer": "",
        "citations": [],
        "confidence": "none",
        "prompt_tags": {},
    }


def to_result(state: dict[str, Any], request: TaskRequest) -> TaskResult:
    return TaskResult(
        status="completed",
        agent=AGENT,
        answer=state["answer"],
        citations=state["citations"],
        data={
            "confidence": state["confidence"],
            "queries": state["queries"],
            "sources_consulted": [
                {k: s[k] for k in ("n", "url", "title", "published_date")}
                for s in state["sources"]
            ],
            "pages_read": len(state["pages"]),
            "fetch_failures": state["fetch_failures"],
            "rate_limited": {
                "searches": len(state["search_skips"]),
                "fetches": len(state["fetch_skips"]),
            },
            "chunks_before": state["chunks_before"],
            "chunks_after": state["chunks_after"],
            "context_chars": state["context_chars"],
            "snippet_only": state["snippet_mode"],
        },
        prompt_versions=state["prompt_tags"],
    )
