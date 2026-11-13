"""The Web agent's graph nodes and routing.

    plan_queries -> search_one (xN, parallel) -> merge_dedupe -> fetch_one (xM, parallel)
        -> chunk -> rerank -> compress -> summarise            (no sources: no_results)

Budget (the Web MCP server rate-limits per process: search bursts at 5, fetch at 10):
at most MAX_QUERIES searches and MAX_URLS page reads per task.

A rate-limited reply is *data*, not an error and never reaches the model: a short wait is
tolerated once, otherwise that search or page is skipped and counted. If every search is
rate limited or fails, the task fails as `unavailable`; if every page read fails, the agent
answers from the search snippets and flags low confidence.

Fetched text is untrusted: it only ever enters the prompt inside the numbered sources
block, and the prompt says to treat it as data.

Nodes get everything from `runtime.context` (a `WebContext`).
"""

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, TypedDict

from langgraph.runtime import Runtime
from langgraph.types import Send

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import AgentError
from multi_agenti_conversation_system.agents.web.context import WebContext
from multi_agenti_conversation_system.agents.web.state import WebState
from multi_agenti_conversation_system.agents.web.text import (
    MIN_SNIPPET_CHARS,
    build_context,
    chunk_text,
    normalize_url,
    strip_invalid_citations,
)
from multi_agenti_conversation_system.agents.web.web_tools import (
    is_rate_limited,
    retry_after,
)
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.prompts import aget_prompt
from multi_agenti_conversation_system.retrieval.hybrid import rrf

logger = logging.getLogger(__name__)

MAX_QUERIES = 3
RESULTS_PER_QUERY = 5
MAX_URLS = 5
FETCH_CHARS = 20_000
TOP_CHUNKS = 6
CONTEXT_CHARS = 12_000  # about 3k tokens
RATE_LIMIT_PATIENCE_S = 3.0  # wait this long once for a refill; longer means skip
SNIPPET_NOTE = (
    "Note: the pages could not be read, so this is based on search snippets only.\n\n"
)
NO_RESULTS = "I couldn't find any relevant web sources for this question."

Rt = Runtime[WebContext]


class SearchTask(TypedDict):
    query: str
    recency: str
    topic: str


class FetchTask(TypedDict):
    url: str
    title: str
    snippet: str
    published_date: str | None


async def _patiently(call) -> dict[str, Any]:
    """Run a tool call; if rate limited with a short wait, wait once and retry."""
    reply = await call()
    if is_rate_limited(reply) and retry_after(reply) <= RATE_LIMIT_PATIENCE_S:
        await asyncio.sleep(retry_after(reply) + 0.1)
        reply = await call()
    return reply


def _clean_queries(queries: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned = []
    for query in queries:
        query = " ".join(query.split())[:400]
        if query and query.lower() not in seen:
            seen.add(query.lower())
            cleaned.append(query)
    return cleaned[:MAX_QUERIES]


# ------------------------------------------------------------------------------- nodes
async def plan_queries(state: WebState, runtime: Rt) -> dict[str, Any]:
    prompt = await aget_prompt("web/plan")
    messages = prompt.compile(
        question=state["question"], today=datetime.now(UTC).date().isoformat()
    )
    update: dict[str, Any] = {
        "recency": "any",
        "topic": "general",
        "prompt_tags": {**state.get("prompt_tags", {}), prompt.name: prompt.tag},
    }
    try:
        plan = await tracing.generation(
            runtime.context.plan,
            messages,
            "plan-queries",
            prompt_link=prompt.link,
            model=settings.CHAT_MODEL,
        )
        update.update(
            queries=_clean_queries(plan.queries), recency=plan.recency, topic=plan.topic
        )
    except Exception as exc:  # noqa: BLE001 - a plan is a bonus: the question itself is a query
        logger.warning("query planning failed (%s), searching the question as-is", exc)
        update["queries"] = []
    if not update.get("queries"):
        update["queries"] = _clean_queries([state["question"]])
    return update


def fan_out_search(state: WebState) -> list[Send]:
    return [
        Send(
            "search_one",
            SearchTask(query=q, recency=state["recency"], topic=state["topic"]),
        )
        for q in state["queries"]
    ]


async def search_one(task: SearchTask, runtime: Rt) -> dict[str, Any]:
    reply = await _patiently(
        lambda: runtime.context.web.search_web(
            task["query"], RESULTS_PER_QUERY, task["recency"], task["topic"]
        )
    )
    if is_rate_limited(reply):
        return {"search_skips": [task["query"]]}
    if not reply.get("ok"):
        return {"search_errors": [str(reply.get("error", "search failed"))]}
    return {
        "hits": [
            {"query": task["query"], "rank": rank, **result}
            for rank, result in enumerate(reply.get("results", []), start=1)
        ]
    }


def merge_dedupe(state: WebState) -> dict[str, Any]:
    """One list of pages: the same page found by several queries counts once, and ranks
    higher for it (reciprocal rank fusion over the per-query rankings)."""
    hits = state.get("hits", [])
    if not hits:
        skipped, errors = state.get("search_skips", []), state.get("search_errors", [])
        if skipped and len(skipped) == len(state["queries"]):
            raise AgentError(
                "unavailable",
                "web search is rate limited, try again shortly",
                retryable=True,
            )
        if errors:
            raise AgentError(
                "unavailable", f"web search failed: {errors[0]}", retryable=True
            )
        return {"urls": []}  # the searches worked and found nothing

    rankings: dict[str, list[str]] = {}
    first_seen: dict[str, dict[str, Any]] = {}
    for hit in sorted(hits, key=lambda h: (h["query"], h["rank"])):
        key = normalize_url(hit["url"])
        ranking = rankings.setdefault(hit["query"], [])
        if key not in ranking:
            ranking.append(key)
        first_seen.setdefault(key, hit)
    fused = rrf(list(rankings.values()))
    return {
        "urls": [
            {
                "url": first_seen[key]["url"],
                "title": first_seen[key].get("title") or first_seen[key]["url"],
                "snippet": first_seen[key].get("snippet") or "",
                "published_date": first_seen[key].get("published_date"),
            }
            for key, _ in fused[:MAX_URLS]
        ]
    }


def route_after_merge(state: WebState) -> list[Send] | str:
    if not state.get("urls"):
        return "no_results"
    return [
        Send(
            "fetch_one",
            FetchTask(
                url=u["url"],
                title=u["title"],
                snippet=u["snippet"],
                published_date=u["published_date"],
            ),
        )
        for u in state["urls"]
    ]


async def fetch_one(task: FetchTask, runtime: Rt) -> dict[str, Any]:
    reply = await _patiently(
        lambda: runtime.context.web.fetch_page(task["url"], FETCH_CHARS)
    )
    if is_rate_limited(reply):
        return {"fetch_skips": [task["url"]]}
    if not reply.get("ok"):
        return {
            "fetch_failures": [
                {
                    "url": task["url"],
                    "error": str(reply.get("error", "fetch failed"))[:200],
                }
            ]
        }
    return {
        "pages": [
            {
                "url": reply.get("final_url") or task["url"],
                "title": reply.get("title") or task["title"],
                "published_date": reply.get("published_date") or task["published_date"],
                "text": reply["text"],
            }
        ]
    }


def chunk(state: WebState) -> dict[str, Any]:
    chunks = [
        {
            "url": page["url"],
            "title": page["title"],
            "published_date": page["published_date"],
            "text": text,
            "snippet_only": False,
        }
        for page in state.get("pages", [])
        for text in chunk_text(page["text"])
    ]
    snippet_mode = False
    if not chunks:  # no page could be read: the search snippets are all there is
        chunks = [
            {
                "url": u["url"],
                "title": u["title"],
                "published_date": u["published_date"],
                "text": u["snippet"].strip(),
                "snippet_only": True,
            }
            for u in state.get("urls", [])
            if len(u["snippet"].strip()) >= MIN_SNIPPET_CHARS
        ]
        snippet_mode = bool(chunks)
    return {
        "chunks": chunks,
        "chunks_before": len(chunks),
        "snippet_mode": snippet_mode,
    }


def route_after_chunk(state: WebState) -> str:
    return "rerank" if state.get("chunks") else "no_results"


async def rerank(state: WebState, runtime: Rt) -> dict[str, Any]:
    """Keep the TOP_CHUNKS chunks that best match the question (a local cross-encoder)."""
    chunks = state["chunks"]
    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(chunks)]
    try:
        ranked = await runtime.context.rerank(state["question"], passages)
        order = [r["id"] for r in ranked if 0 <= r["id"] < len(chunks)]
    except Exception as exc:  # noqa: BLE001 - without re-ranking, page order is still usable
        logger.warning("re-ranking failed (%s), keeping page order", exc)
        order = list(range(len(chunks)))
    top = [chunks[i] for i in order[:TOP_CHUNKS]]
    logger.info("web agent re-rank: %d chunks -> %d kept", len(chunks), len(top))
    return {"chunks": top, "chunks_after": len(top)}


def compress(state: WebState) -> dict[str, Any]:
    """Fit the kept chunks into the context budget and number the sources."""
    sources, used = build_context(state["chunks"], CONTEXT_CHARS)
    return {
        "sources": [
            {**s.as_dict(), "snippet_only": s.snippet_only, "excerpts": s.excerpts}
            for s in sources
        ],
        "context_chars": used,
    }


def _sources_block(sources: list[dict[str, Any]]) -> str:
    blocks = []
    for s in sources:
        when = f", {s['published_date']}" if s.get("published_date") else ""
        blocks.append(
            f"[{s['n']}] {s['title']} ({s['url']}{when})\n"
            + "\n---\n".join(s["excerpts"])
        )
    return "\n\n".join(blocks)


async def summarise(state: WebState, runtime: Rt) -> dict[str, Any]:
    prompt = await aget_prompt("web/summarise")
    messages = prompt.compile(
        question=state["question"], sources=_sources_block(state["sources"])
    )
    raw = await tracing.generation(
        runtime.context.summarise,
        messages,
        "summarise",
        prompt_link=prompt.link,
        model=settings.CHAT_MODEL,
    )
    # Citations are checked in code: the model may cite [7] when there are three sources.
    answer, cited = strip_invalid_citations(
        str(raw), {s["n"] for s in state["sources"]}
    )
    snippet_mode = state.get("snippet_mode", False)
    return {
        "answer": (SNIPPET_NOTE + answer) if snippet_mode else answer,
        "citations": [
            {k: s[k] for k in ("n", "url", "title", "published_date")}
            for s in state["sources"]
            if s["n"] in cited
        ],
        "confidence": "low" if snippet_mode or not cited else "high",
        "prompt_tags": {**state.get("prompt_tags", {}), prompt.name: prompt.tag},
    }


def no_results(state: WebState) -> dict[str, Any]:
    return {"answer": NO_RESULTS, "citations": [], "sources": [], "confidence": "none"}
