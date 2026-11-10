"""The Web agent as an A2A service (:7002).

    uv run python -m multi_agenti_conversation_system.agents.web.app

Needs the Web MCP server (:5556) running. At startup it opens, for the app's whole
lifetime: the MCP session to the Web server, the local re-ranking model (flashrank, a
3 MB cross-encoder downloaded on first use into `.cache/flashrank`), and the chat model.
The graph's checkpointer is in memory: this agent never pauses for a human, so there is
nothing to resume after a restart, and it saves one of Aiven's few connections (G16).
"""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from a2a.types import AgentSkill
from flashrank import Ranker, RerankRequest
from langgraph.checkpoint.memory import InMemorySaver

from multi_agenti_conversation_system.a2a.server import (
    GraphAgentExecutor,
    build_app,
    build_card,
    serve,
)
from multi_agenti_conversation_system.agents.llm_calls import structured, text
from multi_agenti_conversation_system.agents.web.context import WebContext
from multi_agenti_conversation_system.agents.web.graph import (
    AGENT,
    SKILL,
    build_graph,
    build_input,
    to_result,
)
from multi_agenti_conversation_system.agents.web.state import SearchPlan
from multi_agenti_conversation_system.agents.web.web_tools import WebTools
from multi_agenti_conversation_system.config import BASE_DIR, settings
from multi_agenti_conversation_system.llm import get_chat_model
from multi_agenti_conversation_system.mcp.client import open_mcp

RANKER_CACHE = BASE_DIR / ".cache" / "flashrank"


async def make_reranker():
    """A re-rank function over the flashrank model, run off the event loop (CPU-bound)."""
    RANKER_CACHE.mkdir(parents=True, exist_ok=True)
    ranker = await asyncio.to_thread(
        Ranker, cache_dir=str(RANKER_CACHE), log_level="WARNING"
    )

    async def rerank(
        query: str, passages: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        ranked = await asyncio.to_thread(
            ranker.rerank, RerankRequest(query=query, passages=passages)
        )
        return [{"id": r["id"], "score": float(r["score"])} for r in ranked]

    return rerank


card = build_card(
    name=AGENT,
    description=(
        "Answers questions that need current or external information by searching the "
        "web, reading the best pages and writing a short answer with numbered citations "
        "to the pages it actually read."
    ),
    port=settings.WEB_AGENT_PORT,
    skills=[
        AgentSkill(
            id=SKILL,
            name="Answer a question from the web",
            description=(
                "Market trends, news, public facts and anything outside the company "
                "database. Returns the answer, its citations (URLs) and a confidence "
                "level (high / low / none)."
            ),
            tags=["web", "search", "news", "research"],
            examples=[
                "What are the main trends in the global beverage market in 2026?",
                "What is the latest news about EU food labelling rules?",
            ],
        )
    ],
    skill_data={SKILL: None},
)

executor = GraphAgentExecutor(None, AGENT, {SKILL}, build_input, to_result)


@asynccontextmanager
async def resources() -> AsyncIterator[None]:
    async with open_mcp(["web"]) as tools:
        llm = get_chat_model()
        executor.graph = build_graph(InMemorySaver())
        executor.context = WebContext(
            web=WebTools({t.name: t for t in tools["web"]}),
            plan=structured(llm, SearchPlan),
            summarise=text(llm),
            rerank=await make_reranker(),
        )
        yield


app = build_app(card, executor, startup=resources)


def main() -> None:
    serve(app, settings.WEB_AGENT_PORT)


if __name__ == "__main__":
    main()
