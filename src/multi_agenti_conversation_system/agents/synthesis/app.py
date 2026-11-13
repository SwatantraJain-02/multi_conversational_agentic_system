"""The Synthesis agent as an A2A service (:7003).

    uv run python -m multi_agenti_conversation_system.agents.synthesis.app

It combines what the SQL agent found (internal) with what the Web agent found (external)
into one grounded answer. If it isn't given external data it asks the Web agent itself, so
the Web agent (:7002) should be running; if it isn't, the answer degrades to internal data
only, and says so. Its own A2A client knows only the Web agent's address.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from a2a.types import AgentSkill
from langgraph.checkpoint.memory import InMemorySaver

from multi_agenti_conversation_system.a2a.client import AgentRegistry
from multi_agenti_conversation_system.a2a.contracts import SynthesisData
from multi_agenti_conversation_system.a2a.server import (
    GraphAgentExecutor,
    build_app,
    build_card,
    serve,
)
from multi_agenti_conversation_system.agents.llm_calls import text
from multi_agenti_conversation_system.agents.synthesis.context import SynthesisContext
from multi_agenti_conversation_system.agents.synthesis.graph import (
    AGENT,
    SKILL,
    build_graph,
    build_input,
    to_result,
)
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.llm import get_chat_model

card = build_card(
    name=AGENT,
    description=(
        "Combines internal findings (from the SQL agent) with external findings (from the "
        "Web agent) into one grounded answer, keeping the SQL used and the web citations. "
        "If external findings are missing it asks the Web agent for them itself."
    ),
    port=settings.SYNTHESIS_AGENT_PORT,
    skills=[
        AgentSkill(
            id=SKILL,
            name="Synthesise internal and external findings",
            description=(
                "Question plus `internal` and/or `external` results (see the data schema). "
                "Use it when a question needs both company data and outside information, "
                "e.g. comparing our sales with the market."
            ),
            tags=["synthesis", "comparison", "report"],
            examples=[
                "How do our beverage sales compare with the global market trend?"
            ],
        )
    ],
    skill_data={SKILL: SynthesisData},
)

executor = GraphAgentExecutor(None, AGENT, {SKILL}, build_input, to_result)


@asynccontextmanager
async def resources() -> AsyncIterator[None]:
    registry = AgentRegistry([f"http://{settings.A2A_HOST}:{settings.WEB_AGENT_PORT}"])
    try:
        await (
            registry.start()
        )  # a Web agent that isn't up yet is re-checked on the first call
        llm = get_chat_model()
        executor.graph = build_graph(InMemorySaver())
        executor.context = SynthesisContext(
            registry=registry, derive_question=text(llm), compose=text(llm)
        )
        yield
    finally:
        await registry.close()


app = build_app(card, executor, startup=resources)


def main() -> None:
    serve(app, settings.SYNTHESIS_AGENT_PORT)


if __name__ == "__main__":
    main()
