"""The SQL agent as an A2A service (:7001).

    uv run python -m multi_agenti_conversation_system.agents.sql.app

Needs the DB MCP server (:5555) running. At startup it opens, for the app's whole lifetime:
the MCP session to the DB server (tools are discovered, never hard-coded), a small Postgres
pool for the LangGraph checkpointer (`agent_memory`, thread_id = A2A contextId), and the
chat model. The A2A task store is in memory, so restarting the agent while a task waits
for approval loses that task (documented limitation).
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from a2a.types import AgentSkill

from multi_agenti_conversation_system.a2a.server import (
    GraphAgentExecutor,
    build_app,
    build_card,
    serve,
)
from multi_agenti_conversation_system.agents.llm_calls import structured, text
from multi_agenti_conversation_system.agents.sql.context import SqlContext
from multi_agenti_conversation_system.agents.sql.db_tools import DbTools
from multi_agenti_conversation_system.agents.sql.graph import (
    AGENT,
    SKILL,
    build_graph,
    build_input,
    to_result,
)
from multi_agenti_conversation_system.agents.sql.state import SQLGeneration
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.llm import get_chat_model
from multi_agenti_conversation_system.mcp.client import open_mcp
from multi_agenti_conversation_system.memory.checkpointer import open_checkpointer
from multi_agenti_conversation_system.memory.sql_pairs import SqlPairStore


def llm_calls(llm=None):
    """`generate` (structured output) and `summarise` over one chat model."""
    llm = llm or get_chat_model()
    return structured(llm, SQLGeneration), text(llm)


card = build_card(
    name=AGENT,
    description=(
        "Answers questions about the Northwind sales database by writing and running "
        "one read-only SQL query. Asks for human approval when a query touches personal "
        "data or would return a very large result."
    ),
    port=settings.SQL_AGENT_PORT,
    skills=[
        AgentSkill(
            id=SKILL,
            name="Answer a question from the database",
            description=(
                "Business questions answerable from Northwind's orders, customers, "
                "products, employees and shipping data: counts, revenue, rankings, "
                "trends. Returns the answer, the SQL used and the rows. Never writes."
            ),
            tags=["sql", "database", "northwind", "analytics"],
            examples=[
                "How many customers do we have in each country?",
                "Top 5 products by revenue in 2025",
            ],
        )
    ],
    skill_data={SKILL: None},
)

executor = GraphAgentExecutor(None, AGENT, {SKILL}, build_input, to_result)


@asynccontextmanager
async def resources() -> AsyncIterator[None]:
    async with open_mcp(["db"]) as tools:
        db = DbTools({t.name: t for t in tools["db"]})
        snapshot = await db.snapshot()
        generate, summarise = llm_calls()
        async with open_checkpointer() as saver:
            executor.graph = build_graph(saver)
            executor.context = SqlContext(
                db=db,
                pairs=SqlPairStore(),
                generate=generate,
                summarise=summarise,
                table_names=snapshot.table_names,
                table_domains=snapshot.table_domains,
                schema_version=snapshot.schema_version,
                approval_rows=settings.SQL_APPROVAL_ROWS,
                max_rows=settings.MAX_ROWS,
            )
            yield


app = build_app(card, executor, startup=resources)


def main() -> None:
    serve(app, settings.SQL_AGENT_PORT)


if __name__ == "__main__":
    main()
