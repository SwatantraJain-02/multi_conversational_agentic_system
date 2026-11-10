"""The LangGraph checkpointer on Aiven (`agent_memory`), shared by the SQL agent and the supervisor.

    async with open_checkpointer() as saver:
        graph = builder.compile(checkpointer=saver)

A small pool (G16: Aiven's connection cap is shared by every service), checked when a
connection is taken from it because Aiven drops idle connections. `setup()` creates the
checkpoint tables on first use and is a no-op afterwards.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from multi_agenti_conversation_system.config import settings

POOL_MAX_SIZE = 2


@asynccontextmanager
async def open_memory() -> AsyncIterator[tuple[AsyncPostgresSaver, AsyncPostgresStore]]:
    """The checkpointer AND the profile store on ONE pool: the store costs no extra connections."""
    async with _pool() as pool:
        saver, store = AsyncPostgresSaver(pool), AsyncPostgresStore(pool)
        await saver.setup()
        await store.setup()
        yield saver, store


@asynccontextmanager
async def open_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    async with _pool() as pool:
        saver = AsyncPostgresSaver(pool)
        await saver.setup()
        yield saver


def _pool() -> AsyncConnectionPool:
    return AsyncConnectionPool(
        settings.memory_conninfo,
        min_size=1,
        max_size=POOL_MAX_SIZE,
        open=False,
        check=AsyncConnectionPool.check_connection,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
    )
