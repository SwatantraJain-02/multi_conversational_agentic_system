"""What the graph nodes get from the runtime (`Runtime[SqlContext]`), never from globals.

Everything that touches the outside world is here, so tests hand the graph fakes with the
same methods: a fake `db`, a fake `pairs`, and scripted `generate` / `summarise`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from multi_agenti_conversation_system.agents.sql.state import SQLGeneration

Messages = list[dict[str, str]]


@dataclass
class SqlContext:
    db: Any  # DbTools
    pairs: Any  # SqlPairStore
    generate: Callable[[Messages], Awaitable[SQLGeneration]]
    summarise: Callable[[Messages], Awaitable[str]]
    table_names: set[str]
    table_domains: dict[str, str]
    schema_version: str
    approval_rows: int
    max_rows: int  # the server-side cap; a request may ask for fewer, never more
