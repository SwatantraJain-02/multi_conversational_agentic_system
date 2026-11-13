"""What the Web graph nodes get from the runtime (`Runtime[WebContext]`), never from globals.

Everything that touches the outside world is here, so tests hand the graph fakes with the
same methods: a fake `web`, scripted `plan` / `summarise`, and a fake `rerank`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from multi_agenti_conversation_system.agents.web.state import SearchPlan

Messages = list[dict[str, str]]
# (query, passages [{"id", "text"}]) -> [{"id", "score"}], best first
Reranker = Callable[[str, list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]


@dataclass
class WebContext:
    web: Any  # WebTools
    plan: Callable[[Messages], Awaitable[SearchPlan]]
    summarise: Callable[[Messages], Awaitable[str]]
    rerank: Reranker
