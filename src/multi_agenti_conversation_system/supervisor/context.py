"""What the supervisor's nodes get from the runtime (`Runtime[SupervisorContext]`).

`registry` is the A2A client: the only way the supervisor reaches an agent. The three model
calls are plain async functions, so tests hand in scripted ones.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from multi_agenti_conversation_system.guardrails.input_guard import Classification
from multi_agenti_conversation_system.supervisor.state import (
    Contextual,
    Extraction,
    RouteDecision,
)

Messages = list[dict[str, str]]

# Per-skill deadlines. A delegated Synthesis run measured about 60 s; allow generously, since
# the A2A client has no read timeout and this deadline is the only bound on a call.
SKILL_TIMEOUTS_S = {
    "answer_sql_question": 120.0,
    "answer_web_question": 120.0,
    "synthesise_findings": 240.0,
}
DEFAULT_TIMEOUT_S = 120.0


@dataclass
class SupervisorContext:
    registry: Any  # a2a.client.AgentRegistry
    plan: Callable[[Messages], Awaitable[RouteDecision]]
    direct: Callable[[Messages], Awaitable[str]]
    summarise: Callable[[Messages], Awaitable[str]]
    # Input guard (Phase I): None means heuristics only.
    classify_input: Callable[[Messages], Awaitable[Classification]] | None = None
    # Memory (Phase H). None switches that part off: unit tests and the plain Phase G
    # supervisor build the context without them.
    contextualize: Callable[[Messages], Awaitable[Contextual]] | None = None
    extract: Callable[[Messages], Awaitable[Extraction]] | None = None
    profiles: Any = None  # memory.profile.ProfileStore
    facts: Any = None  # memory.facts.FactStore
    episodes: Any = None  # memory.episodes.EpisodeStore
    cache: Any = None  # memory.cache.SemanticCache
    skill_timeouts: dict[str, float] = field(
        default_factory=lambda: dict(SKILL_TIMEOUTS_S)
    )
    max_dispatches: int = 5  # agent calls per turn (a resume doesn't count)
    # History trimming: past this many messages or ~tokens, older ones become the summary.
    history_max_messages: int = 20
    history_max_tokens: int = 3000
    history_keep_last: int = 6
