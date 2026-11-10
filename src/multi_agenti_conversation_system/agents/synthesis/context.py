"""What the Synthesis nodes get from the runtime (`Runtime[SynthesisContext]`).

`registry` is the A2A client: the only way this agent reaches another agent (agents never
import each other). Tests hand in a fake with the same `call` method.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

Messages = list[dict[str, str]]


@dataclass
class SynthesisContext:
    registry: Any  # a2a.client.AgentRegistry
    derive_question: Callable[[Messages], Awaitable[str]]
    compose: Callable[[Messages], Awaitable[str]]
    web_agent: str = "web"
    web_skill: str = "answer_web_question"
    delegate_timeout_s: float = 90.0
