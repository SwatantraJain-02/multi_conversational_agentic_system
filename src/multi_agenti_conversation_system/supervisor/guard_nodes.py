"""The input guard as a graph node (Phase I). It runs first: a blocked message never reaches
memory, the planner or an agent.

    START -> input_guard --(allowed)--> contextualize -> ...
                  |
                  (blocked) -> refuse -> END

A blocked exchange is removed from the conversation (it is kept in the TurnResult and the
trace): an injected message left in the history would be fed to the planner on later turns.
"""

import logging
from typing import Any

from langchain_core.messages import HumanMessage, RemoveMessage
from langgraph.runtime import Runtime

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.guardrails.input_guard import (
    InputVerdict,
    check_input,
)
from multi_agenti_conversation_system.prompts import aget_prompt
from multi_agenti_conversation_system.supervisor.context import SupervisorContext
from multi_agenti_conversation_system.supervisor.state import AgentState

logger = logging.getLogger(__name__)
Rt = Runtime[SupervisorContext]


async def input_guard(state: AgentState, runtime: Rt) -> dict[str, Any]:
    ctx = runtime.context
    prompt = None
    if ctx.classify_input is not None:
        try:
            prompt = await aget_prompt("guard/input")
        except Exception as exc:  # noqa: BLE001 - no prompt: heuristics only, flagged
            logger.warning("guard prompt unavailable: %s", exc)

    async def classify(messages):
        return await tracing.generation(
            ctx.classify_input,
            messages,
            "input-guard",
            prompt_link=prompt.link if prompt else None,
            model=settings.CHAT_MODEL,
        )

    verdict: InputVerdict = await check_input(
        state["question"], classify if ctx.classify_input else None, prompt
    )
    update: dict[str, Any] = {
        "question": verdict.text,
        "guard_blocked": verdict.action == "block",
        "guard_flagged": verdict.flagged,
        "guard_reasons": verdict.reasons,
        "input_pii": verdict.pii,
    }
    if verdict.action == "block":
        logger.warning("input blocked: %s", verdict.reasons)
        update["answer"] = verdict.refusal
        update["intent"] = "blocked"
    return update


def route_after_guard(state: AgentState) -> str:
    return "refuse" if state.get("guard_blocked") else "contextualize"


def refuse(state: AgentState) -> dict[str, Any]:
    """The refusal is the answer; the blocked message leaves the conversation."""
    last = next(
        (m for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        None,
    )
    return {
        "messages": [RemoveMessage(id=last.id)] if last is not None and last.id else [],
        "citations": [],
        "sql_used": None,
        "agents_used": [],
        "partial": False,
    }
