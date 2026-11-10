"""The supervisor as a service object: what the Streamlit UI (Phase J) and the scripts call.

    async with open_supervisor() as supervisor:
        turn = await supervisor.ask("thread-1", "alice", "How many customers per country?")
        if turn.status == "needs_approval":          # a human has to approve a SQL query
            turn = await supervisor.resume("thread-1", approved=True)
        print(turn.answer, turn.citations, turn.plan)

It runs inside the caller's event loop (no `asyncio.run` in here), so a UI can keep one
background loop for the whole process. The conversation is checkpointed per `thread_id` in
Postgres: a new `Supervisor` with the same thread id continues where the last one stopped.
Turns of one thread are serialised (an answer is still being written to the checkpoint when
a fast second request would arrive, G13).

`on_event` receives {"type": "node" | "plan", ...} as the graph progresses, for a progress UI.
"""

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal

from langgraph.types import Command

from multi_agenti_conversation_system import tracing, vectorstore
from multi_agenti_conversation_system.a2a.client import AgentRegistry
from multi_agenti_conversation_system.agents.llm_calls import structured, text
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.guardrails.input_guard import Classification
from multi_agenti_conversation_system.llm import get_chat_model
from multi_agenti_conversation_system.memory import sql_pairs
from multi_agenti_conversation_system.memory.cache import SemanticCache
from multi_agenti_conversation_system.memory.checkpointer import open_memory
from multi_agenti_conversation_system.memory.episodes import EpisodeStore
from multi_agenti_conversation_system.memory.facts import FactStore
from multi_agenti_conversation_system.memory.profile import ProfileStore
from multi_agenti_conversation_system.supervisor.context import SupervisorContext
from multi_agenti_conversation_system.supervisor.graph import build_graph, turn_input
from multi_agenti_conversation_system.supervisor.memory_nodes import write_episode
from multi_agenti_conversation_system.supervisor.state import (
    Contextual,
    Extraction,
    RouteDecision,
)

logger = logging.getLogger(__name__)

EventCallback = Callable[[dict[str, Any]], None]


@dataclass
class TurnResult:
    thread_id: str
    status: Literal["answered", "needs_approval"]
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    sql_used: str | None = None
    agents_used: list[str] = field(default_factory=list)
    intent: str = ""
    plan: list[dict[str, Any]] = field(default_factory=list)
    partial: bool = False
    prompt_versions: dict[str, str] = field(default_factory=dict)
    approval: dict[str, Any] | None = None  # the interrupt payload when needs_approval
    results: dict[str, Any] = field(default_factory=dict)  # step id -> TaskResult dict
    timings: dict[str, list[float]] = field(
        default_factory=dict
    )  # step id -> [start, end]
    cache_hit: bool = False
    standalone_question: str = ""  # what the cache and the learned pairs were keyed on
    blocked: bool = False  # the input guard refused the message
    guard_reasons: list[str] = field(default_factory=list)
    memory_changes: list[str] = field(default_factory=list)  # what this turn learned
    trace_id: str | None = None


class Supervisor:
    def __init__(self, graph, context: SupervisorContext) -> None:
        self.graph, self.context = graph, context
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    async def ask(
        self,
        thread_id: str,
        user_id: str,
        text_: str,
        *,
        domain: str | None = None,
        on_event: EventCallback | None = None,
        trace_ctx: dict[str, str] | None = None,
    ) -> TurnResult:
        """`trace_ctx` (from `tracing.current_trace_ctx()`) nests the turn under a trace
        started elsewhere, e.g. an eval experiment item running in another thread."""
        return await self._turn(
            thread_id,
            user_id,
            turn_input(text_, user_id, domain, thread_id),
            text_,
            on_event,
            trace_ctx,
        )

    async def resume(
        self,
        thread_id: str,
        approved: bool,
        *,
        user_id: str = "",
        on_event: EventCallback | None = None,
        trace_ctx: dict[str, str] | None = None,
    ) -> TurnResult:
        """Answer the approval a turn is waiting for (`TurnResult.approval`)."""
        return await self._turn(
            thread_id,
            user_id,
            Command(resume={"approved": approved}),
            "approval",
            on_event,
            trace_ctx,
        )

    async def history(self, thread_id: str) -> dict[str, Any]:
        """What the checkpoint holds for the thread: the messages and the running summary."""
        values = (await self.graph.aget_state(self._config(thread_id))).values
        return {
            "messages": [
                (type(m).__name__, m.content) for m in values.get("messages", [])
            ],
            "summary": values.get("summary", ""),
        }

    async def end_thread(self, thread_id: str, user_id: str) -> None:
        """The conversation is over (the UI's "New chat"): write its summary as an episode,
        so a later thread of the same user can recall it. Never raises."""
        ctx = self.context
        if ctx.episodes is None or settings.EVAL_MODE:
            return
        try:
            values = (await self.graph.aget_state(self._config(thread_id))).values
            await write_episode(
                ctx,
                thread_id,
                user_id,
                values.get("summary", ""),
                values.get("messages", []),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("end_thread failed: %s: %s", type(exc).__name__, exc)

    async def what_i_know(self, user_id: str) -> dict[str, Any]:
        """For "what do you know about me?" and the UI sidebar: the profile (with the history
        of each value) and the active facts, dated."""
        ctx = self.context
        profile = await ctx.profiles.load(user_id) if ctx.profiles else {}
        facts = await ctx.facts.list_active(user_id) if ctx.facts else []
        return {
            "profile": {
                key: {"value": e["value"], "history": e.get("history", [])}
                for key, e in profile.items()
            },
            "facts": [
                {"attribute": f.attribute, "text": f.text, "dated": f.dated()}
                for f in facts
            ],
        }

    async def forget_fact(self, user_id: str, attribute: str) -> None:
        """ "Forget X": hard-delete the fact and its history."""
        if self.context.facts is not None and attribute:
            await self.context.facts.forget(user_id, attribute)

    async def reset_profile(self, user_id: str) -> None:
        if self.context.profiles is not None:
            await self.context.profiles.reset(user_id)

    async def feedback(self, turn: dict[str, Any], positive: bool) -> None:
        """A thumbs up or down on an answer. Never raises.

        Both are recorded as a Langfuse score on the turn's trace. A thumbs-down also makes
        the answer unusable again: the cached answer for that question is deleted, and the SQL
        pair learned from it (shared by all users) is forgotten, so a wrong answer cannot be
        served, or taught, a second time.
        """
        try:
            await asyncio.to_thread(self._score, turn.get("trace_id"), positive)
            if positive:
                return
            question = turn.get("standalone_question") or ""
            if question and self.context.cache is not None:
                await asyncio.to_thread(
                    vectorstore.delete_where,
                    vectorstore.SEMANTIC_CACHE,
                    {"text": {"$eq": question}},
                )
            if turn.get("sql_used") and "sql" in turn.get("agents_used", []):
                await asyncio.to_thread(sql_pairs.delete, turn["sql_used"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("feedback failed: %s: %s", type(exc).__name__, exc)

    @staticmethod
    def _score(trace_id: str | None, positive: bool) -> None:
        client = tracing.get_langfuse()
        if client is not None and trace_id:
            client.create_score(
                trace_id=trace_id,
                name="user-feedback",
                value=1.0 if positive else 0.0,
                data_type="NUMERIC",
            )
            client.flush()

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    async def _turn(
        self,
        thread_id: str,
        user_id: str,
        graph_input: Any,
        label: str,
        on_event: EventCallback | None,
        trace_ctx: dict[str, str] | None = None,
    ) -> TurnResult:
        config = self._config(thread_id)
        async with self._locks[thread_id]:
            with (
                tracing.session(thread_id, user_id or None),
                tracing.observation(
                    "turn", as_type="chain", input=label, trace_ctx=trace_ctx
                ),
            ):
                trace_id = tracing.current_trace_ctx().get("trace_id")
                approval = None
                async for chunk in self.graph.astream(
                    graph_input, config, context=self.context, stream_mode="updates"
                ):
                    for node, update in chunk.items():
                        if node == "__interrupt__":
                            approval = update[0].value
                        elif on_event:
                            on_event({"type": "node", "node": node})
                            if node == "planner" and update.get("plan") is not None:
                                on_event(
                                    {
                                        "type": "plan",
                                        "intent": update.get("intent"),
                                        "plan": update["plan"],
                                        "error": update.get("plan_error"),
                                    }
                                )
                values = (await self.graph.aget_state(config)).values
        return self._result(thread_id, values, approval, trace_id)

    @staticmethod
    def _result(
        thread_id: str,
        values: dict[str, Any],
        approval: dict[str, Any] | None,
        trace_id: str | None,
    ) -> TurnResult:
        common = {
            "thread_id": thread_id,
            "intent": values.get("intent", ""),
            "plan": values.get("plan", []),
            "results": values.get("results", {}),
            "timings": values.get("timings", {}),
            "trace_id": trace_id,
        }
        if approval is not None:
            return TurnResult(status="needs_approval", approval=approval, **common)
        return TurnResult(
            status="answered",
            answer=values.get("answer", ""),
            citations=values.get("citations", []),
            sql_used=values.get("sql_used"),
            agents_used=values.get("agents_used", []),
            partial=values.get("partial", False),
            cache_hit=values.get("cache_hit", False),
            standalone_question=values.get("standalone_question", ""),
            blocked=values.get("guard_blocked", False),
            guard_reasons=values.get("guard_reasons", []),
            memory_changes=values.get("memory_changes", []),
            prompt_versions=values.get("prompt_versions", {}),
            **common,
        )


@asynccontextmanager
async def open_supervisor(
    agent_urls: list[str] | None = None, **limits: Any
) -> AsyncIterator[Supervisor]:
    """A supervisor wired to the real agents (A2A), model and Aiven checkpointer.

    `agent_urls` defaults to `AGENT_URLS`; `limits` override `SupervisorContext` fields
    (e.g. `history_max_messages`). An agent that is down at startup is re-checked on use.
    """
    registry = AgentRegistry(agent_urls)
    try:
        await registry.start()
        llm = get_chat_model()
        context = SupervisorContext(
            registry=registry,
            plan=structured(llm, RouteDecision),
            direct=text(llm),
            summarise=text(llm),
            classify_input=structured(llm, Classification),
            contextualize=structured(llm, Contextual),
            extract=structured(llm, Extraction),
            facts=FactStore(),
            episodes=EpisodeStore(),
            cache=SemanticCache(),
            **limits,
        )
        async with open_memory() as (saver, store):
            context.profiles = ProfileStore(store)
            yield Supervisor(build_graph(saver), context)
    finally:
        await registry.close()
