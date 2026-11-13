"""A2A client: discovers agents from their cards and calls them, never raising.

    registry = AgentRegistry()
    await registry.start()                              # fetches every card in AGENT_URLS
    result = await registry.call("echo", TaskRequest(sender="supervisor", skill="echo", task="hi"))
    if result.status == "input_required":               # e.g. HITL
        result = await registry.resume(result, {"approved": True})

Every outcome is a `TaskResult`: failures come back as `status="failed"` with an
`error.code` (timeout, unavailable, invalid_output, bad_request, failed, internal), so the
supervisor can re-plan or answer partially instead of crashing.

- Discovery: a card that can't be fetched marks that agent down (it doesn't raise at
  startup). A call to a down agent re-checks its card once, so a restarted agent comes back,
  and so does one that was not running yet when the registry started.
- Timeout: the whole exchange runs under one `asyncio.wait_for`; on timeout the task is
  cancelled best-effort so the agent stops working.
- Invalid output: a fresh call is retried once. A resume is never re-sent.
- Tracing: unless the caller passes an explicit `trace_ctx`, each call is wrapped in an
  `a2a.call:<agent>` span and that span's context goes into the A2A message metadata, so
  the agent's spans nest under it (G20). An explicit `trace_ctx` is sent as given.
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.client.errors import A2AClientError, A2AClientTimeoutError
from a2a.types import (
    AgentCard,
    CancelTaskRequest,
    Message,
    Role,
    SendMessageRequest,
    StreamResponse,
    TaskState,
)
from a2a.utils.errors import A2AError

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import (
    ContractError,
    ResumeRequest,
    TaskRequest,
    TaskResult,
    from_parts,
    to_part,
)
from multi_agenti_conversation_system.config import settings

logger = logging.getLogger(__name__)

CARD_TIMEOUT_S = 3.0
Progress = Callable[
    [str, str], None
]  # (agent name, text such as "node: retrieve_schema")

_TERMINAL = {
    TaskState.TASK_STATE_COMPLETED,
    TaskState.TASK_STATE_FAILED,
    TaskState.TASK_STATE_REJECTED,
    TaskState.TASK_STATE_CANCELED,
    TaskState.TASK_STATE_INPUT_REQUIRED,
}


@dataclass
class AgentInfo:
    url: str
    name: str | None = None
    card: AgentCard | None = None
    up: bool = False
    error: str | None = None
    client: Any = None  # a2a Client, created from the card

    @property
    def skills(self) -> set[str]:
        return {s.id for s in self.card.skills} if self.card else set()


@dataclass
class _Exchange:
    """What one send_message stream delivered."""

    task_id: str = ""
    context_id: str = ""
    state: int = TaskState.TASK_STATE_UNSPECIFIED
    parts: list = field(
        default_factory=list
    )  # artifact parts, then status/message parts
    status_parts: list = field(default_factory=list)

    def add(
        self, event: StreamResponse, agent: str, on_progress: Progress | None
    ) -> None:
        if event.HasField("task"):
            task = event.task
            self.task_id, self.context_id = task.id, task.context_id
            self.state = task.status.state
            for artifact in task.artifacts:
                self.parts.extend(artifact.parts)
            self.status_parts = list(task.status.message.parts)
        elif event.HasField("message"):
            self.status_parts = list(event.message.parts)
            self.state = TaskState.TASK_STATE_COMPLETED
        elif event.HasField("status_update"):
            update = event.status_update
            self.task_id, self.context_id = update.task_id, update.context_id
            self.state = update.status.state
            parts = list(update.status.message.parts)
            if self.state == TaskState.TASK_STATE_WORKING:
                if on_progress:
                    for part in parts:
                        if part.HasField("text"):
                            on_progress(agent, part.text)
            else:
                self.status_parts = parts
        elif event.HasField("artifact_update"):
            self.parts.extend(event.artifact_update.artifact.parts)


def _result_from(exchange: _Exchange, agent: str) -> TaskResult:
    """The TaskResult an exchange ended with, or a failure saying why there is none."""
    if exchange.state not in _TERMINAL:
        return TaskResult.failure(
            agent,
            "invalid_output",
            f"the stream ended in state {TaskState.Name(exchange.state)}",
            retryable=True,
        )
    try:
        # A result artifact wins; failed / input-required results ride in the status message.
        result = from_parts([*exchange.parts, *exchange.status_parts], TaskResult)
    except ContractError as exc:
        if exchange.state in (
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_REJECTED,
        ):
            return TaskResult.failure(agent, "failed", "the agent reported a failure")
        return TaskResult.failure(agent, "invalid_output", str(exc), retryable=True)
    result.a2a_task_id, result.context_id = exchange.task_id, exchange.context_id
    return result


class AgentRegistry:
    def __init__(
        self,
        urls: list[str] | None = None,
        token: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        token = settings.A2A_TOKEN if token is None else token
        self._timeout_s = timeout_s or settings.A2A_TIMEOUT_S
        self._http = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {token}"} if token else {},
            # No read timeout: it would cut off any agent that is silent for that long
            # between events (Synthesis is, while the Web agent works), whatever the call's
            # own deadline says. `asyncio.wait_for` in `_exchange` bounds the whole call.
            timeout=httpx.Timeout(30.0, connect=CARD_TIMEOUT_S, read=None),
        )
        self._factory = ClientFactory(
            ClientConfig(httpx_client=self._http, streaming=True)
        )
        self._by_url = {
            url.rstrip("/"): AgentInfo(url.rstrip("/"))
            for url in urls or settings.AGENT_URLS
        }

    # ------------------------------------------------------------------ discovery
    async def start(self) -> None:
        await self.refresh()

    async def close(self) -> None:
        await self._http.aclose()

    async def refresh(self) -> None:
        await asyncio.gather(
            *(self._refresh_one(info) for info in self._by_url.values())
        )

    async def _refresh_one(self, info: AgentInfo) -> None:
        try:
            card = await asyncio.wait_for(
                A2ACardResolver(self._http, info.url).get_agent_card(), CARD_TIMEOUT_S
            )
            info.card, info.name = card, card.name
            info.client = self._factory.create(card)
            info.up, info.error = True, None
        except Exception as exc:  # noqa: BLE001 - a missing agent must not stop startup
            info.up, info.error = False, f"{type(exc).__name__}: {exc}"
            logger.warning("agent at %s is down: %s", info.url, info.error)

    def agents(self) -> list[AgentInfo]:
        return list(self._by_url.values())

    def _find(self, name: str) -> AgentInfo | None:
        return next((i for i in self._by_url.values() if i.name == name), None)

    def capabilities(self) -> list[dict[str, Any]]:
        """Skills of every agent that is up, rendered from the cards (for the planner)."""
        return [
            {
                "agent": info.name,
                "skill": skill.id,
                "description": skill.description,
                "examples": list(skill.examples),
            }
            for info in self._by_url.values()
            if info.up and info.card
            for skill in info.card.skills
        ]

    # ---------------------------------------------------------------------- calls
    async def call(
        self,
        agent: str,
        request: TaskRequest,
        *,
        trace_ctx: dict[str, Any] | None = None,
        on_progress: Progress | None = None,
    ) -> TaskResult:
        """Send a fresh task. Never raises; retries once on an invalid reply."""
        timeout = request.constraints.timeout_s or self._timeout_s
        part = to_part(request)
        result = await self._send(
            agent, part, None, None, trace_ctx, timeout, on_progress
        )
        if result.error and result.error.code == "invalid_output":
            result = await self._send(
                agent, part, None, None, trace_ctx, timeout, on_progress
            )
        return result

    async def resume(
        self,
        waiting: TaskResult,
        value: dict[str, Any],
        *,
        sender: str = "supervisor",
        trace_ctx: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        on_progress: Progress | None = None,
    ) -> TaskResult:
        """Answer an input_required result on its own task. Sent once, never retried."""
        if waiting.status != "input_required" or not waiting.a2a_task_id:
            return TaskResult.failure(
                waiting.agent,
                "bad_request",
                "nothing to resume: not an input_required result",
            )
        return await self._send(
            waiting.agent,
            to_part(ResumeRequest(sender=sender, value=value)),
            waiting.a2a_task_id,
            waiting.context_id,
            trace_ctx,
            timeout_s or self._timeout_s,
            on_progress,
        )

    async def _send(
        self, agent, part, task_id, context_id, trace_ctx, timeout, on_progress
    ) -> TaskResult:
        info = self._find(agent)
        if info is None:
            # An agent that was down at startup has no name yet: look at those again.
            await asyncio.gather(
                *(self._refresh_one(i) for i in self._by_url.values() if i.name is None)
            )
            info = self._find(agent)
        elif not info.up:
            await self._refresh_one(info)  # a restarted agent comes back
        if info is None or not info.up:
            detail = info.error if info else f"no agent named '{agent}'"
            return TaskResult.failure(
                agent, "unavailable", detail or "agent is down", retryable=True
            )

        # Our own span (a child of whatever the caller is in), unless the caller pinned a context.
        span = (
            nullcontext()
            if trace_ctx
            else tracing.observation(f"a2a.call:{agent}", as_type="span")
        )
        with span:
            return await self._exchange(
                info,
                agent,
                part,
                task_id,
                context_id,
                trace_ctx or tracing.current_trace_ctx(),
                timeout,
                on_progress,
            )

    async def _exchange(
        self, info, agent, part, task_id, context_id, trace_ctx, timeout, on_progress
    ) -> TaskResult:
        message = Message(
            message_id=uuid.uuid4().hex,
            role=Role.ROLE_USER,
            parts=[part],
            task_id=task_id or "",
            context_id=context_id or "",
        )
        if trace_ctx:
            message.metadata.update({"trace_ctx": trace_ctx})
        exchange = _Exchange()

        async def consume() -> None:
            async for event in info.client.send_message(
                SendMessageRequest(message=message)
            ):
                exchange.add(event, agent, on_progress)

        try:
            await asyncio.wait_for(consume(), timeout)
        except TimeoutError, A2AClientTimeoutError:
            await self._cancel(info, exchange.task_id)
            return TaskResult.failure(
                agent, "timeout", f"no answer within {timeout:g}s", retryable=True
            )
        except A2AClientError as exc:
            info.up, info.error = False, str(exc)
            return TaskResult.failure(agent, "unavailable", str(exc), retryable=True)
        except A2AError as exc:
            return TaskResult.failure(agent, "failed", f"{type(exc).__name__}: {exc}")
        except Exception as exc:
            logger.exception("a2a call to %s broke", agent)
            return TaskResult.failure(agent, "internal", type(exc).__name__)
        return _result_from(exchange, agent)

    async def _cancel(self, info: AgentInfo, task_id: str) -> None:
        if not task_id:
            return
        try:
            await asyncio.wait_for(
                info.client.cancel_task(CancelTaskRequest(id=task_id)), CARD_TIMEOUT_S
            )
        except Exception:
            logger.debug("cancel of task %s failed", task_id, exc_info=True)
