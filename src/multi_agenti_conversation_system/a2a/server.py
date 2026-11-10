"""Generic A2A server for an agent built as a compiled LangGraph.

    graph = ...compile(checkpointer=...)
    executor = GraphAgentExecutor(graph, "echo", {"echo"}, build_input, to_result)
    app = build_app(card, executor)            # Starlette: card + JSON-RPC + /health + bearer auth

How a graph run maps onto an A2A task:
- a fresh message carries a `TaskRequest`; `build_input(request)` makes the graph input;
- every graph node that finishes publishes a `working` status update (progress);
- the graph ends normally    -> artifact holding the `TaskResult`, task `completed`;
- the graph calls interrupt() -> task `input-required`, the interrupt payload (a dict)
  travels in the `TaskResult.data`; the caller answers with a `ResumeRequest` on the same
  task, which resumes the graph with `Command(resume=value)`;
- any failure -> task `failed` with a `TaskResult.error`. The executor never lets an
  exception escape: an uncaught one would reach the caller as a bare protocol error that
  carries the exception text.

The graph's `thread_id` is the A2A `contextId`, so every task in one context shares one
checkpoint. That has a consequence for agent authors: `build_input` must reset every
per-task field (sql, error, rows, ...), or a second task in the same context starts from
the previous task's leftovers.

The A2A task store here is in memory: restarting an agent while a task waits in
input-required loses that task (a documented limitation, see the plan).
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Collection
from contextlib import AbstractAsyncContextManager, asynccontextmanager, nullcontext
from typing import Any

import uvicorn
from a2a.helpers.proto_helpers import (
    new_message,
    new_task_from_user_message,
    new_text_message,
)
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
    HTTPAuthSecurityScheme,
    SecurityRequirement,
    SecurityScheme,
    TaskState,
)
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH, PROTOCOL_VERSION_CURRENT
from google.protobuf.json_format import MessageToDict
from langgraph.types import Command
from pydantic import BaseModel
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import (
    AgentError,
    ContractError,
    ResumeRequest,
    TaskRequest,
    TaskResult,
    contract_extension,
    from_parts,
    to_part,
)
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.mcp.common import (
    BearerAuth,
    require_token_off_loopback,
)

logger = logging.getLogger(__name__)

BuildInput = Callable[[TaskRequest], dict[str, Any]]
ToResult = Callable[[dict[str, Any], TaskRequest], TaskResult | Awaitable[TaskResult]]


class GraphAgentExecutor(AgentExecutor):
    """Runs one compiled LangGraph per A2A task (see the module docstring)."""

    def __init__(
        self,
        graph,
        agent: str,
        skills: Collection[str],
        build_input: BuildInput,
        to_result: ToResult,
        context: Any = None,
    ) -> None:
        # `graph` and `context` may be set later: an agent whose graph needs async resources
        # (MCP session, checkpointer) builds them in the app's startup hook (`build_app`).
        # `context` is passed to every run as the graph's runtime context (LangGraph `Runtime`).
        self.graph, self.agent, self.context = graph, agent, context
        self.skills = set(skills)
        self.build_input, self.to_result = build_input, to_result
        # a2a task id -> the TaskRequest that started it, kept while it waits in
        # input-required so `to_result` sees the real request after a resume. In memory,
        # like the A2A task store.
        self._requests: dict[str, TaskRequest] = {}

    async def execute(self, context: RequestContext, event_queue) -> None:
        message, task = context.message, context.current_task
        if task is None:  # a new task: the protocol wants the Task published first
            await event_queue.enqueue_event(new_task_from_user_message(message))
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)

        resuming = (
            task is not None
            and task.status.state == TaskState.TASK_STATE_INPUT_REQUIRED
        )
        try:
            if task is not None and not resuming:
                raise ContractError(
                    "a follow-up is only accepted on an input-required task"
                )
            trace_ctx = MessageToDict(message.metadata).get("trace_ctx", {})
            result = await self._run(
                context, updater, message.parts, resuming, trace_ctx
            )
        except ContractError as exc:
            result = TaskResult.failure(self.agent, "bad_request", str(exc))
        except AgentError as exc:
            result = TaskResult.failure(
                self.agent, exc.code, exc.message, retryable=exc.retryable
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("agent %s crashed on task %s", self.agent, context.task_id)
            result = TaskResult.failure(
                self.agent, "internal", "the agent hit an internal error"
            )
        if result.status != "input_required":
            self._requests.pop(context.task_id, None)
        await self._finish(updater, context, result)

    async def _run(self, context, updater, parts, resuming, trace_ctx) -> TaskResult:
        config = {
            "configurable": {"thread_id": context.context_id},
            "metadata": {"a2a_task_id": context.task_id, "trace_ctx": trace_ctx},
        }
        if resuming:
            resume = from_parts(parts, ResumeRequest)
            graph_input: Any = Command(resume=resume.value)
            request = self._requests.get(context.task_id)
            if request is None:  # e.g. the agent restarted while the task was waiting
                raise ContractError("the waiting task is unknown to this agent")
        else:
            request = from_parts(parts, TaskRequest)
            if request.skill not in self.skills:
                raise ContractError(
                    f"unknown skill '{request.skill}'; this agent has {sorted(self.skills)}"
                )
            graph_input = self.build_input(request)
            self._requests[context.task_id] = request

        interrupt_payload: dict[str, Any] | None = None
        # The agent's span, a child of the caller's span from the message metadata (G20).
        with tracing.observation(
            f"agent:{self.agent}",
            as_type="agent",
            trace_ctx=trace_ctx,
            metadata={"a2a_task_id": context.task_id, "resumed": resuming},
        ):
            async for chunk in self.graph.astream(
                graph_input, config, context=self.context, stream_mode="updates"
            ):
                for node, update in chunk.items():
                    if node == "__interrupt__":
                        value = update[0].value
                        interrupt_payload = (
                            value if isinstance(value, dict) else {"value": value}
                        )
                    else:
                        await updater.start_work(
                            new_text_message(
                                f"node: {node}", context.context_id, context.task_id
                            )
                        )

        if interrupt_payload is not None:
            return TaskResult(
                status="input_required",
                agent=self.agent,
                answer=str(interrupt_payload.get("reason", "")),
                data=interrupt_payload,
            )
        state = (await self.graph.aget_state(config)).values
        result = self.to_result(state, request)
        return await result if asyncio.iscoroutine(result) else result

    async def _finish(self, updater: TaskUpdater, context, result: TaskResult) -> None:
        part = to_part(result)
        if result.status == "completed":
            await updater.add_artifact([part])
            await updater.complete()
        elif result.status == "input_required":
            await updater.requires_input(
                new_message([part], context.context_id, context.task_id)
            )
        else:
            await updater.failed(
                new_message([part], context.context_id, context.task_id)
            )

    async def cancel(self, context: RequestContext, event_queue) -> None:
        await TaskUpdater(event_queue, context.task_id, context.context_id).cancel()


# --------------------------------------------------------------------------- card
def build_card(
    *,
    name: str,
    description: str,
    port: int,
    skills: list[AgentSkill],
    skill_data: dict[str, type[BaseModel] | None],
    version: str = "0.1.0",
) -> AgentCard:
    """An agent card for a JSON-RPC agent on localhost, with bearer auth and contracts."""
    card = AgentCard(
        name=name,
        description=description,
        version=version,
        supported_interfaces=[
            AgentInterface(
                url=f"http://{settings.A2A_HOST}:{port}/",
                protocol_binding="JSONRPC",
                protocol_version=PROTOCOL_VERSION_CURRENT,
            )
        ],
        capabilities=AgentCapabilities(
            streaming=True, extensions=[contract_extension(skill_data)]
        ),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=skills,
    )
    if settings.A2A_TOKEN:
        card.security_schemes["bearer"].CopyFrom(
            SecurityScheme(
                http_auth_security_scheme=HTTPAuthSecurityScheme(scheme="Bearer")
            )
        )
        card.security_requirements.append(SecurityRequirement(schemes={"bearer": {}}))
    return card


# ---------------------------------------------------------------------------- app
def build_app(
    card: AgentCard,
    executor: AgentExecutor,
    startup: Callable[[], AbstractAsyncContextManager[None]] | None = None,
):
    """Starlette app: public card + /health, JSON-RPC behind the bearer token.

    `startup` is an async context manager factory entered for the app's whole lifetime
    (inside the lifespan): it opens the agent's resources and sets them on the executor.
    """
    handler = DefaultRequestHandler(
        agent_executor=executor, task_store=InMemoryTaskStore(), agent_card=card
    )

    async def health(request: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "agent": card.name})

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        async with startup() if startup else nullcontext():
            yield
        # Flush the buffered spans while uvicorn shuts down gracefully. Code after
        # uvicorn.run() would not run: uvicorn re-raises SIGTERM, which kills the process.
        await asyncio.to_thread(tracing.shutdown)

    app = Starlette(
        lifespan=lifespan,
        routes=[
            *create_agent_card_routes(card),
            *create_jsonrpc_routes(handler, "/"),
            Route("/health", health),
        ],
    )
    return BearerAuth(
        app,
        token=settings.A2A_TOKEN,
        public_paths=("/health", AGENT_CARD_WELL_KNOWN_PATH),
    )


def serve(app, port: int) -> None:
    require_token_off_loopback(settings.A2A_TOKEN, settings.A2A_HOST, "A2A_TOKEN")
    try:
        uvicorn.run(app, host=settings.A2A_HOST, port=port, log_level="warning")
    finally:
        tracing.shutdown()  # send the spans still buffered (SIGTERM / Ctrl-C end up here)
