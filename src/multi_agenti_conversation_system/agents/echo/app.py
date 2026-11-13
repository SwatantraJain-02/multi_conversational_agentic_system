"""Echo agent: proves the A2A plumbing end to end (Phase C). Not part of the product.

    uv run python -m multi_agenti_conversation_system.agents.echo.app     # :7009

Skill `echo` takes the task text and replies with it. Switches for the tests:
- a task starting with "ask:" pauses in input-required until the caller resumes it;
- a task "crash" raises inside the graph (the caller must see a clean `internal` failure);
- a task "slow" sleeps 5 s first (to exercise the caller's timeout).
"""

import asyncio
from typing import Any, TypedDict

from a2a.types import AgentSkill
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import TaskRequest, TaskResult
from multi_agenti_conversation_system.a2a.server import (
    GraphAgentExecutor,
    build_app,
    build_card,
    serve,
)
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.prompts import aget_prompt

AGENT = "echo"
PROMPT = "plumbing/echo-reply"


class EchoState(TypedDict, total=False):
    task: str
    answer: str
    approved: bool | None
    trace_ctx: dict[str, Any]
    prompt_tag: str


async def listen(state: EchoState) -> dict[str, Any]:
    if state["task"] == "crash":
        raise RuntimeError("secret internal detail")
    if state["task"] == "slow":
        await asyncio.sleep(5)
    return {}


def confirm(state: EchoState) -> dict[str, Any]:
    if state["task"].startswith("ask:"):
        decision = interrupt({"reason": "approve this echo?", "task": state["task"]})
        return {"approved": bool(decision.get("approved"))}
    return {"approved": None}


async def reply(state: EchoState, config: RunnableConfig) -> dict[str, Any]:
    # The trace context of the message that ran this node (after a resume: the resume's).
    trace_ctx = (config.get("metadata") or {}).get("trace_ctx", {})
    if state.get("approved") is False:
        return {"answer": "rejected", "trace_ctx": trace_ctx}
    prompt = await aget_prompt(PROMPT)
    # A generation linked to the prompt version: Langfuse shows which version produced it.
    with tracing.observation(
        "echo-reply", as_type="generation", prompt=prompt.link, input=state["task"]
    ) as generation:
        answer = prompt.compile(task=state["task"])
        if generation:
            generation.update(output=answer)
    return {"answer": answer, "trace_ctx": trace_ctx, "prompt_tag": prompt.tag}


def build_graph():
    graph = StateGraph(EchoState)
    graph.add_node("listen", listen)
    graph.add_node("confirm", confirm)
    graph.add_node("reply", reply)
    graph.add_edge(START, "listen")
    graph.add_edge("listen", "confirm")
    graph.add_edge("confirm", "reply")
    graph.add_edge("reply", END)
    return graph.compile(checkpointer=InMemorySaver())


def build_input(request: TaskRequest) -> dict[str, Any]:
    # Reset every per-task field: tasks in one A2A context share one checkpoint.
    return {"task": request.task, "answer": "", "approved": None}


def to_result(state: dict[str, Any], request: TaskRequest) -> TaskResult:
    return TaskResult(
        status="completed",
        agent=AGENT,
        answer=state["answer"],
        data={
            "approved": state.get("approved"),
            "task_id": request.task_id,
            "trace_ctx": state.get("trace_ctx", {}),
        },
        prompt_versions={PROMPT: state["prompt_tag"]}
        if state.get("prompt_tag")
        else {},
    )


def make_app():
    card = build_card(
        name=AGENT,
        description="Plumbing test agent: echoes the task back.",
        port=settings.ECHO_AGENT_PORT,
        skills=[
            AgentSkill(
                id="echo",
                name="Echo",
                description="Repeat the task text. Prefix 'ask:' to require approval first.",
                tags=["test"],
                examples=["hello", "ask: do the thing"],
            )
        ],
        skill_data={"echo": None},
    )
    executor = GraphAgentExecutor(
        build_graph(), AGENT, {"echo"}, build_input, to_result
    )
    return build_app(card, executor)


app = make_app()


def main() -> None:
    serve(app, settings.ECHO_AGENT_PORT)


if __name__ == "__main__":
    main()
