"""The supervisor's graph nodes and routing.

    planner -> supervisor --(ready steps)--> run_step x N (parallel, Send) --> supervisor
                  |  \\--(a step needs approval)--> hitl_gate -> resume_dispatch --> supervisor
                  |--(plan done / budget / stuck)--> finalize -> summarize_history -> END
                  \\--(intent direct)--------------> direct_answer -> summarize_history -> END

- `planner` (LLM, structured) proposes a plan; `validate_plan` decides if it is acceptable,
  with one repair attempt. Sub-agents don't see the conversation, so every step's task is a
  standalone question written by the planner.
- `supervisor` is deterministic: it dispatches every step whose dependencies are finished (in
  parallel, in one superstep), within a budget of `max_dispatches` per turn, and calls no LLM.
- HITL across A2A takes three nodes because `interrupt()` re-runs its whole node (G22):
  `run_step` sends the task; if the agent answers `input_required` that result waits in
  `results`; `hitl_gate` only calls `interrupt()`; `resume_dispatch` sends the decision on the
  same A2A task. Approvals are handled one at a time, so the payload names its step.
- `finalize` composes the answer in code from what the agents returned: it never writes a
  competing answer. A failed step gives a partial answer that names what is missing.
  Re-planning after a failed step is deliberately not done (see the plan document).

Nodes get everything from `runtime.context`. They never raise for an agent's failure: that is
data (`TaskResult.error`).
"""

import logging
import time
from typing import Any, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage
from langgraph.runtime import Runtime
from langgraph.types import Send, interrupt

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import (
    TaskConstraints,
    TaskContext,
    TaskRequest,
    TaskResult,
)
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.guardrails.output_guard import guard_output
from multi_agenti_conversation_system.prompts import aget_prompt
from multi_agenti_conversation_system.supervisor.context import (
    DEFAULT_TIMEOUT_S,
    SupervisorContext,
)
from multi_agenti_conversation_system.supervisor.memory_nodes import memory_block
from multi_agenti_conversation_system.supervisor.plan import (
    SYNTHESIS_AGENT,
    validate_plan,
    wire_synthesis,
)
from multi_agenti_conversation_system.supervisor.state import AgentState

logger = logging.getLogger(__name__)

HISTORY_MESSAGES_FOR_PLANNING = 6
PLAN_ATTEMPTS = 2  # the first plan, and one repair
Rt = Runtime[SupervisorContext]


class StepTask(TypedDict):
    step: dict[str, Any]
    deps: dict[str, dict[str, Any]]
    domain: str | None
    profile: dict[str, str]
    memories: list[str]  # the top-k relevant facts, not the whole store


# ----------------------------------------------------------------------------- helpers
def _text(message: AnyMessage) -> str:
    return message.content if isinstance(message.content, str) else str(message.content)


def history_text(state: AgentState, last: int = HISTORY_MESSAGES_FOR_PLANNING) -> str:
    """The summary of trimmed history, then the last few messages, as plain text."""
    lines = []
    if state.get("summary"):
        lines.append(f"(Earlier in this conversation: {state['summary']})")
    # The newest message is the question itself; the planner gets it separately.
    for message in state.get("messages", [])[:-1][-last:]:
        speaker = "User" if isinstance(message, HumanMessage) else "Assistant"
        lines.append(f"{speaker}: {_text(message)[:600]}")
    return "\n".join(lines) or "(this is the start of the conversation)"


def render_capabilities(capabilities: list[dict[str, Any]]) -> str:
    return (
        "\n".join(
            f"- agent `{c['agent']}`, skill `{c['skill']}`: {' '.join(c['description'].split())}"
            + (f" Examples: {'; '.join(c['examples'])}" if c["examples"] else "")
            for c in capabilities
        )
        or "(no agents are available)"
    )


def _pending(state: AgentState) -> list[str]:
    """Step ids whose result is waiting for a human, in plan order."""
    return [
        s["id"]
        for s in state["plan"]
        if state["results"].get(s["id"], {}).get("status") == "input_required"
    ]


# ------------------------------------------------------------------------------- nodes
async def planner(state: AgentState, runtime: Rt) -> dict[str, Any]:
    ctx = runtime.context
    capabilities = ctx.registry.capabilities()
    if not capabilities:
        return {
            "plan": [],
            "intent": "none",
            "plan_error": "no agents are available right now",
        }

    prompt = await aget_prompt("supervisor/plan")
    feedback, error = "", "no plan"
    for _ in range(PLAN_ATTEMPTS):
        messages = prompt.compile(
            capabilities=render_capabilities(capabilities),
            history=memory_block(state) + history_text(state),
            question=state.get("standalone_question") or state["question"],
            domain=state.get("selected_domain") or "none",
            feedback=feedback,
        )
        try:
            decision = await tracing.generation(
                ctx.plan,
                messages,
                "plan",
                prompt_link=prompt.link,
                model=settings.CHAT_MODEL,
            )
        except Exception as exc:  # noqa: BLE001 - an unusable reply is an attempt, not a crash
            logger.warning("planner call failed: %s", exc)
            error = "your reply was not valid JSON for the plan schema (or the model failed)"
            feedback = f"Your previous reply was unusable: {error}. Reply with valid JSON only.\n\n"
            continue
        error = validate_plan(decision, capabilities) or ""
        if not error:
            return {
                "intent": decision.intent,
                "plan": [s.model_dump() for s in decision.plan],
                "plan_error": "",
                "prompt_versions": {
                    **state.get("prompt_versions", {}),
                    prompt.name: prompt.tag,
                },
            }
        logger.info("plan rejected: %s", error)
        feedback = (
            f"Your previous plan was rejected: {error}\nWrite a corrected plan.\n\n"
        )
    return {"plan": [], "intent": "none", "plan_error": f"no valid plan ({error})"}


def supervisor(state: AgentState, runtime: Rt) -> dict[str, Any]:
    """Deterministic: which steps can run now? No LLM."""
    if state.get("plan_error") or not state["plan"] or _pending(state):
        return {"ready": []}
    finished = set(
        state["results"]
    )  # completed or failed; input_required excluded above
    dispatched = list(state["dispatched"])
    ready = [
        s["id"]
        for s in state["plan"]
        if s["id"] not in dispatched and all(d in finished for d in s["depends_on"])
    ]
    budget = runtime.context.max_dispatches - state["dispatch_count"]
    over = ready[max(budget, 0) :]
    ready = ready[: max(budget, 0)]
    return {
        "ready": ready,
        "dispatched": [
            *dispatched,
            *ready,
            *over,
        ],  # `over` is never dispatched: no loop
        "dispatch_count": state["dispatch_count"] + len(ready),
        "skipped": [*state["skipped"], *over],
    }


def route_after_supervisor(state: AgentState) -> list[Send] | str:
    if state.get("plan_error"):
        return "finalize"
    if not state["plan"]:
        return "direct_answer" if state["intent"] == "direct" else "finalize"
    if _pending(state):
        return "hitl_gate"
    if state["ready"]:
        steps = {s["id"]: s for s in state["plan"]}
        return [
            Send(
                "run_step",
                StepTask(
                    step=steps[sid],
                    deps={d: state["results"][d] for d in steps[sid]["depends_on"]},
                    domain=state.get("selected_domain"),
                    profile=state.get("profile", {}),
                    memories=state.get("facts", []),
                ),
            )
            for sid in state["ready"]
        ]
    return "finalize"  # everything ran, or what is left can't (budget, a skipped dependency)


async def run_step(task: StepTask, runtime: Rt) -> dict[str, Any]:
    """Send one step to its agent over A2A. The result (a failure too) is data."""
    ctx, step = runtime.context, task["step"]
    data = wire_synthesis(task["deps"]) if step["agent"] == SYNTHESIS_AGENT else {}
    session = (
        {"selected_domain": task["domain"]}
        if task["domain"] and step["agent"] == "sql"
        else {}
    )
    request = TaskRequest(
        sender="supervisor",
        skill=step["skill"],
        task=step["task"],
        data=data,
        context=TaskContext(
            profile=task["profile"], memories=task["memories"], session=session
        ),
        constraints=TaskConstraints(
            timeout_s=ctx.skill_timeouts.get(step["skill"], DEFAULT_TIMEOUT_S)
        ),
    )
    started = time.time()
    result = await ctx.registry.call(step["agent"], request)
    return {
        "results": {step["id"]: result.model_dump(mode="json")},
        "timings": {step["id"]: [started, time.time()]},
    }


def hitl_gate(state: AgentState) -> dict[str, Any]:
    """Pause for a human on the first waiting step. Only interrupt() here (G22)."""
    step_id = _pending(state)[0]
    waiting = state["results"][step_id]
    decision = interrupt(
        {
            "step_id": step_id,
            "agent": waiting["agent"],
            "reason": waiting["answer"],
            **waiting["data"],
        }
    )
    approved = isinstance(decision, dict) and decision.get("approved") is True
    return {
        "hitl_decision": {"step_id": step_id, "approved": approved},
        "approvals": [
            {"step_id": step_id, "agent": waiting["agent"], "approved": approved}
        ],
    }


async def resume_dispatch(state: AgentState, runtime: Rt) -> dict[str, Any]:
    """Send the human's decision on the agent's own A2A task; its result replaces the waiting one."""
    ctx, decision = runtime.context, state["hitl_decision"]
    step_id = decision["step_id"]
    waiting = TaskResult.model_validate(state["results"][step_id])
    skill = next(s["skill"] for s in state["plan"] if s["id"] == step_id)
    result = await ctx.registry.resume(
        waiting,
        {"approved": decision["approved"]},
        timeout_s=ctx.skill_timeouts.get(skill, DEFAULT_TIMEOUT_S),
    )
    return {"results": {step_id: result.model_dump(mode="json")}, "hitl_decision": None}


def finalize(state: AgentState) -> dict[str, Any]:
    """Compose the answer from the agents' results, in code. A gap is named, never hidden."""
    if state.get("plan_error"):
        answer = (
            f"I couldn't work out how to answer that ({state['plan_error']}). "
            "Could you rephrase it, or ask about something specific?"
        )
        return _answer(state, answer, partial=True)

    steps = {s["id"]: s for s in state["plan"]}
    results = state["results"]
    done = [
        (sid, results[sid])
        for sid in steps
        if results.get(sid, {}).get("status") == "completed"
    ]
    failed = [
        (sid, results[sid])
        for sid in steps
        if results.get(sid, {}).get("status") == "failed"
    ]
    never_ran = [sid for sid in steps if sid not in results]

    synthesised = [r for sid, r in done if steps[sid]["agent"] == SYNTHESIS_AGENT]
    if synthesised:
        primary = synthesised[-1]
        body, citations = primary["answer"], primary["citations"]
        sql = primary.get("sql") or next(
            (r.get("sql") for _, r in done if r.get("sql")), None
        )
    elif len(done) == 1:
        body, citations, sql = (
            done[0][1]["answer"],
            done[0][1]["citations"],
            done[0][1].get("sql"),
        )
    else:
        body = "\n\n".join(f"**{r['agent']}**: {r['answer']}" for _, r in done)
        citations = [c for _, r in done for c in r["citations"]]
        sql = next((r.get("sql") for _, r in done if r.get("sql")), None)

    # Gaps the agents didn't already cover themselves (Synthesis names its own).
    covered = bool(synthesised)
    gaps = [
        f"the {r['agent']} agent: {r['error']['message'] if r.get('error') else 'failed'}"
        for _, r in failed
        if not covered
    ] + [
        f"step {sid} was not run ({'budget' if sid in state['skipped'] else 'a dependency failed'})"
        for sid in never_ran
    ]
    if gaps and body:
        body = f"{body}\n\nPart of this could not be completed: {'; '.join(gaps)}."
    elif gaps:
        body = f"I couldn't answer that: {'; '.join(gaps)}."
    partial = bool(gaps) or any(r["data"].get("partial") for _, r in done)
    return _answer(state, body, citations=citations, sql=sql, partial=partial)


def _answer(
    state: AgentState,
    answer: str,
    *,
    citations: list[dict[str, Any]] | None = None,
    sql: str | None = None,
    partial: bool = False,
) -> dict[str, Any]:
    # The last look: secrets and unapproved personal data out, dangling [n] markers out.
    answer = guard_output(
        answer,
        citations or [],
        pii_approved=any(a.get("approved") for a in state.get("approvals", [])),
    )
    results = state.get("results", {})
    agents = [r["agent"] for r in results.values()]
    if any(
        r["agent"] == SYNTHESIS_AGENT and r["data"].get("delegated")
        for r in results.values()
    ):
        agents.append("web")  # Synthesis asked the Web agent itself
    versions = dict(state.get("prompt_versions", {}))
    for result in results.values():
        versions.update(result.get("prompt_versions", {}))
    return {
        "answer": answer,
        "citations": citations or [],
        "sql_used": sql,
        "agents_used": list(dict.fromkeys(agents)),
        "partial": partial,
        "prompt_versions": versions,
        "messages": [AIMessage(content=answer)],
    }


async def direct_answer(state: AgentState, runtime: Rt) -> dict[str, Any]:
    """No lookup needed: reply from the conversation (greetings, 'what can you do', follow-ups)."""
    ctx = runtime.context
    prompt = await aget_prompt("supervisor/direct")
    messages = prompt.compile(
        capabilities=render_capabilities(ctx.registry.capabilities()),
        history=memory_block(state) + history_text(state),
        question=state["question"],
    )
    try:
        reply = await tracing.generation(
            ctx.direct,
            messages,
            "direct-answer",
            prompt_link=prompt.link,
            model=settings.CHAT_MODEL,
        )
        answer = str(reply).strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("direct answer failed: %s", exc)
        return _answer(
            state,
            "I couldn't answer that right now (the language model is unavailable).",
            partial=True,
        )
    update = _answer(state, answer)
    update["prompt_versions"] = {**update["prompt_versions"], prompt.name: prompt.tag}
    return update


async def summarize_history(state: AgentState, runtime: Rt) -> dict[str, Any]:
    """Past the limits, fold the older messages into `summary` and drop them."""
    ctx, messages = runtime.context, state["messages"]
    tokens = sum(len(_text(m)) for m in messages) // 4
    if len(messages) <= ctx.history_max_messages and tokens <= ctx.history_max_tokens:
        return {}
    old = messages[: -ctx.history_keep_last] if ctx.history_keep_last else messages
    if not old:
        return {}
    prompt = await aget_prompt("supervisor/summarise")
    transcript = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {_text(m)[:1200]}"
        for m in old
    )
    try:
        summary = await tracing.generation(
            ctx.summarise,
            prompt.compile(
                summary=state.get("summary") or "(none)", transcript=transcript
            ),
            "summarise-history",
            prompt_link=prompt.link,
            model=settings.CHAT_MODEL,
        )
    except Exception as exc:  # noqa: BLE001 - trimming is housekeeping: keep the history as is
        logger.warning("history summary failed (%s), keeping all messages", exc)
        return {}
    return {
        "summary": str(summary).strip(),
        "messages": [RemoveMessage(id=m.id) for m in old],
    }
