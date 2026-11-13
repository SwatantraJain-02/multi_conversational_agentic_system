"""Validation of a plan, in code. The model proposes; this decides what is acceptable.

A free model will sometimes emit a broken plan (an agent that doesn't exist, a step that
depends on a later one, twelve steps). Rules, so a bad plan never reaches an agent:
- every (agent, skill) must exist on an agent that is up right now (from the cards);
- ids are unique, and `depends_on` may only name EARLIER steps: the plan is acyclic by
  construction, no cycle detection needed;
- at most MAX_STEPS steps; a task is non-empty;
- a synthesis step must depend on at least one sql or web step, because its inputs are
  wired from those results (`wire_synthesis`), never written by the model.
"""

from typing import Any

from multi_agenti_conversation_system.a2a.contracts import (
    MAX_TASK_CHARS,
    SynthesisData,
    TaskResult,
)
from multi_agenti_conversation_system.supervisor.state import RouteDecision

MAX_STEPS = 4
SYNTHESIS_AGENT = "synthesis"


def validate_plan(
    decision: RouteDecision, capabilities: list[dict[str, Any]]
) -> str | None:
    """None if the plan is acceptable, otherwise one sentence saying what is wrong
    (it is shown to the model so it can fix the plan)."""
    offered = {(c["agent"], c["skill"]) for c in capabilities}
    if decision.intent == "direct":
        return "a direct answer takes an empty plan" if decision.plan else None
    if not decision.plan:
        return f"intent '{decision.intent}' needs at least one step"
    if len(decision.plan) > MAX_STEPS:
        return f"too many steps ({len(decision.plan)}); at most {MAX_STEPS}"

    agents_by_id: dict[str, str] = {}
    for step in decision.plan:
        if not step.id.strip() or step.id in agents_by_id:
            return f"step ids must be unique and non-empty (got '{step.id}')"
        if (step.agent, step.skill) not in offered:
            return (
                f"step {step.id}: there is no skill '{step.skill}' on an available agent "
                f"'{step.agent}'. Available: {sorted(offered)}"
            )
        if not step.task.strip() or len(step.task) > MAX_TASK_CHARS:
            return f"step {step.id}: the task must be 1-{MAX_TASK_CHARS} characters"
        unknown = [d for d in step.depends_on if d not in agents_by_id]
        if unknown:
            return (
                f"step {step.id} depends on {unknown}, which are not EARLIER steps "
                "(a step may only depend on steps listed before it)"
            )
        if step.agent == SYNTHESIS_AGENT and not any(
            agents_by_id[d] in ("sql", "web") for d in step.depends_on
        ):
            return f"step {step.id}: a synthesis step must depend on a sql or web step"
        agents_by_id[step.id] = step.agent
    return None


def wire_synthesis(dependency_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """`TaskRequest.data` for a synthesis step: the sql result is `internal`, the web result
    `external`. Wired here from what actually came back; the model never writes it."""
    internal = external = None
    for result in dependency_results.values():
        if result["agent"] == "sql" and internal is None:
            internal = TaskResult.model_validate(result)
        elif result["agent"] == "web" and external is None:
            external = TaskResult.model_validate(result)
    return SynthesisData(internal=internal, external=external).model_dump(mode="json")
