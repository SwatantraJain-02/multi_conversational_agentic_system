"""validate_plan and wire_synthesis: a bad plan must never reach an agent."""

import pytest

from multi_agenti_conversation_system.a2a.contracts import TaskResult
from multi_agenti_conversation_system.supervisor.plan import (
    MAX_STEPS,
    validate_plan,
    wire_synthesis,
)
from multi_agenti_conversation_system.supervisor.state import PlanStep, RouteDecision

CAPS = [
    {"agent": "sql", "skill": "answer_sql_question"},
    {"agent": "web", "skill": "answer_web_question"},
    {"agent": "synthesis", "skill": "synthesise_findings"},
]


def step(id_, agent, skill=None, task="a standalone question", deps=()):
    skill = (
        skill
        or {
            "sql": "answer_sql_question",
            "web": "answer_web_question",
            "synthesis": "synthesise_findings",
        }[agent]
    )
    return PlanStep(id=id_, agent=agent, skill=skill, task=task, depends_on=list(deps))


def plan(intent, *steps):
    return RouteDecision(intent=intent, plan=list(steps))


@pytest.mark.parametrize(
    "decision",
    [
        plan("sql", step("s1", "sql")),
        plan("web", step("s1", "web")),
        plan("compare", step("s1", "sql"), step("s2", "synthesis", deps=["s1"])),
        plan(
            "hybrid",
            step("s1", "sql"),
            step("s2", "web"),
            step("s3", "synthesis", deps=["s1", "s2"]),
        ),
        plan("direct"),
    ],
)
def test_the_canonical_shapes_are_valid(decision):
    assert validate_plan(decision, CAPS) is None


@pytest.mark.parametrize(
    "decision,fragment",
    [
        (plan("direct", step("s1", "sql")), "empty plan"),
        (plan("sql"), "at least one step"),
        (plan("sql", step("s1", "ghost", "do_it")), "no skill 'do_it'"),
        (
            plan("sql", step("s1", "sql", "answer_web_question")),
            "no skill 'answer_web_question'",
        ),
        (plan("sql", step("s1", "sql"), step("s1", "web")), "unique"),
        (plan("sql", step("", "sql")), "unique and non-empty"),
        (plan("sql", step("s1", "sql", task="  ")), "task must be"),
        (plan("sql", step("s1", "sql", task="x" * 4001)), "task must be"),
        (
            plan("compare", step("s1", "synthesis", deps=["s2"]), step("s2", "sql")),
            "not EARLIER",
        ),
        (plan("compare", step("s1", "sql", deps=["s1"])), "not EARLIER"),
        (plan("compare", step("s1", "synthesis")), "must depend on a sql or web step"),
        (plan("compare", step("s1", "synthesis", deps=["s0"])), "not EARLIER"),
        (
            plan("sql", *[step(f"s{i}", "sql") for i in range(MAX_STEPS + 1)]),
            "too many steps",
        ),
    ],
)
def test_broken_plans_are_rejected_with_a_reason_the_model_can_use(decision, fragment):
    assert fragment in validate_plan(decision, CAPS)


def test_an_agent_that_is_down_cannot_be_planned():
    only_sql = [c for c in CAPS if c["agent"] == "sql"]
    assert "no skill 'answer_web_question'" in validate_plan(
        plan("web", step("s1", "web")), only_sql
    )


def test_a_synthesis_step_that_depends_on_another_synthesis_is_not_enough():
    decision = plan(
        "compare",
        step("s1", "sql"),
        step("s2", "synthesis", deps=["s1"]),
        step("s3", "synthesis", deps=["s2"]),
    )
    assert "must depend on a sql or web step" in validate_plan(decision, CAPS)


def result(agent, **kw):
    return TaskResult(
        **{"status": "completed", "agent": agent, "answer": f"{agent} answer", **kw}
    ).model_dump(mode="json")


def test_synthesis_inputs_are_wired_from_the_results_by_agent():
    data = wire_synthesis({"s1": result("sql", sql="SELECT 1"), "s2": result("web")})
    assert data["internal"]["agent"] == "sql" and data["internal"]["sql"] == "SELECT 1"
    assert data["external"]["agent"] == "web"


def test_a_missing_side_stays_missing_and_a_failed_one_is_passed_on_as_failed():
    only_sql = wire_synthesis({"s1": result("sql")})
    assert (
        only_sql["external"] is None and only_sql["internal"]["status"] == "completed"
    )
    failed_web = result(
        "web",
        status="failed",
        answer="",
        error={"code": "unavailable", "message": "down"},
    )
    data = wire_synthesis({"s1": result("sql"), "s2": failed_web})
    assert (
        data["external"]["status"] == "failed"
    )  # Synthesis decides what to do with it
