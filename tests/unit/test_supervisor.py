"""The supervisor graph with a fake A2A registry (with real latency), a scripted planner and
fake models. Covers routing, parallel dispatch, the dispatch budget, HITL across A2A, partial
answers, history trimming, and per-turn isolation."""

import asyncio
import time

from langgraph.checkpoint.memory import InMemorySaver

from multi_agenti_conversation_system.a2a.contracts import TaskResult
from multi_agenti_conversation_system.supervisor.context import SupervisorContext
from multi_agenti_conversation_system.supervisor.graph import build_graph
from multi_agenti_conversation_system.supervisor.service import Supervisor
from multi_agenti_conversation_system.supervisor.state import PlanStep, RouteDecision

CAPS = [
    {
        "agent": "sql",
        "skill": "answer_sql_question",
        "description": "company data",
        "examples": [],
    },
    {
        "agent": "web",
        "skill": "answer_web_question",
        "description": "the web",
        "examples": ["market trends"],
    },
    {
        "agent": "synthesis",
        "skill": "synthesise_findings",
        "description": "combine",
        "examples": [],
    },
]
SKILL = {
    "sql": "answer_sql_question",
    "web": "answer_web_question",
    "synthesis": "synthesise_findings",
}


def step(id_, agent, task="a standalone question", deps=()):
    return PlanStep(
        id=id_,
        agent=agent,
        skill=SKILL.get(agent, "do_it"),
        task=task,
        depends_on=list(deps),
    )


def decision(intent, *steps):
    return RouteDecision(intent=intent, plan=list(steps), reason="test")


def done(agent, answer=None, **kw):
    return TaskResult(
        status="completed", agent=agent, answer=answer or f"{agent} answer", **kw
    )


class FakeRegistry:
    """Replies by agent; records every request with the time it started and ended."""

    def __init__(self, replies=None, delays=None, caps=CAPS):
        self.replies, self.delays, self.caps = replies or {}, delays or {}, caps
        self.calls, self.resumes = [], []

    def capabilities(self):
        return self.caps

    async def call(self, agent, request, **kw):
        started = time.monotonic()
        await asyncio.sleep(self.delays.get(agent, 0))
        self.calls.append((agent, request, started, time.monotonic()))
        reply = self.replies.get(agent) or done(agent)
        return reply(request) if callable(reply) else reply

    async def resume(self, waiting, value, **kw):
        self.resumes.append((waiting, value, kw))
        reply = self.replies.get("resume") or done(
            waiting.agent, "after approval", sql="SELECT 1"
        )
        return reply


class Planner:
    def __init__(self, *outputs):
        self.outputs, self.seen = list(outputs), []

    async def __call__(self, messages):
        self.seen.append(messages)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


class Text:
    def __init__(self, reply="ok"):
        self.reply, self.seen = reply, []

    async def __call__(self, messages):
        self.seen.append(messages)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


def make(registry, planner, saver=None, **limits):
    direct, summarise = (
        Text("Hello! I can look up data."),
        Text("They asked about X; we found Y."),
    )
    ctx = SupervisorContext(
        registry=registry, plan=planner, direct=direct, summarise=summarise, **limits
    )
    graph = build_graph(saver or InMemorySaver())
    return Supervisor(graph, ctx), direct, summarise


# ----------------------------------------------------------------------- the shapes
async def test_a_sql_question_goes_to_the_sql_agent_and_its_answer_is_the_answer():
    registry = FakeRegistry(
        {
            "sql": done(
                "sql", "There are 91 customers.", sql="SELECT COUNT(*) FROM customers"
            )
        }
    )
    sup, *_ = make(
        registry, Planner(decision("sql", step("s1", "sql", "How many customers?")))
    )
    turn = await sup.ask("t", "u", "How many customers do we have?")
    assert turn.status == "answered" and turn.answer == "There are 91 customers."
    assert turn.sql_used == "SELECT COUNT(*) FROM customers" and turn.agents_used == [
        "sql"
    ]
    assert (
        turn.intent == "sql"
        and [s["id"] for s in turn.plan] == ["s1"]
        and not turn.partial
    )
    assert [(a, r.task, r.skill) for a, r, *_ in registry.calls] == [
        ("sql", "How many customers?", "answer_sql_question")
    ]
    history = await sup.history("t")
    assert [kind for kind, _ in history["messages"]] == ["HumanMessage", "AIMessage"]


async def test_a_web_question_goes_to_the_web_agent_with_its_citations():
    cites = [{"n": 1, "url": "https://a.com", "title": "A", "published_date": None}]
    registry = FakeRegistry({"web": done("web", "Trends are X [1].", citations=cites)})
    sup, *_ = make(registry, Planner(decision("web", step("s1", "web"))))
    turn = await sup.ask("t", "u", "market trends?")
    assert (
        turn.citations == cites
        and turn.agents_used == ["web"]
        and turn.sql_used is None
    )


async def test_compare_runs_sql_then_synthesis_wired_from_the_sql_result():
    registry = FakeRegistry(
        {
            "sql": done("sql", "Revenue 103k", sql="SELECT 2"),
            "synthesis": done(
                "synthesis",
                "Our 103k vs market +5% [1].",
                sql="SELECT 2",
                data={"delegated": True},
                citations=[
                    {
                        "n": 1,
                        "url": "https://m.com",
                        "title": "M",
                        "published_date": None,
                    }
                ],
            ),
        }
    )
    sup, *_ = make(
        registry,
        Planner(
            decision(
                "compare",
                step("s1", "sql"),
                step("s2", "synthesis", "compare ours with the market", ["s1"]),
            )
        ),
    )
    turn = await sup.ask("t", "u", "compare")
    sql_call, synth_call = registry.calls
    assert sql_call[0] == "sql" and synth_call[0] == "synthesis"
    assert (
        synth_call[1].data["internal"]["answer"] == "Revenue 103k"
        and synth_call[1].data["external"] is None
    )
    assert turn.answer == "Our 103k vs market +5% [1]." and turn.sql_used == "SELECT 2"
    assert turn.agents_used == [
        "sql",
        "synthesis",
        "web",
    ]  # Synthesis asked the Web agent itself


async def test_independent_steps_run_in_parallel_then_synthesis_gets_both():
    registry = FakeRegistry(delays={"sql": 0.3, "web": 0.3})
    plan = decision(
        "hybrid",
        step("s1", "sql"),
        step("s2", "web"),
        step("s3", "synthesis", "combine", ["s1", "s2"]),
    )
    sup, *_ = make(registry, Planner(plan))
    turn = await sup.ask("t", "u", "both")
    spans = {a: (s, e) for a, _, s, e in registry.calls}
    assert (
        spans["sql"][0] < spans["web"][1] and spans["web"][0] < spans["sql"][1]
    )  # intervals overlap
    assert spans["synthesis"][0] >= max(
        spans["sql"][1], spans["web"][1]
    )  # only after both finished
    synth_request = next(r for a, r, *_ in registry.calls if a == "synthesis")
    assert (
        synth_request.data["internal"]["agent"] == "sql"
        and synth_request.data["external"]["agent"] == "web"
    )
    assert (
        turn.timings["s1"][0] < turn.timings["s2"][1]
        and turn.timings["s2"][0] < turn.timings["s1"][1]
    )


async def test_a_direct_intent_answers_without_calling_any_agent():
    registry = FakeRegistry()
    sup, direct, _ = make(registry, Planner(decision("direct")))
    turn = await sup.ask("t", "u", "hi there")
    assert (
        turn.answer == "Hello! I can look up data."
        and registry.calls == []
        and turn.agents_used == []
    )
    assert "hi there" in direct.seen[0][1]["content"]


async def test_the_domain_focus_reaches_only_the_sql_step():
    registry = FakeRegistry()
    plan = decision(
        "hybrid",
        step("s1", "sql"),
        step("s2", "web"),
        step("s3", "synthesis", "c", ["s1", "s2"]),
    )
    sup, *_ = make(registry, Planner(plan))
    await sup.ask("t", "u", "q", domain="sales")
    by_agent = {a: r.context.session for a, r, *_ in registry.calls}
    assert by_agent["sql"] == {"selected_domain": "sales"} and by_agent["web"] == {}


async def test_each_skill_gets_its_own_deadline():
    registry = FakeRegistry()
    plan = decision("compare", step("s1", "sql"), step("s2", "synthesis", "c", ["s1"]))
    sup, *_ = make(registry, Planner(plan))
    await sup.ask("t", "u", "q")
    assert {a: r.constraints.timeout_s for a, r, *_ in registry.calls} == {
        "sql": 120.0,
        "synthesis": 240.0,
    }


# ---------------------------------------------------------------------- the planner
async def test_an_invalid_plan_is_repaired_once_with_the_reason():
    bad = decision("sql", step("s1", "ghost"))
    planner = Planner(bad, decision("sql", step("s1", "sql")))
    registry = FakeRegistry()
    sup, *_ = make(registry, planner)
    turn = await sup.ask("t", "u", "q")
    assert turn.answer == "sql answer" and len(planner.seen) == 2
    assert (
        "was rejected" in planner.seen[1][1]["content"]
        and "ghost" in planner.seen[1][1]["content"]
    )


async def test_two_bad_plans_give_a_graceful_answer_and_call_no_agent():
    registry = FakeRegistry()
    sup, *_ = make(registry, Planner(decision("sql", step("s1", "ghost"))))
    turn = await sup.ask("t", "u", "q")
    assert registry.calls == [] and turn.partial is True
    assert "couldn't work out how to answer" in turn.answer


async def test_an_unparseable_planner_reply_counts_as_an_attempt_and_is_repaired():
    planner = Planner(ValueError("not json"), decision("sql", step("s1", "sql")))
    sup, *_ = make(FakeRegistry(), planner)
    turn = await sup.ask("t", "u", "q")
    assert turn.answer == "sql answer" and "unusable" in planner.seen[1][1]["content"]


async def test_no_agents_up_is_said_plainly():
    sup, *_ = make(FakeRegistry(caps=[]), Planner(decision("direct")))
    turn = await sup.ask("t", "u", "q")
    assert "no agents are available" in turn.answer and turn.partial


async def test_the_planner_sees_the_conversation_so_follow_ups_can_be_made_standalone():
    registry = FakeRegistry({"sql": done("sql", "91 customers in total.")})
    planner = Planner(
        decision("sql", step("s1", "sql", "How many customers in Germany?"))
    )
    sup, *_ = make(registry, planner)
    await sup.ask("t", "u", "How many customers do we have?")
    await sup.ask("t", "u", "and in Germany?")
    second = planner.seen[1][1]["content"]
    assert (
        "User: How many customers do we have?" in second
        and "Assistant: 91 customers in total." in second
    )
    assert second.rstrip().endswith("and in Germany?")


# -------------------------------------------------------------- failure and the budget
async def test_a_failed_only_step_gives_an_answer_that_names_what_is_missing():
    down = TaskResult.failure("web", "unavailable", "Network communication error")
    sup, *_ = make(
        FakeRegistry({"web": down}), Planner(decision("web", step("s1", "web")))
    )
    turn = await sup.ask("t", "u", "trends?")
    assert turn.status == "answered" and turn.partial is True
    assert (
        "couldn't answer" in turn.answer
        and "web agent" in turn.answer
        and "Network communication error" in turn.answer
    )


async def test_a_failed_input_still_reaches_synthesis_which_degrades_itself():
    down = TaskResult.failure("web", "unavailable", "down")
    registry = FakeRegistry(
        {
            "web": down,
            "synthesis": done(
                "synthesis", "Internal only (partial).", data={"partial": True}
            ),
        }
    )
    plan = decision(
        "hybrid",
        step("s1", "sql"),
        step("s2", "web"),
        step("s3", "synthesis", "c", ["s1", "s2"]),
    )
    sup, *_ = make(registry, Planner(plan))
    turn = await sup.ask("t", "u", "q")
    synth = next(r for a, r, *_ in registry.calls if a == "synthesis")
    assert synth.data["external"]["status"] == "failed"
    assert (
        turn.answer == "Internal only (partial)." and turn.partial is True
    )  # propagated, not hidden


async def test_part_of_the_plan_failing_is_noted_when_nothing_covers_it():
    down = TaskResult.failure("web", "timeout", "no answer within 120s")
    registry = FakeRegistry({"web": down})
    plan = decision("hybrid", step("s1", "sql"), step("s2", "web"))  # no synthesis step
    sup, *_ = make(registry, Planner(plan))
    turn = await sup.ask("t", "u", "q")
    assert (
        "sql answer" in turn.answer
        and "Part of this could not be completed" in turn.answer
        and turn.partial
    )


async def test_the_dispatch_budget_is_enforced_and_the_skipped_steps_are_named():
    registry = FakeRegistry()
    plan = decision(
        "hybrid",
        step("s1", "sql"),
        step("s2", "web"),
        step("s3", "sql"),
        step("s4", "web"),
    )
    sup, *_ = make(registry, Planner(plan), max_dispatches=2)
    turn = await sup.ask("t", "u", "q")
    assert len(registry.calls) == 2
    assert turn.partial and "was not run (budget)" in turn.answer


async def test_a_step_whose_dependency_was_never_run_is_not_run_either():
    registry = FakeRegistry()
    plan = decision(
        "compare",
        step("s1", "sql"),
        step("s2", "web"),
        step("s3", "synthesis", "c", ["s2"]),
    )
    sup, *_ = make(registry, Planner(plan), max_dispatches=1)  # only s1 runs
    turn = await sup.ask("t", "u", "q")
    assert [a for a, *_ in registry.calls] == ["sql"] and turn.partial
    assert "step s3 was not run" in turn.answer


# ----------------------------------------------------------------------------- HITL
WAITING = TaskResult(
    status="input_required",
    agent="sql",
    answer="Approval needed: it reads personal-data columns.",
    data={
        "sql": "SELECT contact_name FROM customers",
        "reason": "Approval needed",
        "pii_columns": ["customers.contact_name"],
        "est_rows": 11,
    },
    a2a_task_id="task-1",
    context_id="ctx-1",
)


async def test_a_sql_step_waiting_for_approval_pauses_the_turn_and_resumes_on_the_same_task():
    registry = FakeRegistry({"sql": WAITING})
    sup, *_ = make(registry, Planner(decision("sql", step("s1", "sql"))))
    paused = await sup.ask("t", "u", "contact names?")
    assert paused.status == "needs_approval"
    assert paused.approval["step_id"] == "s1" and paused.approval["sql"].startswith(
        "SELECT contact_name"
    )
    assert paused.approval["pii_columns"] == ["customers.contact_name"]
    assert registry.resumes == []  # nothing sent until a human decides

    turn = await sup.resume("t", approved=True)
    waiting, value, _kw = registry.resumes[0]
    assert (waiting.a2a_task_id, waiting.context_id, value) == (
        "task-1",
        "ctx-1",
        {"approved": True},
    )
    assert (
        turn.status == "answered"
        and turn.answer == "after approval"
        and turn.sql_used == "SELECT 1"
    )
    assert len(registry.calls) == 1  # the resume was not a second dispatch


async def test_a_declined_approval_is_sent_and_the_agents_reply_is_the_answer():
    registry = FakeRegistry(
        {
            "sql": WAITING,
            "resume": done("sql", "The query was not run: approval was declined."),
        }
    )
    sup, *_ = make(registry, Planner(decision("sql", step("s1", "sql"))))
    await sup.ask("t", "u", "q")
    turn = await sup.resume("t", approved=False)
    assert (
        registry.resumes[0][1] == {"approved": False}
        and "declined" in turn.answer
        and not turn.partial
    )


async def test_an_approval_inside_a_larger_plan_lets_the_rest_continue():
    registry = FakeRegistry({"sql": WAITING})
    plan = decision("compare", step("s1", "sql"), step("s2", "synthesis", "c", ["s1"]))
    sup, *_ = make(registry, Planner(plan))
    paused = await sup.ask("t", "u", "q")
    assert paused.status == "needs_approval" and [a for a, *_ in registry.calls] == [
        "sql"
    ]  # synthesis waits
    turn = await sup.resume("t", approved=True)
    assert [a for a, *_ in registry.calls] == [
        "sql",
        "synthesis",
    ] and turn.answer == "synthesis answer"
    synth = registry.calls[1][1]
    assert (
        synth.data["internal"]["answer"] == "after approval"
    )  # the approved result, not the waiting one


# --------------------------------------------------------------- turns and memory
async def test_a_second_turn_starts_clean_but_keeps_the_conversation():
    registry = FakeRegistry({"web": done("web", "web answer")})
    planner = Planner(
        decision("sql", step("s1", "sql")), decision("web", step("s1", "web"))
    )
    sup, *_ = make(registry, planner)
    first = await sup.ask("t", "u", "one")
    second = await sup.ask("t", "u", "two")
    assert first.agents_used == ["sql"] and second.agents_used == [
        "web"
    ]  # no leftovers from turn 1
    assert second.results.keys() == {"s1"} and second.results["s1"]["agent"] == "web"
    assert len((await sup.history("t"))["messages"]) == 4


async def test_threads_are_independent():
    sup, *_ = make(FakeRegistry(), Planner(decision("sql", step("s1", "sql"))))
    await sup.ask("a", "u", "one")
    await sup.ask("b", "u", "two")
    assert (
        len((await sup.history("a"))["messages"]) == 2
        and len((await sup.history("b"))["messages"]) == 2
    )


async def test_a_new_supervisor_with_the_same_saver_continues_the_conversation():
    """What a restart does: the process is new, the checkpoint (Postgres in production) is not."""
    saver = InMemorySaver()
    registry = FakeRegistry({"sql": done("sql", "91 customers.")})
    first, *_ = make(registry, Planner(decision("sql", step("s1", "sql"))), saver=saver)
    await first.ask("t", "u", "How many customers?")

    planner = Planner(decision("direct"))
    second, *_ = make(registry, planner, saver=saver)
    assert len((await second.history("t"))["messages"]) == 2
    await second.ask("t", "u", "which is more?")
    assert "Assistant: 91 customers." in planner.seen[0][1]["content"]


async def test_turns_of_one_thread_never_run_at_the_same_time():
    registry = FakeRegistry(delays={"sql": 0.2})
    sup, *_ = make(registry, Planner(decision("sql", step("s1", "sql"))))
    await asyncio.gather(sup.ask("t", "u", "one"), sup.ask("t", "u", "two"))
    (_, _, s1, e1), (_, _, s2, e2) = registry.calls
    assert s2 >= e1 or s1 >= e2  # one finished before the other began


async def test_progress_events_include_the_plan():
    events = []
    sup, *_ = make(FakeRegistry(), Planner(decision("sql", step("s1", "sql"))))
    await sup.ask("t", "u", "q", on_event=events.append)
    nodes = [e["node"] for e in events if e["type"] == "node"]
    assert (
        nodes[0] == "input_guard"
        and "planner" in nodes
        and "run_step" in nodes
        and nodes[-1] == "summarize_history"
    )
    plan_event = next(e for e in events if e["type"] == "plan")
    assert plan_event["intent"] == "sql" and plan_event["plan"][0]["agent"] == "sql"


# ----------------------------------------------------------------- history trimming
async def test_long_histories_are_folded_into_a_summary_keeping_the_last_messages():
    registry = FakeRegistry()
    sup, _, summarise = make(
        registry,
        Planner(decision("sql", step("s1", "sql"))),
        history_max_messages=6,
        history_keep_last=2,
    )
    for i in range(4):  # 8 messages > 6
        await sup.ask("t", "u", f"question {i}")
    history = await sup.history("t")
    assert history["summary"] == "They asked about X; we found Y."
    assert len(history["messages"]) <= 6 and history["messages"][-1][1] == "sql answer"
    assert "question 0" in summarise.seen[0][1]["content"]  # the oldest was folded in


async def test_a_failing_summariser_keeps_the_history_instead_of_failing_the_turn():
    registry = FakeRegistry()
    sup, _, summarise = make(
        registry,
        Planner(decision("sql", step("s1", "sql"))),
        history_max_messages=2,
        history_keep_last=1,
    )
    summarise.reply = RuntimeError("model down")
    turn = await sup.ask("t", "u", "q")
    assert turn.status == "answered" and len((await sup.history("t"))["messages"]) == 2


async def test_short_histories_are_left_alone():
    sup, _, summarise = make(
        FakeRegistry(), Planner(decision("sql", step("s1", "sql")))
    )
    await sup.ask("t", "u", "q")
    assert summarise.seen == [] and (await sup.history("t"))["summary"] == ""
