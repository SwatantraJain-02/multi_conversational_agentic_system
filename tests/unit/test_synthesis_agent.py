"""The Synthesis agent graph, with a fake A2A registry in place of the real Web agent.

The key behaviour: with no usable external result it ALWAYS delegates to the Web agent
(deterministic), with its own task id as the parent; with one it never does; and a failed
delegation degrades to a partial answer instead of failing the task.
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from multi_agenti_conversation_system.a2a.contracts import (
    AgentError,
    ContractError,
    SynthesisData,
    TaskRequest,
    TaskResult,
)
from multi_agenti_conversation_system.agents.synthesis.context import SynthesisContext
from multi_agenti_conversation_system.agents.synthesis.graph import (
    build_graph,
    build_input,
    to_result,
)

CITES = [
    {"n": 1, "url": "https://a.com/1", "title": "A", "published_date": None},
    {"n": 2, "url": "https://b.com/2", "title": "B", "published_date": None},
]


def internal(**kw):
    base = {
        "status": "completed",
        "agent": "sql",
        "answer": "Beverages revenue was 103k in 2025.",
        "sql": "SELECT 1",
        "data": {
            "columns": ["y", "revenue"],
            "rows": [[2025, 103000.5]],
            "row_count": 1,
        },
        "prompt_versions": {"sql/generate": "sql/generate@v1"},
    }
    return TaskResult(**{**base, **kw})


def external(**kw):
    base = {
        "status": "completed",
        "agent": "web",
        "answer": "The global beverage market grew 5% in 2025.",
        "citations": CITES,
        "data": {"confidence": "high"},
        "prompt_versions": {"web/summarise": "web/summarise@v1"},
    }
    return TaskResult(**{**base, **kw})


class FakeRegistry:
    def __init__(self, reply=None):
        self.reply, self.calls = reply, []

    async def call(self, agent, request, **kw):
        self.calls.append((agent, request))
        return self.reply or external()


class Model:
    def __init__(
        self,
        compose="Our revenue is 103k (internal data) vs market +5% [1]. See also [9].",
    ):
        self.compose_text, self.seen = compose, []

    async def derive(self, messages):
        self.seen.append(("derive", messages))
        return "global beverage market trend 2025"

    async def compose(self, messages):
        self.seen.append(("compose", messages))
        return self.compose_text


def make(registry, model=None):
    model = model or Model()
    ctx = SynthesisContext(
        registry=registry, derive_question=model.derive, compose=model.compose
    )
    return build_graph(InMemorySaver()), ctx, model


def request(question="Compare our beverage revenue with the market", **data):
    return TaskRequest(
        sender="supervisor",
        skill="synthesise_findings",
        task=question,
        data=SynthesisData(**data).model_dump(mode="json"),
    )


async def run(graph, ctx, req, thread="t1"):
    return await graph.ainvoke(
        build_input(req), {"configurable": {"thread_id": thread}}, context=ctx
    )


# ------------------------------------------------------------------- both inputs given
async def test_with_both_results_it_does_not_delegate():
    registry = FakeRegistry()
    graph, ctx, model = make(registry)
    req = request(internal=internal(), external=external())
    state = await run(graph, ctx, req)
    out = to_result(state, req)
    assert registry.calls == []  # the trigger is "no external data", not "always"
    assert out.ok and out.data["delegated"] is False and out.data["partial"] is False
    assert out.sql == "SELECT 1"
    assert [c["n"] for c in out.citations] == [1]  # [9] had no source and was removed
    assert "[9]" not in out.answer and "[1]" in out.answer
    assert [kind for kind, _ in model.seen] == ["compose"]


async def test_the_prompt_carries_both_sides_and_marks_external_text_untrusted():
    graph, ctx, model = make(FakeRegistry())
    await run(graph, ctx, request(internal=internal(), external=external()))
    system, user = (m["content"] for m in model.seen[0][1])
    assert "untrusted" in system and "(internal data)" in system
    assert (
        "SELECT 1" in user and "103000.5" in user and "[1] A (https://a.com/1)" in user
    )


async def test_prompt_versions_of_all_three_agents_are_reported():
    graph, ctx, _ = make(FakeRegistry())
    req = request(internal=internal(), external=external())
    out = to_result(await run(graph, ctx, req), req)
    assert set(out.prompt_versions) == {
        "sql/generate",
        "web/summarise",
        "synthesis/compare",
    }


# ------------------------------------------------------------------------- delegation
async def test_with_only_internal_it_delegates_to_the_web_agent_itself():
    registry = FakeRegistry(external())
    graph, ctx, model = make(registry)
    req = request(internal=internal())
    out = to_result(await run(graph, ctx, req), req)

    assert [(a, r.skill) for a, r in registry.calls] == [("web", "answer_web_question")]
    delegated = registry.calls[0][1]
    assert delegated.sender == "synthesis"
    assert delegated.parent_task_id == req.task_id  # the delegation edge
    assert (
        delegated.task == "global beverage market trend 2025"
    )  # derived, external side only
    assert out.ok and out.data["delegated"] is True and out.data["partial"] is False
    assert [c["n"] for c in out.citations] == [1] and out.sql == "SELECT 1"
    assert [kind for kind, _ in model.seen] == ["derive", "compose"]


async def test_a_given_external_question_is_used_as_is():
    registry = FakeRegistry()
    graph, ctx, model = make(registry)
    await run(
        graph, ctx, request(internal=internal(), external_question="EU soda tax 2025")
    )
    assert registry.calls[0][1].task == "EU soda tax 2025"
    assert [kind for kind, _ in model.seen] == ["compose"]  # nothing to derive


async def test_a_failed_external_result_counts_as_missing_and_is_re_fetched():
    failed = TaskResult.failure("web", "timeout", "no answer within 60s")
    registry = FakeRegistry()
    graph, ctx, _ = make(registry)
    out = to_result(
        await run(graph, ctx, request(internal=internal(), external=failed)), request()
    )
    assert (
        len(registry.calls) == 1
        and out.data["delegated"] is True
        and out.data["partial"] is False
    )


async def test_an_unavailable_web_agent_gives_a_partial_answer_that_names_the_gap():
    down = TaskResult.failure("web", "unavailable", "Network communication error")
    registry = FakeRegistry(down)
    graph, ctx, model = make(
        registry, Model("Our beverage revenue was 103k (internal data).")
    )
    req = request(internal=internal())
    out = to_result(await run(graph, ctx, req), req)
    assert (
        out.ok and out.data["partial"] is True and out.data["missing"] == ["external"]
    )
    assert "unavailable" in out.data["delegation_error"]
    assert out.answer.startswith("Note: external market data")
    assert "unavailable" in out.answer and out.citations == [] and out.sql == "SELECT 1"
    assert "GAPS: external market data" in model.seen[-1][1][1]["content"]


async def test_missing_internal_data_is_a_partial_answer_from_external_only():
    graph, ctx, _ = make(FakeRegistry())
    req = request(external=external())
    out = to_result(await run(graph, ctx, req), req)
    assert out.ok and out.data["missing"] == ["internal"] and out.sql is None
    assert out.answer.startswith("Note: internal data (not provided)")


async def test_nothing_usable_at_all_fails_as_unavailable():
    down = TaskResult.failure("web", "unavailable", "down")
    graph, ctx, _ = make(FakeRegistry(down))
    with pytest.raises(AgentError) as caught:
        await run(graph, ctx, request(external_question="x"))
    assert caught.value.code == "unavailable"


# --------------------------------------------------------------------------- the input
def test_an_empty_request_is_rejected_at_the_boundary():
    with pytest.raises(ContractError, match="needs"):
        build_input(request())


def test_bad_data_is_a_contract_error_not_a_crash():
    req = TaskRequest(
        sender="s",
        skill="synthesise_findings",
        task="q",
        data={"internal": {"status": "weird"}},
    )
    with pytest.raises(ContractError, match="invalid SynthesisData"):
        build_input(req)


async def test_a_second_task_in_the_same_context_starts_clean():
    registry = FakeRegistry()
    graph, ctx, _ = make(registry)
    await run(graph, ctx, request(internal=internal()), thread="same")
    req = request(internal=internal(), external=external())
    state = await run(graph, ctx, req, thread="same")
    assert state["delegated"] is False and state["delegation_error"] == ""
    assert len(registry.calls) == 1  # only the first task delegated


async def test_pii_rows_stay_out_of_the_trace(monkeypatch):
    from contextlib import contextmanager

    from multi_agenti_conversation_system import tracing

    seen = {}

    @contextmanager
    def fake(name, **kw):
        seen[name] = kw.get("input")
        yield None

    monkeypatch.setattr(tracing, "observation", fake)
    pii = internal(
        data={
            "columns": ["contact_name"],
            "rows": [["Maria Anders"]],
            "pii_columns": ["customers.contact_name"],
        }
    )
    graph, ctx, _ = make(FakeRegistry())
    await run(graph, ctx, request(internal=pii, external=external()))
    assert seen["compose"] == tracing.OMITTED and "Maria" not in str(seen)
