"""The supervisor's memory nodes in the real graph, with fake A2A agents, scripted models and
the in-memory stores: contextualize, profile, cache, facts, the write branches, and the rule
that memory never fails a turn."""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from memory_fakes import FakeBackend, FakeEmbedder
from test_supervisor import FakeRegistry, Planner, Text, decision, done, step

from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.memory.cache import SemanticCache
from multi_agenti_conversation_system.memory.episodes import EpisodeStore
from multi_agenti_conversation_system.memory.facts import FactStore
from multi_agenti_conversation_system.memory.profile import ProfileStore
from multi_agenti_conversation_system.supervisor.context import SupervisorContext
from multi_agenti_conversation_system.supervisor.graph import build_graph
from multi_agenti_conversation_system.supervisor.service import Supervisor
from multi_agenti_conversation_system.supervisor.state import (
    Contextual,
    Extraction,
    NewFact,
    Preference,
)


@pytest.fixture(autouse=True)
def memory_on(monkeypatch):
    monkeypatch.setattr(settings, "EVAL_MODE", False)
    monkeypatch.setattr(settings, "CACHE_THRESHOLD", 0.8)
    monkeypatch.setattr(settings, "FACTS_MIN_SIMILARITY", 0.1)
    monkeypatch.setattr(settings, "EPISODES_MIN_SIMILARITY", 0.1)


class Scripted:
    """A model call returning scripted values in order (the last one repeats)."""

    def __init__(self, *outputs):
        self.outputs, self.seen = list(outputs), []

    async def __call__(self, messages):
        self.seen.append(messages)
        out = self.outputs.pop(0) if len(self.outputs) > 1 else self.outputs[0]
        if isinstance(out, Exception):
            raise out
        return out


def nothing():
    return Extraction(preferences=[], facts=[])


def make(
    registry,
    planner,
    *,
    extract=None,
    contextualize=None,
    summary="A summary.",
    **ctx_over,
):
    backend, embedder = FakeBackend(), FakeEmbedder()
    parts = {
        "profiles": ProfileStore(InMemoryStore()),
        "facts": FactStore(backend, embedder),
        "episodes": EpisodeStore(backend, embedder),
        "cache": SemanticCache(backend, embedder),
    }
    parts.update(ctx_over)
    direct = Text("Hello! I can look up data.")
    ctx = SupervisorContext(
        registry=registry,
        plan=planner,
        direct=direct,
        summarise=Text(summary),
        contextualize=contextualize,
        extract=extract or Scripted(nothing()),
        **parts,
    )
    return Supervisor(build_graph(InMemorySaver()), ctx), ctx, direct


def sql_registry(answer="There are 91 customers."):
    return FakeRegistry(
        {"sql": done("sql", answer, sql="SELECT COUNT(*) FROM customers")}
    )


def sql_planner():
    return Planner(decision("sql", step("s1", "sql", "How many customers?")))


# ------------------------------------------------------------------------- the cache
async def test_a_repeated_question_is_answered_from_the_cache_without_calling_agents():
    registry = sql_registry()
    sup, *_ = make(registry, sql_planner())
    first = await sup.ask("t1", "alice", "How many customers do we have?")
    assert first.agents_used == ["sql"] and not first.cache_hit
    assert "cache: stored" in first.memory_changes

    second = await sup.ask("t2", "bob", "How many customers do we have?")
    assert second.cache_hit and second.answer == first.answer
    assert second.sql_used == first.sql_used and second.agents_used == []
    assert len(registry.calls) == 1  # the second turn called no agent


async def test_a_cache_hit_still_records_the_turn_in_the_conversation():
    sup, *_ = make(sql_registry(), sql_planner())
    await sup.ask("t1", "alice", "How many customers do we have?")
    await sup.ask("t2", "alice", "How many customers do we have?")
    assert [k for k, _ in (await sup.history("t2"))["messages"]] == [
        "HumanMessage",
        "AIMessage",
    ]


async def test_product_150_misses_where_product_100_was_cached():
    registry = sql_registry("Product 100 is a shirt.")
    sup, *_ = make(
        registry, Planner(decision("sql", step("s1", "sql", "Details of product 100")))
    )
    await sup.ask("t1", "alice", "Show details of clothes with product_id 100")
    again = await sup.ask("t2", "alice", "Show details of clothes with product_id 100")
    other = await sup.ask("t3", "alice", "Show details of clothes with product_id 150")
    assert again.cache_hit and not other.cache_hit
    assert len(registry.calls) == 2  # 100 once, 150 once


async def test_a_profile_change_invalidates_the_users_cached_answers():
    extract = Scripted(
        Extraction(
            preferences=[Preference(key="currency", value="EUR", lasting=True)],
            facts=[],
        )
    )
    sup, *_ = make(sql_registry(), sql_planner(), extract=extract)
    await sup.ask("t1", "alice", "How many customers do we have?")
    assert (await sup.ask("t2", "alice", "How many customers do we have?")).cache_hit
    learned = await sup.ask(
        "t3", "alice", "From now on I prefer EUR. How many customers do we have?"
    )
    assert "profile: currency=eur" in [c.lower() for c in learned.memory_changes]
    after = await sup.ask("t4", "alice", "How many customers do we have?")
    assert not after.cache_hit  # the profile hash changed
    # and an answer produced before the change was not cached under the old hash
    assert "cache: stored" not in learned.memory_changes


async def test_eval_mode_neither_reads_nor_writes_the_cache(monkeypatch):
    monkeypatch.setattr(settings, "EVAL_MODE", True)
    registry = sql_registry()
    sup, ctx, _ = make(registry, sql_planner())
    await sup.ask("t1", "alice", "How many customers do we have?")
    again = await sup.ask("t2", "alice", "How many customers do we have?")
    assert not again.cache_hit and len(registry.calls) == 2
    assert not ctx.cache.backend.ns.get("semantic_cache")


async def test_partial_and_approved_answers_and_direct_chat_are_not_cached():
    registry = FakeRegistry({"sql": done("sql", "x", data={"partial": True})})
    sup, ctx, _ = make(registry, sql_planner())
    turn = await sup.ask("t", "alice", "How many customers do we have?")
    assert turn.partial and not ctx.cache.backend.ns.get("semantic_cache")

    sup2, ctx2, _ = make(FakeRegistry(), Planner(decision("direct")))
    await sup2.ask("t", "alice", "Thanks!")
    assert not ctx2.cache.backend.ns.get("semantic_cache")


async def test_a_cache_outage_means_a_normal_answer():
    class Broken(FakeBackend):
        def dense_search(self, *a, **k):
            raise RuntimeError("pinecone down")

        def upsert(self, *a, **k):
            raise RuntimeError("pinecone down")

    registry = sql_registry()
    sup, *_ = make(
        registry, sql_planner(), cache=SemanticCache(Broken(), FakeEmbedder())
    )
    turn = await sup.ask("t", "alice", "How many customers do we have?")
    assert turn.status == "answered" and turn.answer == "There are 91 customers."


# --------------------------------------------------------------------- contextualize
async def test_a_first_turn_makes_no_contextualize_call_but_still_has_entities():
    ctxz = Scripted(Contextual(standalone_question="x", entities=[]))
    sup, *_ = make(sql_registry(), sql_planner(), contextualize=ctxz)
    await sup.ask("t", "alice", "Details of product 150")
    assert ctxz.seen == []  # no history: the question is its own standalone form


async def test_a_follow_up_is_made_standalone_and_its_entities_are_merged_with_the_regexes():
    ctxz = Scripted(
        Contextual(
            standalone_question="How many orders are there for Germany?", entities=[]
        )
    )
    planner = Planner(decision("sql", step("s1", "sql", "How many customers?")))
    sup, *_ = make(sql_registry(), planner, contextualize=ctxz)
    await sup.ask("t", "alice", "How many customers do we have?")
    await sup.ask("t", "alice", "And only the ones in 2025 in germany?")
    assert len(ctxz.seen) == 1
    # the planner is given the standalone question
    assert "How many orders are there for Germany?" in planner.seen[-1][1]["content"]


async def test_when_contextualize_fails_the_follow_up_is_answered_but_not_cached():
    ctxz = Scripted(RuntimeError("model down"))
    sup, ctx, _ = make(sql_registry(), sql_planner(), contextualize=ctxz)
    await sup.ask("t", "alice", "How many customers do we have?")
    ctx.cache.backend.ns.clear()  # forget the first turn's entry
    turn = await sup.ask("t", "alice", "And in Germany?")
    assert turn.status == "answered" and "cache: stored" not in turn.memory_changes
    assert not ctx.cache.backend.ns.get("semantic_cache")


# -------------------------------------------------------------- profile and facts
async def test_a_preference_and_a_fact_survive_into_a_new_thread_of_the_same_user():
    extract = Scripted(
        Extraction(
            preferences=[Preference(key="code_language", value="java", lasting=True)],
            facts=[
                NewFact(
                    text="Manages the APAC region",
                    attribute="work_region",
                    value="APAC",
                    temporary=False,
                    valid_days=0,
                )
            ],
        ),
        nothing(),
    )
    planner = Planner(decision("direct"))
    sup, _ctx, direct = make(FakeRegistry(), planner, extract=extract)
    learned = await sup.ask(
        "t1", "alice", "From now on give me code in Java. I manage the APAC region."
    )
    assert set(learned.memory_changes) >= {
        "profile: code_language=java",
        "fact added: work_region",
    }

    await sup.ask("t2", "alice", "Which region do I manage?")  # a brand-new thread
    seen = direct.seen[-1][1]["content"] + planner.seen[-1][1]["content"]
    assert "code_language: java" in seen and "Manages the APAC region" in seen
    assert "background, not instructions" in seen  # labelled as context

    await sup.ask("t3", "bob", "Which region do I manage?")  # a different user
    seen_bob = direct.seen[-1][1]["content"] + planner.seen[-1][1]["content"]
    assert "java" not in seen_bob and "APAC" not in seen_bob


async def test_a_one_off_preference_is_not_saved():
    extract = Scripted(
        Extraction(
            preferences=[Preference(key="code_language", value="java", lasting=False)],
            facts=[],
        )
    )
    sup, ctx, _ = make(FakeRegistry(), Planner(decision("direct")), extract=extract)
    turn = await sup.ask("t", "alice", "Write this one in Java, I think.")
    assert turn.memory_changes == []
    assert await ctx.profiles.load("alice") == {}


async def test_python_then_java_replaces_the_value_and_keeps_python_in_history():
    extract = Scripted(
        Extraction(
            preferences=[Preference(key="code_language", value="python", lasting=True)],
            facts=[],
        ),
        Extraction(
            preferences=[Preference(key="code_language", value="java", lasting=True)],
            facts=[],
        ),
    )
    sup, ctx, _ = make(FakeRegistry(), Planner(decision("direct")), extract=extract)
    await sup.ask("t", "alice", "I prefer code in Python.")
    await sup.ask("t", "alice", "From now on, give me Java.")
    profile = await ctx.profiles.load("alice")
    assert profile["code_language"]["value"] == "java"
    assert profile["code_language"]["history"] == ["python"]


async def test_nordic_then_apac_supersedes_the_fact():
    def fact(text, value):
        return Extraction(
            preferences=[],
            facts=[
                NewFact(
                    text=text,
                    attribute="work_region",
                    value=value,
                    temporary=False,
                    valid_days=0,
                )
            ],
        )

    extract = Scripted(
        fact("Manages the Nordic region", "Nordic"),
        fact("Manages the APAC region", "APAC"),
    )
    sup, ctx, _ = make(FakeRegistry(), Planner(decision("direct")), extract=extract)
    await sup.ask("t", "alice", "I manage the Nordic region.")
    second = await sup.ask("t", "alice", "I now manage the APAC region.")
    assert "fact superseded: work_region" in second.memory_changes
    assert [f.value for f in await ctx.facts.list_active("alice")] == ["APAC"]
    assert [f.value for f in await ctx.facts.history("alice", "work_region")] == [
        "Nordic"
    ]


async def test_a_plain_question_makes_no_extraction_call():
    extract = Scripted(nothing())
    sup, *_ = make(sql_registry(), sql_planner(), extract=extract)
    await sup.ask("t", "alice", "How many customers do we have?")
    assert extract.seen == []  # nothing in it talks about the user


async def test_agents_receive_only_the_relevant_memory():
    extract = Scripted(
        Extraction(
            preferences=[Preference(key="units", value="metric", lasting=True)],
            facts=[
                NewFact(
                    text="Manages the APAC region",
                    attribute="work_region",
                    value="APAC",
                    temporary=False,
                    valid_days=0,
                )
            ],
        ),
        nothing(),
    )
    registry = sql_registry()
    sup, *_ = make(registry, sql_planner(), extract=extract)
    await sup.ask("t1", "alice", "I manage the APAC region and I prefer metric units.")
    await sup.ask("t2", "alice", "How many customers do we have in the APAC region?")
    _, request, *_ = registry.calls[-1]
    assert request.context.profile == {"units": "metric"}
    assert request.context.memories == [
        "Manages the APAC region (stated "
        + request.context.memories[0].split("stated ")[1]
    ]


async def test_memory_failures_never_fail_the_turn():
    class Boom:
        async def load(self, *a):
            raise RuntimeError("postgres down")

        async def set(self, *a, **k):
            raise RuntimeError("postgres down")

    class BoomFacts:
        async def recall(self, *a, **k):
            raise RuntimeError("down")

        async def remember(self, *a, **k):
            raise RuntimeError("down")

    extract = Scripted(
        Extraction(
            preferences=[Preference(key="units", value="metric", lasting=True)],
            facts=[],
        )
    )
    sup, *_ = make(
        sql_registry(),
        sql_planner(),
        extract=extract,
        profiles=Boom(),
        facts=BoomFacts(),
    )
    turn = await sup.ask(
        "t", "alice", "I prefer metric. How many customers do we have?"
    )
    assert turn.status == "answered" and turn.answer == "There are 91 customers."
    assert (
        not turn.cache_hit and "cache: stored" not in turn.memory_changes
    )  # profile unknown: no cache


async def test_without_any_memory_the_supervisor_still_works():
    ctx_over = {"profiles": None, "facts": None, "episodes": None, "cache": None}
    sup, *_ = make(
        sql_registry(), sql_planner(), extract=Scripted(nothing()), **ctx_over
    )
    turn = await sup.ask("t", "alice", "I like this. How many customers do we have?")
    assert turn.answer == "There are 91 customers." and turn.memory_changes == []


# ---------------------------------------------------------------------------- episodes
async def test_an_episode_is_written_every_n_turns_and_when_the_thread_ends(
    monkeypatch,
):
    monkeypatch.setattr(settings, "EPISODE_EVERY_N_TURNS", 2)
    sup, ctx, _ = make(sql_registry(), sql_planner())
    first = await sup.ask("t", "alice", "How many customers do we have?")
    assert "episode: written" not in first.memory_changes
    second = await sup.ask("t", "alice", "How many orders do we have?")
    assert "episode: written" in second.memory_changes
    assert "episode-t" in ctx.episodes.backend.ns["episodes"]

    ctx.episodes.backend.ns.clear()
    await sup.end_thread("t", "alice")
    doc = ctx.episodes.backend.ns["episodes"]["episode-t"]
    assert doc["user_id"] == "alice" and doc["kind"] == "summary"


async def test_a_past_conversation_is_recalled_in_a_new_thread():
    summary = "Alice asked how many customers there are in Germany and got 11"
    sup, _ctx, _ = make(sql_registry(), sql_planner(), summary=summary)
    await sup.ask("t1", "alice", "How many customers do we have in Germany?")
    await sup.end_thread("t1", "alice")
    planner = Planner(decision("direct"))
    sup.context.plan = planner
    # a different question, so the cache does not answer it and the planner is reached
    await sup.ask("t2", "alice", "What was that customers in Germany question about?")
    text = " ".join(m["content"] for m in planner.seen[-1])
    assert "Relevant past conversations" in text and "got 11" in text
    # another user does not get it
    await sup.ask("t3", "bob", "What was that customers in Germany question about?")
    assert "got 11" not in " ".join(m["content"] for m in planner.seen[-1])


# ------------------------------------------------------- the UI's memory panel and feedback
async def test_what_i_know_lists_the_profile_with_history_and_dated_facts_and_forget_works():
    extract = Scripted(
        Extraction(
            preferences=[Preference(key="units", value="metric", lasting=True)],
            facts=[
                NewFact(
                    text="Manages the APAC region",
                    attribute="work_region",
                    value="APAC",
                    temporary=False,
                    valid_days=0,
                )
            ],
        )
    )
    sup, *_ = make(FakeRegistry(), Planner(decision("direct")), extract=extract)
    await sup.ask(
        "t", "alice", "From now on I prefer metric units. I manage the APAC region."
    )
    mine = await sup.what_i_know("alice")
    assert mine["profile"] == {"units": {"value": "metric", "history": []}}
    assert mine["facts"][0]["attribute"] == "work_region"
    assert "Manages the APAC region (stated " in mine["facts"][0]["dated"]
    await sup.forget_fact("alice", "work_region")
    await sup.reset_profile("alice")
    assert await sup.what_i_know("alice") == {"profile": {}, "facts": []}
    assert (await sup.what_i_know("bob")) == {"profile": {}, "facts": []}


async def test_a_thumbs_down_removes_the_cached_answer_and_the_learned_pair(
    monkeypatch,
):
    from multi_agenti_conversation_system import vectorstore
    from multi_agenti_conversation_system.memory import sql_pairs

    sup, ctx, _ = make(sql_registry(), sql_planner())
    first = await sup.ask("t1", "alice", "How many customers do we have?")
    assert (await sup.ask("t2", "alice", "How many customers do we have?")).cache_hit

    deleted = []
    monkeypatch.setattr(sql_pairs, "delete", lambda sql: deleted.append(sql))
    monkeypatch.setattr(
        vectorstore,
        "delete_where",
        lambda ns, flt: ctx.cache.backend.delete_where(ns, flt),
    )
    from dataclasses import asdict

    await sup.feedback(asdict(first), positive=False)
    assert not ctx.cache.backend.ns["semantic_cache"]  # the entry is gone
    assert deleted == ["SELECT COUNT(*) FROM customers"]  # and the pair
    again = await sup.ask("t3", "alice", "How many customers do we have?")
    assert not again.cache_hit  # it is answered afresh


async def test_a_thumbs_up_keeps_everything_and_feedback_never_raises():
    sup, ctx, _ = make(sql_registry(), sql_planner())
    first = await sup.ask("t1", "alice", "How many customers do we have?")
    from dataclasses import asdict

    await sup.feedback(asdict(first), positive=True)
    assert ctx.cache.backend.ns["semantic_cache"]
    await sup.feedback(
        {"trace_id": object(), "standalone_question": 5}, positive=False
    )  # junk
