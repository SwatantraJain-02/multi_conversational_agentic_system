"""The memory stores without Pinecone or the network: entity keys, filters, the profile,
facts (re-statement, contradiction, expiry, per-user isolation), the semantic cache
(the product 100 vs 150 problem) and episodes."""

import pytest
from langgraph.store.memory import InMemoryStore
from memory_fakes import FakeBackend, FakeEmbedder

from multi_agenti_conversation_system import vectorstore
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.memory import cache as cache_mod
from multi_agenti_conversation_system.memory import profile as profile_mod
from multi_agenti_conversation_system.memory.cache import SemanticCache, lookup_filter
from multi_agenti_conversation_system.memory.entities import entity_key
from multi_agenti_conversation_system.memory.episodes import EpisodeStore
from multi_agenti_conversation_system.memory.facts import DAY, FactStore, fact_id
from multi_agenti_conversation_system.memory.profile import ProfileStore

NOW = 1_800_000_000  # a fixed "today"


# ------------------------------------------------------------------------ entity key
def test_entity_key_separates_two_ids_and_ignores_wording():
    a = entity_key("Show details of clothes with product_id 100")
    b = entity_key("Show details of clothes with product_id 150")
    assert a == "100" and b == "150" and a != b
    assert entity_key("What are the details for product 100?") == a


def test_entity_key_finds_years_names_quotes_and_merges_model_entities():
    assert entity_key("How many orders did Germany place in 2025?") == "2025|germany"
    assert entity_key("Revenue for 'Chai' in Q3 2026") == "2026|chai|q3"
    # a name typed in lowercase has no capital letter: only the model's list has it
    assert entity_key("orders from germany", ["Germany"]) == "germany"
    assert entity_key("1,000 units") == entity_key("1000 units") == "1000"


def test_entity_key_is_a_string_when_there_are_no_entities():
    assert entity_key("How many customers do we have?") == ""
    assert entity_key("") == ""


def test_a_number_the_model_dropped_still_changes_the_key():
    # the model lists no entities at all; the regexes alone separate 100 from 150
    assert entity_key("product 100", []) != entity_key("product 150", [])


# ----------------------------------------------------------------------- the filters
def test_build_filter_never_drops_a_clause():
    flt = vectorstore.build_filter(
        eq={"entity_key": "", "profile_hash": "abc"}, gt={"expires_at": 5}
    )
    assert flt == {
        "entity_key": {"$eq": ""},
        "profile_hash": {"$eq": "abc"},
        "expires_at": {"$gt": 5},
    }
    with pytest.raises(ValueError):
        vectorstore.build_filter(eq={"entity_key": None})
    # the old helper silently drops it: exactly the bug build_filter exists to prevent
    assert vectorstore.eq_filter(entity_key=None, profile_hash="a") == {
        "profile_hash": {"$eq": "a"}
    }


def test_a_cache_lookup_filter_always_has_every_key():
    flt = lookup_filter("", "h", "pv", now=NOW)
    assert set(flt) == {
        "entity_key",
        "profile_hash",
        "prompt_version",
        "data_version",
        "expires_at",
    }
    assert flt["entity_key"] == {"$eq": ""} and flt["expires_at"] == {"$gt": NOW}


# ------------------------------------------------------------------------- profile
def test_profile_replaces_by_key_and_keeps_history():
    p, c1 = profile_mod.apply({}, "code_language", "Python", now=1)
    p, c2 = profile_mod.apply(p, "Code Language", "Java", now=2)
    assert c1["old"] == "" and c2 == {
        "key": "code_language",
        "old": "Python",
        "new": "Java",
    }
    assert p["code_language"]["value"] == "Java" and p["code_language"]["history"] == [
        "Python"
    ]
    same, change = profile_mod.apply(p, "code_language", "java")
    assert change is None and same == p  # a re-statement changes nothing


def test_profile_history_keeps_the_last_five():
    p = {}
    for i in range(9):
        p, _ = profile_mod.apply(p, "units", f"v{i}")
    assert p["units"]["history"] == ["v3", "v4", "v5", "v6", "v7"]


def test_profile_hash_tracks_values_only_and_exists_when_empty():
    empty = profile_mod.profile_hash({})
    assert empty and empty == profile_mod.profile_hash({})
    p1, _ = profile_mod.apply({}, "currency", "EUR", now=1)
    p2, _ = profile_mod.apply({}, "currency", "eur", now=999)  # other time and case
    assert profile_mod.profile_hash(p1) == profile_mod.profile_hash(p2) != empty
    p3, _ = profile_mod.apply(p1, "currency", "USD")
    assert profile_mod.profile_hash(p3) != profile_mod.profile_hash(p1)


async def test_profile_store_round_trip_per_user():
    store = ProfileStore(InMemoryStore())
    await store.set("alice", "code_language", "python")
    await store.set("alice", "code_language", "java")
    assert profile_mod.values(await store.load("alice")) == {"code_language": "java"}
    assert await store.load("bob") == {}
    await store.reset("alice")
    assert await store.load("alice") == {}


# --------------------------------------------------------------------------- facts
@pytest.fixture
def facts():
    return FactStore(FakeBackend(), FakeEmbedder())


async def test_a_new_fact_is_recalled_dated(facts):
    out = await facts.remember(
        "alice", "Manages the APAC region", "work_region", "APAC", now=NOW
    )
    assert out == "added"
    found = await facts.recall(
        "alice", "Which region does the user manage?", min_similarity=0.1, now=NOW
    )
    assert [f.text for f in found] == ["Manages the APAC region"]
    assert "stated 2027-" in found[0].dated(NOW) or "stated 20" in found[0].dated(NOW)


async def test_the_same_fact_again_is_a_restatement_not_a_second_fact(facts):
    await facts.remember("alice", "Manages APAC", "work_region", "APAC", now=NOW)
    out = await facts.remember(
        "alice", "Manages APAC", "work_region", "apac", now=NOW + DAY
    )
    assert out == "restated"
    active = await facts.list_active("alice", now=NOW + DAY)
    assert len(active) == 1 and active[0].last_confirmed_at == NOW + DAY


async def test_a_contradiction_supersedes_and_keeps_history(facts):
    await facts.remember(
        "alice", "Manages the Nordic region", "work_region", "Nordic", now=NOW
    )
    out = await facts.remember(
        "alice", "Manages the APAC region", "work_region", "APAC", now=NOW + DAY
    )
    assert out == "superseded"
    assert [f.value for f in await facts.list_active("alice", now=NOW + DAY)] == [
        "APAC"
    ]
    old = await facts.history("alice", "work_region")
    assert [f.value for f in old] == ["Nordic"]  # nothing was deleted
    # retrieval never returns the superseded one, however similar it is to the question
    found = await facts.recall(
        "alice", "Manages the Nordic region", min_similarity=0.0, now=NOW + DAY
    )
    assert [f.value for f in found] == ["APAC"]


async def test_nordic_to_apac_is_a_contradiction_despite_near_identical_text(facts):
    # similarity alone would call these duplicates: the attribute key does not
    emb = FakeEmbedder()
    a, b = (
        emb.embed_query("manages the nordic region"),
        emb.embed_query("manages the apac region"),
    )
    assert sum(x * y for x, y in zip(a, b, strict=True)) > 0.6
    await facts.remember(
        "u", "Manages the Nordic region", "work_region", "Nordic", now=NOW
    )
    assert (
        await facts.remember(
            "u", "Manages the APAC region", "work_region", "APAC", now=NOW
        )
        == "superseded"
    )


async def test_a_temporary_fact_disappears_after_valid_until(facts):
    await facts.remember(
        "alice",
        "Preparing the Q3 review",
        "current_project",
        "Q3 review",
        kind="temporary",
        valid_days=30,
        now=NOW,
    )
    q = "What is the user preparing?"
    assert await facts.recall("alice", q, min_similarity=0.1, now=NOW + 29 * DAY)
    assert not await facts.recall("alice", q, min_similarity=0.1, now=NOW + 31 * DAY)
    assert not await facts.list_active("alice", now=NOW + 31 * DAY)


async def test_a_temporary_fact_without_a_date_lasts_thirty_days(facts):
    await facts.remember(
        "a", "In Berlin this week", "location", "Berlin", kind="temporary", now=NOW
    )
    assert await facts.list_active("a", now=NOW + 29 * DAY)
    assert not await facts.list_active("a", now=NOW + 31 * DAY)


async def test_a_different_user_never_sees_the_fact(facts):
    await facts.remember(
        "alice", "Manages the APAC region", "work_region", "APAC", now=NOW
    )
    assert not await facts.recall(
        "bob", "Manages the APAC region", min_similarity=0.0, now=NOW
    )
    assert not await facts.list_active("bob", now=NOW)
    # and both of them can hold the same attribute without touching each other
    await facts.remember("bob", "Manages Nordic", "work_region", "Nordic", now=NOW)
    assert [f.value for f in await facts.list_active("alice", now=NOW)] == ["APAC"]
    assert fact_id("alice", "work_region", "") != fact_id("bob", "work_region", "")


async def test_every_fact_query_is_filtered_by_user_and_status(facts):
    await facts.remember("alice", "Manages APAC", "work_region", "APAC", now=NOW)
    await facts.recall("alice", "APAC", min_similarity=0.0, now=NOW)
    flt = facts.backend.filters[-1]
    assert flt["user_id"] == {"$eq": "alice"} and flt["status"] == {"$eq": "active"}


async def test_old_unconfirmed_facts_are_tagged_and_rank_lower(facts):
    await facts.remember("a", "Likes tea", "drink", "tea", now=NOW)
    old = (await facts.list_active("a", now=NOW))[0]
    assert "may be outdated" not in old.dated(NOW + 100 * DAY)
    assert "may be outdated" in old.dated(NOW + 200 * DAY)
    from multi_agenti_conversation_system.memory.facts import rank_score

    assert rank_score(0.9, NOW, NOW) > rank_score(0.9, NOW - 400 * DAY, NOW)


async def test_forget_deletes_the_fact_and_its_history(facts):
    await facts.remember("a", "Manages Nordic", "work_region", "Nordic", now=NOW)
    await facts.remember("a", "Manages APAC", "work_region", "APAC", now=NOW)
    await facts.forget("a", "work_region")
    assert not await facts.list_active("a", now=NOW) and not await facts.history(
        "a", "work_region"
    )


async def test_a_recall_failure_is_an_empty_recall():
    class Broken(FakeBackend):
        def dense_search(self, *a, **k):
            raise RuntimeError("pinecone is down")

    assert await FactStore(Broken(), FakeEmbedder()).recall("a", "q", now=NOW) == []


# -------------------------------------------------------------------------- cache
@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(settings, "CACHE_THRESHOLD", 0.8)
    return SemanticCache(FakeBackend(), FakeEmbedder())


async def _store(
    cache, q, key="", profile="p1", version="v1", agents=("sql",), now=NOW, answer="A"
):
    return await cache.store(
        q,
        key,
        profile,
        version,
        answer=answer,
        citations=[],
        sql="SELECT 1",
        agents=list(agents),
        route="sql",
        now=now,
    )


async def test_a_paraphrase_with_the_same_entities_hits(cache):
    await _store(
        cache,
        "Show details of clothes with product_id 100",
        entity_key("product 100"),
        answer="P100",
    )
    hit = await cache.lookup(
        "Show the details of clothes with product_id 100",
        entity_key("product 100"),
        "p1",
        "v1",
        now=NOW,
    )
    assert hit and hit.answer == "P100" and hit.sql == "SELECT 1"


async def test_product_100_does_not_answer_product_150(cache):
    q100 = "Show details of clothes with product_id 100"
    q150 = "Show details of clothes with product_id 150"
    await _store(cache, q100, entity_key(q100), answer="P100")
    # near-identical wording: only the entity key stands between 150 and product 100's answer
    sim = sum(
        a * b
        for a, b in zip(
            FakeEmbedder().embed_query(q100),
            FakeEmbedder().embed_query(q150),
            strict=True,
        )
    )
    assert sim > 0.8
    assert await cache.lookup(q150, entity_key(q150), "p1", "v1", now=NOW) is None


async def test_the_empty_entity_key_does_not_match_a_keyed_entry(cache):
    # the question lost its entity (e.g. extraction failed): "" must not match "150"
    await _store(cache, "Show details of product 150", key="150")
    assert (
        await cache.lookup("Show details of product", "", "p1", "v1", now=NOW) is None
    )
    assert cache.backend.filters[-1]["entity_key"] == {"$eq": ""}


async def test_a_changed_profile_misses(cache):
    await _store(cache, "revenue by year", profile="eur-hash")
    assert await cache.lookup("revenue by year", "", "usd-hash", "v1", now=NOW) is None
    assert await cache.lookup("revenue by year", "", "eur-hash", "v1", now=NOW)


async def test_a_new_prompt_version_misses(cache):
    await _store(cache, "revenue by year", version="v1")
    assert await cache.lookup("revenue by year", "", "p1", "v2", now=NOW) is None


async def test_a_new_data_version_misses(cache, monkeypatch):
    await _store(cache, "revenue by year")
    monkeypatch.setattr(settings, "DATA_VERSION", "2")
    assert await cache.lookup("revenue by year", "", "p1", "v1", now=NOW) is None


async def test_web_answers_expire_after_a_day_database_answers_do_not(cache):
    await _store(cache, "market trends", agents=("web",))
    await _store(cache, "orders per year", agents=("sql",))
    later = NOW + 25 * 3600
    assert await cache.lookup("market trends", "", "p1", "v1", now=NOW + 3600)
    assert await cache.lookup("market trends", "", "p1", "v1", now=later) is None
    assert await cache.lookup("orders per year", "", "p1", "v1", now=later)


async def test_below_the_threshold_is_a_miss(cache):
    await _store(cache, "orders per year in germany")
    assert (
        await cache.lookup("completely unrelated words here", "", "p1", "v1", now=NOW)
        is None
    )


async def test_a_cache_failure_is_a_miss():
    class Broken(FakeBackend):
        def dense_search(self, *a, **k):
            raise RuntimeError("down")

    assert (
        await SemanticCache(Broken(), FakeEmbedder()).lookup("q", "", "p", "v") is None
    )


def test_what_is_never_cached():
    ok = {
        "partial": False,
        "approvals": [],
        "intent": "sql",
        "standalone_ok": True,
        "answer": "x",
    }
    assert cache_mod.cacheable(**ok)
    assert not cache_mod.cacheable(**{**ok, "partial": True})
    assert not cache_mod.cacheable(**{**ok, "approvals": [{"approved": True}]})
    assert not cache_mod.cacheable(**{**ok, "intent": "direct"})
    assert not cache_mod.cacheable(**{**ok, "intent": "cached"})
    assert not cache_mod.cacheable(**{**ok, "standalone_ok": False})
    assert not cache_mod.cacheable(**{**ok, "answer": "  "})


# ------------------------------------------------------------------------ episodes
async def test_episodes_are_per_user_and_exclude_the_current_thread():
    store = EpisodeStore(FakeBackend(), FakeEmbedder())
    await store.write(
        "t1", "alice", "Alice asked about orders in Germany and got 122", now=NOW
    )
    await store.write("t2", "alice", "Alice asked about coffee prices", now=NOW)
    q = "orders in Germany"
    found = await store.recall("alice", q, min_similarity=0.1)
    assert (
        found and found[0].thread_id == "t1" and "conversation on" in found[0].dated()
    )
    assert not await store.recall("bob", q, min_similarity=0.0)
    assert all(
        e.thread_id != "t1"
        for e in await store.recall("alice", q, exclude_thread="t1", min_similarity=0.0)
    )
    await store.write(
        "t1", "alice", "Alice asked about orders in Germany, then France", now=NOW
    )
    assert (
        len(
            [
                e
                for e in await store.recall("alice", q, k=5, min_similarity=0.0)
                if e.thread_id == "t1"
            ]
        )
        == 1
    )
