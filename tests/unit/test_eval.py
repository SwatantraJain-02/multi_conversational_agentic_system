"""The eval harness without a model: the dataset is well formed, the scorers say what they
mean, result sets are compared as sets, and the summary/CSV come out right."""

import csv
from types import SimpleNamespace

from multi_agenti_conversation_system.evals import run_eval
from multi_agenti_conversation_system.evals.dataset import (
    expected_of,
    input_of,
    load_cases,
)
from multi_agenti_conversation_system.evals.judge import (
    CRITERIA,
    Criterion,
    Judge,
    Judgement,
)
from multi_agenti_conversation_system.evals.scorers import rows_match, score_item

CATEGORIES = {"normal", "complex", "hybrid", "ambiguous", "failure", "memory", "cache"}


# ---------------------------------------------------------------------------- dataset
def test_the_dataset_is_well_formed():
    cases = load_cases()
    assert len(cases) >= 39
    assert len({c["id"] for c in cases}) == len(cases)
    assert {c["category"] for c in cases} == CATEGORIES
    for c in cases:
        assert c["question"].strip()
        assert set(input_of(c)) | set(expected_of(c)) == set(c)
        assert not set(input_of(c)) & set(
            expected_of(c)
        )  # nothing is both input and answer key
    assert (
        load_cases("hybrid", limit=2)
        == [c for c in cases if c["category"] == "hybrid"][:2]
    )


def test_every_gold_query_is_a_valid_read_only_select():
    from multi_agenti_conversation_system.guardrails.sql_validator import (
        validate_select,
    )
    from multi_agenti_conversation_system.mcp.servers.db_server.catalog import (
        schema_tables,
    )

    tables = set(schema_tables())
    for c in load_cases():
        if "gold_sql" in c:
            assert validate_select(c["gold_sql"], 1000, tables).ok, c["id"]


def test_the_dataset_covers_the_required_failure_kinds():
    flags = {k for c in load_cases("failure") for k in (c.get("expect") or {})}
    assert {"blocked", "refused", "approval", "no_agents"} <= flags


# -------------------------------------------------------------------------- row match
def test_result_sets_match_regardless_of_order_and_formatting():
    assert rows_match([("USA", 13), ("France", 11)], [("France", 11), ("USA", 13)])
    assert rows_match([(1234.5,)], [(1234.5004,)])  # tolerance
    assert rows_match([("Germany",)], [("germany ",)])  # case and spacing
    assert not rows_match([("USA", 13)], [("USA", 14)])
    assert not rows_match([("USA", 13)], [("USA", 13), ("UK", 7)])  # row count


def test_the_agents_rows_may_carry_extra_columns_but_not_miss_values():
    assert rows_match(
        [(20, "Sir Rodney's", "Confections", 81.0)], [("Sir Rodney's", 81.0)]
    )
    assert not rows_match([(20, "Sir Rodney's")], [("Sir Rodney's", 81.0)])


def test_each_gold_row_needs_its_own_agent_row():
    assert not rows_match([("a", 1), ("a", 1)], [("a", 1), ("b", 2)])


# ---------------------------------------------------------------------------- scorers
def names(scores):
    return {n: v for n, v, _ in scores}


def test_route_and_agents():
    exp = {
        "expected_intent": ["compare", "hybrid"],
        "expected_agents": ["sql", "synthesis"],
    }
    ok = {
        "intent": "hybrid",
        "agents_used": ["sql", "web", "synthesis"],
        "status": "answered",
        "answer": "x",
    }
    assert names(score_item(exp, ok))["route_correct"] == 1.0
    assert names(score_item(exp, ok))["agents_correct"] == 1.0
    bad = {**ok, "intent": "sql", "agents_used": ["sql"]}
    assert names(score_item(exp, bad))["route_correct"] == 0.0
    assert names(score_item(exp, bad))["agents_correct"] == 0.0


def test_a_case_that_must_use_no_agent_fails_if_one_ran():
    exp = {"expected_agents": [], "expect": {"no_agents": True}}
    assert names(score_item(exp, {"agents_used": []}))["agents_correct"] == 1.0
    assert names(score_item(exp, {"agents_used": ["web"]}))["agents_correct"] == 0.0
    assert names(score_item(exp, {"agents_used": ["web"]}))["no_agents_called"] == 0.0


def test_scorers_that_do_not_apply_are_absent():
    got = names(score_item({}, {"status": "answered", "answer": "hi"}))
    assert set(got) == {"answered", "no_secret_leak"}  # nothing else was asked about


def test_the_sql_scorers_use_the_result_set():
    rows = {
        "SELECT a": [("USA", 13)],
        "SELECT g": [("USA", 13)],
        "SELECT bad": [("USA", 1)],
    }

    def execute(sql):
        if sql == "boom":
            raise RuntimeError("syntax error")
        return rows[sql]

    exp = {"gold_sql": "SELECT g"}
    assert (
        names(score_item(exp, {"sql_used": "SELECT a"}, execute))["result_matches_gold"]
        == 1.0
    )
    assert (
        names(score_item(exp, {"sql_used": "SELECT bad"}, execute))[
            "result_matches_gold"
        ]
        == 0.0
    )
    got = names(score_item(exp, {"sql_used": "boom"}, execute))
    assert got["sql_executes"] == 0.0 and got["result_matches_gold"] == 0.0
    assert (
        names(score_item(exp, {"sql_used": None}, execute))["result_matches_gold"]
        == 0.0
    )


def test_required_facts_is_a_fraction():
    exp = {"required_facts": ["nordic", "region"]}
    assert (
        names(score_item(exp, {"answer": "You manage the Nordic area"}))[
            "required_facts"
        ]
        == 0.5
    )
    assert names(score_item(exp, {"answer": "NORDIC region"}))["required_facts"] == 1.0


def test_guardrail_refusal_approval_and_cache_scorers():
    assert (
        names(score_item({"expect": {"blocked": True}}, {"blocked": True}))[
            "guardrail_blocked"
        ]
        == 1.0
    )
    assert (
        names(score_item({"expect": {"blocked": True}}, {"blocked": False}))[
            "guardrail_blocked"
        ]
        == 0.0
    )
    refused = {"results": {"s1": {"data": {"refused": True}}}, "sql_used": None}
    assert names(score_item({"expect": {"refused": True}}, refused))["refused"] == 1.0
    ran_anyway = {**refused, "sql_used": "DELETE FROM x"}
    assert (
        names(score_item({"expect": {"refused": True}}, ran_anyway))["refused"] == 0.0
    )
    assert (
        names(score_item({"expect": {"approval": True}}, {"approval_requested": True}))[
            "approval_requested"
        ]
        == 1.0
    )
    assert (
        names(score_item({"expect": {"cache_hit": False}}, {"cache_hit": True}))[
            "cache_behaviour"
        ]
        == 0.0
    )
    assert (
        names(score_item({"expect": {"cache_hit": True}}, {"cache_hit": True}))[
            "cache_behaviour"
        ]
        == 1.0
    )


def test_citations_secrets_and_latency():
    assert (
        names(score_item({"expect": {"citations": True}}, {"citations": []}))[
            "citations_present"
        ]
        == 0.0
    )
    assert (
        names(score_item({"expect": {"citations": True}}, {"citations": [{"n": 1}]}))[
            "citations_present"
        ]
        == 1.0
    )
    leak = names(score_item({}, {"answer": "key sk-abcdefghijklmnopqrstuvwx"}))
    assert leak["no_secret_leak"] == 0.0
    assert (
        names(score_item({}, {"answer": "ok", "latency_ms": 1234}))["latency_ms"]
        == 1234.0
    )


def test_a2a_schema_valid_checks_every_result():
    good = {
        "status": "completed",
        "agent": "sql",
        "answer": "x",
        "citations": [],
        "data": {},
        "prompt_versions": {},
    }
    assert names(score_item({}, {"results": {"s1": good}}))["a2a_schema_valid"] == 1.0
    assert (
        names(score_item({}, {"results": {"s1": {"status": "nonsense"}}}))[
            "a2a_schema_valid"
        ]
        == 0.0
    )


def test_a_crashed_case_scores_as_not_answered():
    assert (
        names(score_item({}, {"status": "error", "answer": "", "error": "Boom"}))[
            "answered"
        ]
        == 0.0
    )


# ------------------------------------------------------------------------------- judge
async def test_the_judge_returns_five_clamped_scores_and_skips_failures():
    crit = lambda s: Criterion(score=s, reason="because")

    async def call(messages):
        return Judgement(
            relevance=crit(9),
            correctness=crit(0),
            groundedness=crit(4),
            completeness=crit(3),
            clarity=crit(5),
        )

    got = await Judge(call).judge("q", "an answer", [{"n": 1, "title": "A"}])
    assert [n for n, *_ in got] == [f"judge_{c}" for c in CRITERIA]
    assert [v for _, v, _ in got] == [5.0, 1.0, 4.0, 3.0, 5.0]  # clamped to 1..5

    async def broken(messages):
        raise RuntimeError("down")

    assert await Judge(broken).judge("q", "an answer", []) is None
    assert await Judge(call).judge("q", "  ", []) is None  # nothing to judge


# ------------------------------------------------------------------------------ export
def _result(rows):
    items = [
        SimpleNamespace(
            item={"input": {"id": cid, "category": cat, "question": f"q {cid}"}},
            output={
                "intent": "sql",
                "agents_used": ["sql"],
                "status": "answered",
                "answer": "a\nb",
                "cache_hit": False,
            },
            evaluations=[SimpleNamespace(name=n, value=v) for n, v in scores.items()],
        )
        for cid, cat, scores in rows
    ]
    return SimpleNamespace(item_results=items, dataset_run_url="https://example/run")


def test_export_writes_a_csv_and_a_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "RESULTS", tmp_path)
    result = _result(
        [
            (
                "n01",
                "normal",
                {"route_correct": 1.0, "latency_ms": 900.0, "judge_clarity": 4.0},
            ),
            ("n02", "normal", {"route_correct": 0.0, "latency_ms": 1100.0}),
            ("f01", "failure", {"guardrail_blocked": 1.0}),
        ]
    )
    csv_path, md_path = run_eval.export(result, "demo", "development")
    rows = list(csv.DictReader(csv_path.open()))
    assert [r["case"] for r in rows] == ["n01", "n02", "f01"]
    assert rows[0]["route_correct"] == "1.0" and rows[2]["route_correct"] == ""
    assert "\n" not in rows[0]["answer"]
    md = md_path.read_text()
    assert "| route_correct | 2 | 0.50 |" in md and "| judge_clarity | 1 | 4.00 |" in md
    assert "`n02`: route_correct" in md and "| failure | 1 | 1.00 |" in md
    assert "https://example/run" in md


def test_the_summary_reports_no_failures_when_there_are_none():
    md = run_eval.summary(
        [{"case": "a", "category": "x", "route_correct": 1.0}],
        ["route_correct"],
        "r",
        "development",
        None,
    )
    assert "- none" in md


def test_the_summary_flags_cases_where_the_model_call_failed():
    rows = [
        {"case": "n01", "category": "normal", "answer": "fine", "answered": 1},
        {
            "case": "n02",
            "category": "normal",
            "answer": "no valid plan (the model call failed: rate limit)",
            "answered": 1,
        },
    ]
    md = run_eval.summary(rows, ["answered"], "r", "development", None)
    assert (
        "Not a quality result" in md
        and "n02" in md
        and "n01" not in md.split("Not a quality")[1].split("\n")[0]
    )


def test_preflight_stops_the_run_when_the_model_is_unusable(monkeypatch):
    import pytest

    class Down:
        async def ainvoke(self, _):
            raise RuntimeError("Rate limit exceeded")

    monkeypatch.setattr(
        "multi_agenti_conversation_system.llm.get_chat_model", lambda: Down()
    )
    with pytest.raises(SystemExit, match="Nothing was run"):
        run_eval.preflight()


def test_evidence_shows_the_sql_and_rows_but_never_pii_rows():
    from multi_agenti_conversation_system.evals.judge import evidence_text

    results = {
        "s1": {
            "agent": "sql",
            "sql": "SELECT country, COUNT(*) FROM customers GROUP BY 1",
            "data": {
                "columns": ["country", "n"],
                "rows": [["USA", 13]],
                "row_count": 1,
            },
        },
        "s2": {
            "agent": "sql",
            "sql": "SELECT phone FROM customers",
            "data": {"pii_columns": ["phone"], "rows": [["secret"]], "row_count": 1},
        },
        "s3": {"agent": "sql", "data": {"rejected": True}},
        "s4": {"agent": "sql", "data": {"refused": True}},
        "s5": {"agent": "web", "data": {}},
    }
    text = evidence_text(results)
    assert "USA" in text and "13" in text
    assert "secret" not in text and "personal data" in text
    assert "approval was declined" in text and "refused the request" in text
    assert evidence_text(None) == "" and evidence_text({"w": {"agent": "web"}}) == ""


async def test_the_judge_passes_the_evidence_to_the_prompt():
    seen = {}

    async def call(messages):
        seen["text"] = str(messages)
        return Judgement(**{c: {"score": 4, "reason": "ok"} for c in CRITERIA})

    await Judge(call).judge("q", "a", [], "", evidence="s1: SQL run: SELECT 1")
    assert "SELECT 1" in seen["text"]
