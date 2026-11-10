"""The SQL agent graph, run with a fake database, a scripted model and a fake pair store.

Every path of the Phase D gate is exercised deterministically here (a live model can't be
made to fail on cue): self-correction, retries running out, approval granted and declined,
a refused write, PII, and "an approval covers one exact SQL string".
"""

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from multi_agenti_conversation_system.a2a.contracts import AgentError, TaskRequest
from multi_agenti_conversation_system.agents.sql.context import SqlContext
from multi_agenti_conversation_system.agents.sql.graph import (
    build_graph,
    build_input,
    to_result,
)
from multi_agenti_conversation_system.agents.sql.state import SQLGeneration
from multi_agenti_conversation_system.memory.sql_pairs import Pair

GOOD = "SELECT COUNT(*) AS n FROM orders"
FIXED = "SELECT COUNT(*) AS n FROM orders WHERE freight > 10"
PII_SQL = "SELECT contact_name FROM customers"
PII_SQL_2 = "SELECT contact_name, country FROM customers"


def ok_explain(sql, rows=5, pii=()):
    return {
        "ok": True,
        "sql": sql,
        "estimated_rows": rows,
        "touches_pii": bool(pii),
        "pii_columns": list(pii),
    }


def ok_run(rows=((3,),), pii=()):
    return {
        "ok": True,
        "columns": ["n"],
        "rows": [list(r) for r in rows],
        "row_count": len(rows),
        "truncated": False,
        "pii_columns": list(pii),
    }


class FakeDb:
    def __init__(self, explain=None, run=None):
        self.explain = explain or (lambda sql: ok_explain(sql))
        self.run = run or (lambda sql: ok_run())
        self.explained, self.ran = [], []

    async def search_schema(self, question, k, domain):
        return {
            "ok": True,
            "tables": [{"document": "Table: orders\nColumns: order_id, freight"}],
            "degraded": False,
        }

    async def get_few_shots(self, question, k, domain):
        return {
            "ok": True,
            "examples": [{"question": "seed question", "sql": "SELECT 1"}],
        }

    async def explain_query(self, sql):
        self.explained.append(sql)
        return self.explain(sql)

    async def run_select_query(self, sql, max_rows):
        self.ran.append(sql)
        return self.run(sql)


class FakePairs:
    def __init__(self, found=()):
        self.found, self.written, self.searched = list(found), [], []

    async def search(self, question, k, schema_version, domain=None):
        self.searched.append(question)
        return self.found

    async def write(self, question, sql, tables, domain, schema_version):
        self.written.append((question, sql))
        return "sqlpair-test"


class ScriptedModel:
    """Returns the scripted generations in order and records every prompt it was given."""

    def __init__(self, *generations):
        self.queue, self.seen = list(generations), []

    async def generate(self, messages):
        self.seen.append(messages)
        return self.queue.pop(0)

    async def summarise(self, messages):
        return "There are 3 orders."


def gen(sql, tables=("orders",), refusal=None):
    return SQLGeneration(
        sql=sql, explanation="because", tables_used=list(tables), refusal=refusal
    )


def make(model, db=None, pairs=None, approval_rows=100):
    db, pairs = db or FakeDb(), pairs or FakePairs()
    ctx = SqlContext(
        db=db,
        pairs=pairs,
        generate=model.generate,
        summarise=model.summarise,
        table_names={"orders", "customers"},
        table_domains={"orders": "sales", "customers": "customers"},
        schema_version="v1",
        approval_rows=approval_rows,
        max_rows=100,
    )
    return build_graph(InMemorySaver()), ctx, db, pairs


def request(question="How many orders?", **kw):
    return TaskRequest(sender="t", skill="answer_sql_question", task=question, **kw)


async def run(graph, ctx, req=None, thread="t1"):
    config = {"configurable": {"thread_id": thread}}
    out = await graph.ainvoke(build_input(req or request()), config, context=ctx)
    return out, config


async def resume(graph, ctx, config, approved):
    return await graph.ainvoke(
        Command(resume={"approved": approved}), config, context=ctx
    )


# ------------------------------------------------------------------------ happy path
async def test_a_plain_question_runs_and_is_learned():
    model = ScriptedModel(gen(GOOD))
    graph, ctx, db, pairs = make(model)
    state, _ = await run(graph, ctx)
    assert db.ran == [GOOD] and state["attempts"] == 1
    assert state["rows"] == [[3]] and state["answer"] == "There are 3 orders."
    assert pairs.written == [("How many orders?", GOOD)]
    result = to_result(state, request())
    assert result.ok and result.sql == GOOD and result.data["attempts"] == 1
    assert result.data["rows"] == [[3]] and isinstance(result.data["rows"][0][0], int)
    assert result.prompt_versions["sql/generate"] == "sql/generate@local"


async def test_the_model_sees_seed_examples_and_the_schema():
    model = ScriptedModel(gen(GOOD))
    graph, ctx, *_ = make(model)
    await run(graph, ctx)
    prompt = "\n".join(m["content"] for m in model.seen[0])
    assert "Table: orders" in prompt and "seed question" in prompt
    assert "How many orders?" in prompt


async def test_learned_pairs_are_fused_with_seed_examples():
    pairs = FakePairs([Pair("learned question", "SELECT 2")])
    model = ScriptedModel(gen(GOOD))
    graph, ctx, *_ = make(model, pairs=pairs)
    await run(graph, ctx)
    prompt = "\n".join(m["content"] for m in model.seen[0])
    assert "learned question" in prompt and "seed question" in prompt


# ------------------------------------------------------------------ self-correction
async def test_a_failed_explain_is_fed_back_and_corrected():
    def explain(sql):
        if sql == GOOD:
            return {
                "ok": False,
                "error": 'UndefinedColumn: column "nope" does not exist',
            }
        return ok_explain(sql)

    model = ScriptedModel(gen(GOOD), gen(FIXED))
    graph, ctx, db, _ = make(model, FakeDb(explain=explain))
    state, _ = await run(graph, ctx)
    assert state["attempts"] == 2 and db.ran == [FIXED]
    second_prompt = "\n".join(m["content"] for m in model.seen[1])
    assert (
        "UndefinedColumn" in second_prompt and GOOD in second_prompt
    )  # error + failed SQL
    assert to_result(state, request()).data["attempts"] == 2


async def test_an_execution_error_is_corrected_too():
    def run_sql(sql):
        return {"ok": False, "error": "DivisionByZero"} if sql == GOOD else ok_run()

    model = ScriptedModel(gen(GOOD), gen(FIXED))
    graph, ctx, db, _ = make(model, FakeDb(run=run_sql))
    state, _ = await run(graph, ctx)
    assert (
        db.ran == [GOOD, FIXED] and state["attempts"] == 2 and state["row_count"] == 1
    )


async def test_retries_run_out_and_the_task_fails_cleanly():
    bad = {"ok": False, "error": "UndefinedColumn"}
    model = ScriptedModel(gen(GOOD), gen(GOOD), gen(GOOD))
    graph, ctx, db, pairs = make(model, FakeDb(explain=lambda sql: bad))
    state, _ = await run(graph, ctx)
    assert state["attempts"] == 3  # 1 + SQL_MAX_RETRIES
    assert db.ran == [] and pairs.written == []
    result = to_result(state, request())
    assert result.status == "failed" and result.error.code == "failed"
    assert "3 attempt" in result.error.message


# ---------------------------------------------------------------------- guardrails
async def test_the_validator_stops_a_write_before_the_database_sees_it():
    model = ScriptedModel(gen("DELETE FROM orders"), gen(GOOD))
    graph, ctx, db, _ = make(model)
    _state, _ = await run(graph, ctx)
    assert db.explained == [GOOD] and db.ran == [
        GOOD
    ]  # the DELETE never left the agent
    assert "Only SELECT" in "\n".join(m["content"] for m in model.seen[1])


async def test_a_model_that_keeps_writing_fails_without_touching_the_database():
    model = ScriptedModel(*[gen("DROP TABLE orders")] * 3)
    graph, ctx, db, _ = make(model)
    state, _ = await run(graph, ctx)
    assert db.explained == [] and db.ran == []
    assert to_result(state, request()).status == "failed"


async def test_a_refusal_is_a_completed_task_not_a_failure():
    model = ScriptedModel(gen("", refusal="this database is read-only"))
    graph, ctx, db, pairs = make(model)
    state, _ = await run(graph, ctx, request("Delete all orders"))
    assert db.explained == [] and db.ran == [] and pairs.written == []
    result = to_result(state, request())
    assert result.ok and result.data["refused"] is True and "read-only" in result.answer


# ------------------------------------------------------------------------ approval
def pii_db(extra=None):
    def explain(sql):
        return ok_explain(sql, pii=("customers.contact_name",))

    return FakeDb(explain=explain, **(extra or {}))


async def test_pii_pauses_for_approval_then_runs_after_a_yes():
    model = ScriptedModel(gen(PII_SQL, ("customers",)))
    graph, ctx, db, pairs = make(
        model, pii_db({"run": lambda sql: ok_run(pii=("customers.contact_name",))})
    )
    out, config = await run(graph, ctx)
    payload = out["__interrupt__"][0].value
    assert payload["sql"] == PII_SQL and payload["pii_columns"] == [
        "customers.contact_name"
    ]
    assert "personal-data" in payload["reason"] and db.ran == []  # nothing ran yet
    state = await resume(graph, ctx, config, approved=True)
    assert db.ran == [PII_SQL] and state["row_count"] == 1
    assert pairs.written == []  # a PII query is never learned
    assert to_result(state, request()).data["pii_columns"] == ["customers.contact_name"]


async def test_a_declined_approval_runs_nothing_and_completes():
    model = ScriptedModel(gen(PII_SQL, ("customers",)))
    graph, ctx, db, pairs = make(model, pii_db())
    _out, config = await run(graph, ctx)
    state = await resume(graph, ctx, config, approved=False)
    assert db.ran == [] and pairs.written == []
    result = to_result(state, request())
    assert result.ok and result.data["rejected"] is True and result.sql == PII_SQL


async def test_anything_but_an_explicit_true_is_a_no():
    model = ScriptedModel(gen(PII_SQL, ("customers",)))
    graph, ctx, db, _ = make(model, pii_db())
    _out, config = await run(graph, ctx)
    state = await graph.ainvoke(
        Command(resume={"approved": "yes"}), config, context=ctx
    )
    assert db.ran == [] and state["rejected"] is True


RAW_LIST = "SELECT order_id, freight FROM orders"
BY_YEAR = "SELECT EXTRACT(YEAR FROM order_date) AS y, COUNT(*) FROM orders GROUP BY 1"


async def test_a_big_raw_result_needs_approval_even_without_pii():
    graph, ctx, db, _ = make(
        ScriptedModel(gen(RAW_LIST)),
        FakeDb(explain=lambda sql: ok_explain(sql, rows=100)),
    )
    out, _ = await run(graph, ctx)
    assert "cut off" in out["__interrupt__"][0].value["reason"] and db.ran == []


async def test_an_aggregate_never_asks_because_of_the_row_estimate():
    """The planner's estimate for a join with an aggregate hits the 100-row clamp even when
    three rows come back; asking a human to approve that would stall every supervisor run."""
    for sql in (GOOD, BY_YEAR):
        graph, ctx, db, _ = make(
            ScriptedModel(gen(sql)), FakeDb(explain=lambda s: ok_explain(s, rows=100))
        )
        out, _ = await run(graph, ctx)
        assert "__interrupt__" not in out and db.ran == [sql]


async def test_an_aggregate_over_pii_still_asks():
    sql = "SELECT COUNT(contact_name) FROM customers"
    graph, ctx, db, _ = make(ScriptedModel(gen(sql, ("customers",))), pii_db())
    out, _ = await run(graph, ctx)
    assert "personal-data" in out["__interrupt__"][0].value["reason"] and db.ran == []


async def test_a_small_raw_result_does_not_ask():
    graph, ctx, db, _ = make(
        ScriptedModel(gen(RAW_LIST)),
        FakeDb(explain=lambda sql: ok_explain(sql, rows=99)),
    )
    out, _ = await run(graph, ctx)
    assert "__interrupt__" not in out and db.ran == [RAW_LIST]


async def test_an_approval_covers_one_exact_sql_string():
    """Approve, then the query errors, then the corrected SQL must ask again."""

    def run_sql(sql):
        return {"ok": False, "error": "UndefinedColumn"} if sql == PII_SQL else ok_run()

    model = ScriptedModel(gen(PII_SQL, ("customers",)), gen(PII_SQL_2, ("customers",)))
    graph, ctx, db, _ = make(model, pii_db({"run": run_sql}))
    out, config = await run(graph, ctx)
    out = await resume(graph, ctx, config, approved=True)
    assert db.ran == [PII_SQL]  # the approved string ran once, and failed
    assert (
        out["__interrupt__"][0].value["sql"] == PII_SQL_2
    )  # the new string asks again
    assert db.ran == [PII_SQL]  # ... and has not run unapproved
    state = await resume(graph, ctx, config, approved=True)
    assert db.ran == [PII_SQL, PII_SQL_2] and state["row_count"] == 1


# --------------------------------------------------------------------- reliability
async def test_a_database_outage_is_unavailable_not_a_sql_error():
    class DownDb(FakeDb):
        async def explain_query(self, sql):
            raise AgentError(
                "unavailable", "database tools unavailable", retryable=True
            )

    model = ScriptedModel(gen(GOOD), gen(GOOD))
    graph, ctx, *_ = make(model, DownDb())
    with pytest.raises(AgentError) as caught:
        await run(graph, ctx)
    assert caught.value.code == "unavailable"
    assert len(model.seen) == 1  # the outage was never handed to the model to "fix"


async def test_a_pair_write_failure_does_not_fail_the_answer():
    class BrokenPairs(FakePairs):
        async def write(self, *a, **k):
            raise RuntimeError("pinecone down")

    graph, ctx, *_ = make(ScriptedModel(gen(GOOD)), pairs=BrokenPairs())
    state, _ = await run(graph, ctx)
    assert to_result(state, request()).ok and state["learned_pair"] is None


async def test_a_failed_summary_still_returns_the_rows():
    model = ScriptedModel(gen(GOOD))

    async def broken(messages):
        raise RuntimeError("model down")

    model.summarise = broken
    graph, ctx, *_ = make(model)
    state, _ = await run(graph, ctx)
    assert state["answer"] == "The query returned 1 row(s)." and state["rows"] == [[3]]


# ------------------------------------------------------------------------ plumbing
async def test_a_second_task_in_the_same_context_starts_clean():
    model = ScriptedModel(gen(PII_SQL, ("customers",)), gen(GOOD))
    graph, ctx, _db, _ = make(
        model,
        FakeDb(
            explain=lambda sql: ok_explain(
                sql, pii=("c.x",) if "contact" in sql else ()
            )
        ),
    )
    _out, config = await run(graph, ctx, request("first"), thread="same")
    await resume(graph, ctx, config, approved=False)  # first task: rejected
    state, _ = await run(graph, ctx, request("second"), thread="same")
    assert state["rejected"] is False and state["approved_sql"] == ""
    assert state["question"] == "second" and state["attempts"] == 1
    assert to_result(state, request()).data["rows"] == [[3]]


def test_max_rows_can_only_be_lowered_by_the_caller():
    from multi_agenti_conversation_system.a2a.contracts import TaskConstraints

    low = build_input(request(constraints=TaskConstraints(max_rows=10)))
    high = build_input(request(constraints=TaskConstraints(max_rows=10_000)))
    assert low["max_rows"] == 10 and high["max_rows"] == 100


def test_the_session_domain_reaches_the_graph():
    from multi_agenti_conversation_system.a2a.contracts import TaskContext

    req = request(context=TaskContext(session={"selected_domain": "sales"}))
    assert build_input(req)["domain"] == "sales"


# ------------------------------------------------------------------- trace hygiene
async def test_pii_rows_and_answers_stay_out_of_traces(monkeypatch):
    """The mask can't recognise names, so a PII result must not reach a span at all."""
    from contextlib import contextmanager

    from multi_agenti_conversation_system import tracing
    from multi_agenti_conversation_system.agents.sql.nodes import OMITTED

    recorded: dict[str, dict] = {}

    class Span:
        def __init__(self, name):
            self.name = name

        def update(self, **kw):
            recorded[self.name]["output"] = kw["output"]

    @contextmanager
    def fake_observation(name, **kw):
        recorded[name] = {"input": kw.get("input")}
        yield Span(name)

    monkeypatch.setattr(tracing, "observation", fake_observation)
    secret_rows = [["Maria Anders"]]
    db = FakeDb(
        explain=lambda sql: ok_explain(sql, pii=("customers.contact_name",)),
        run=lambda sql: ok_run(rows=secret_rows, pii=("customers.contact_name",)),
    )
    graph, ctx, *_ = make(ScriptedModel(gen(PII_SQL, ("customers",))), db)
    _out, config = await run(graph, ctx)
    await resume(graph, ctx, config, approved=True)

    assert recorded["explain-result"] == {"input": OMITTED, "output": OMITTED}
    assert "Maria Anders" not in str(recorded)
    # The SQL-generation span carries schema and examples, never data: it is not omitted.
    assert recorded["generate-sql"]["input"] != OMITTED


async def test_a_non_pii_result_is_traced_normally(monkeypatch):
    from contextlib import contextmanager

    from multi_agenti_conversation_system import tracing
    from multi_agenti_conversation_system.agents.sql.nodes import OMITTED

    seen: dict[str, object] = {}

    @contextmanager
    def fake_observation(name, **kw):
        seen[name] = kw.get("input")
        yield None

    monkeypatch.setattr(tracing, "observation", fake_observation)
    graph, ctx, *_ = make(ScriptedModel(gen(GOOD)))
    await run(graph, ctx)
    assert seen["explain-result"] != OMITTED


# ----------------------------------------------- database outage vs. a mistake in the SQL
@pytest.mark.parametrize(
    "error",
    [
        "OperationalError: connection is closed",
        "AdminShutdown: terminating connection due to administrator command",
        "InterfaceError: connection already closed",
        "couldn't get a connection after 30.00 sec",
        "Database unavailable",
    ],
)
@pytest.mark.parametrize("where", ["explain", "run"])
async def test_the_servers_outage_replies_are_unavailable_not_sql_errors(error, where):
    """The real reply shape of the DB server: {"ok": False, "error": "ClassName: text"}."""
    reply = {"ok": False, "error": error}
    db = (
        FakeDb(explain=lambda sql: reply)
        if where == "explain"
        else FakeDb(run=lambda sql: reply)
    )
    model = ScriptedModel(gen(GOOD), gen(GOOD), gen(GOOD))
    graph, ctx, *_ = make(model, db)
    with pytest.raises(AgentError) as caught:
        await run(graph, ctx)
    assert caught.value.code == "unavailable" and caught.value.retryable
    assert len(model.seen) == 1  # the outage was never handed to the model to "fix"


@pytest.mark.parametrize(
    "error",
    [
        'UndefinedColumn: column "nope" does not exist',
        'SyntaxError: syntax error at or near "FORM"',
        "DivisionByZero: division by zero",
        "QueryCanceled: canceling statement due to statement timeout",
    ],
)
async def test_sql_mistakes_and_statement_timeouts_go_back_to_the_model(error):
    def run_sql(sql):
        return {"ok": False, "error": error} if sql == GOOD else ok_run()

    model = ScriptedModel(gen(GOOD), gen(FIXED))
    graph, ctx, db, _ = make(model, FakeDb(run=run_sql))
    state, _ = await run(graph, ctx)
    assert state["attempts"] == 2 and db.ran == [GOOD, FIXED]
    assert error.split(":")[0] in "\n".join(m["content"] for m in model.seen[1])
