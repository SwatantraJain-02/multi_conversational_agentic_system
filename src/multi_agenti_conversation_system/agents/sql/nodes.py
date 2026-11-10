"""The SQL agent's graph nodes and the routing between them.

    retrieve_schema -> generate_sql -> validate -> approval_check -> [hitl_gate] -> execute
                           ^              |             |                            |
                           +--------------+-------------+--------- (error) ----------+
                                                                                     v
                                                              explain_result -> write_sql_pair

Self-correction: a failure in validate, approval_check (EXPLAIN) or execute goes back to
generate_sql with the error text, at most `max_retries` times, then the run fails.

Approval binds to one exact SQL string (`approved_sql`). A corrected query is a different
string, so it passes approval_check again and asks again if it needs to; `execute` also
refuses to run a query that needs approval but isn't the approved one.

`hitl_gate` does nothing but `interrupt()`: resuming re-runs the whole node (G22), so any
side effect would repeat. The EXPLAIN call lives in the node before it.

Nodes get everything from `runtime.context` (a `SqlContext`), and raise `AgentError` for
conditions the caller should see (a dependency is down).
"""

import asyncio
import json
from typing import Any

from langgraph.graph import END
from langgraph.runtime import Runtime
from langgraph.types import interrupt

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.a2a.contracts import AgentError
from multi_agenti_conversation_system.agents.sql.context import Messages, SqlContext
from multi_agenti_conversation_system.agents.sql.db_tools import is_db_down
from multi_agenti_conversation_system.agents.sql.state import SQLGeneration, SqlState
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.guardrails.sql_validator import (
    is_aggregate_query,
    validate_select,
)
from multi_agenti_conversation_system.prompts import Prompt, aget_prompt
from multi_agenti_conversation_system.retrieval.hybrid import rrf

SCHEMA_TABLES = 5
SEED_EXAMPLES = 3
LEARNED_EXAMPLES = 3
EXAMPLES_SHOWN = 3
ROWS_IN_PROMPT = 20
OMITTED = tracing.OMITTED

Rt = Runtime[SqlContext]


# ----------------------------------------------------------------------------- helpers
def _tag(state: SqlState, prompt: Prompt) -> dict[str, str]:
    return {**state.get("prompt_tags", {}), prompt.name: prompt.tag}


def _unavailable(reply: dict[str, Any], what: str) -> AgentError:
    return AgentError("unavailable", f"{what}: {reply.get('error')}", retryable=True)


def _format_examples(examples: list[dict[str, str]]) -> str:
    if not examples:
        return "(none)"
    return "\n\n".join(f"Q: {e['question']}\nSQL:\n{e['sql']}" for e in examples)


def _feedback(state: SqlState) -> str:
    if not state.get("error"):
        return ""
    return (
        "Your previous query did not work.\n"
        f"SQL:\n{state.get('failed_sql', '')}\nProblem: {state['error']}\n"
        "Write a corrected query.\n\n"
    )


async def _generation(
    call, messages: Messages, prompt: Prompt, name: str, *, sensitive: bool = False
) -> Any:
    return await tracing.generation(
        call,
        messages,
        name,
        prompt_link=prompt.link,
        model=settings.CHAT_MODEL,
        sensitive=sensitive,
    )


# ------------------------------------------------------------------------------- nodes
async def retrieve_schema(state: SqlState, runtime: Rt) -> dict[str, Any]:
    """Relevant tables, plus examples: seed few-shots and learned pairs fused by a second RRF."""
    ctx, question, domain = runtime.context, state["question"], state.get("domain")
    schema, shots, learned = await asyncio.gather(
        ctx.db.search_schema(question, SCHEMA_TABLES, domain),
        ctx.db.get_few_shots(question, SEED_EXAMPLES, domain),
        ctx.pairs.search(question, LEARNED_EXAMPLES, ctx.schema_version, domain),
    )
    if not schema.get("ok"):
        error = str(schema.get("error", ""))
        if error.startswith("Unknown domain"):
            raise AgentError("bad_request", error)
        raise _unavailable(schema, "schema search failed")

    seed = [
        {"question": e["question"], "sql": e["sql"]}
        for e in (shots.get("examples", []) if shots.get("ok") else [])
    ]
    mine = [{"question": p.question, "sql": p.sql} for p in learned]
    by_question = {e["question"]: e for e in [*seed, *mine]}
    fused = rrf([[e["question"] for e in seed], [e["question"] for e in mine]])
    return {
        "schema_docs": [t["document"] for t in schema["tables"]],
        "examples": [by_question[q] for q, _ in fused[:EXAMPLES_SHOWN]],
        "degraded": bool(schema.get("degraded")),
    }


async def generate_sql(state: SqlState, runtime: Rt) -> dict[str, Any]:
    prompt = await aget_prompt("sql/generate")
    messages = prompt.compile(
        question=state["question"],
        domain=state.get("domain") or "any",
        schema="\n\n".join(state.get("schema_docs", [])),
        examples=_format_examples(state.get("examples", [])),
        feedback=_feedback(state),
        max_rows=str(state["max_rows"]),
    )
    generation: SQLGeneration = await _generation(
        runtime.context.generate, messages, prompt, "generate-sql"
    )
    sql = generation.sql.strip().rstrip(";").strip()
    update: dict[str, Any] = {
        "attempts": state["attempts"] + 1,
        "prompt_tags": _tag(state, prompt),
        "explanation": generation.explanation,
        "tables_used": generation.tables_used,
        "sql": sql,
        "error": "",
    }
    if generation.refusal and not sql:
        update["refused"] = generation.refusal
    elif not sql:
        update["error"] = "the model returned no SQL"
    return update


async def validate(state: SqlState, runtime: Rt) -> dict[str, Any]:
    """Layer 1 of the SQL guardrail (the MCP server checks again): one SELECT, known tables."""
    if state.get("error"):  # generate_sql already failed this attempt
        return {"failed_sql": state.get("sql", "")}
    ctx = runtime.context
    checked = validate_select(state["sql"], ctx.max_rows, ctx.table_names)
    if not checked.ok:
        return {"error": checked.error or "query rejected", "failed_sql": state["sql"]}
    return {"error": "", "tables_used": sorted(checked.tables)}


async def approval_check(state: SqlState, runtime: Rt) -> dict[str, Any]:
    """EXPLAIN the query (read-only) and decide whether a human has to approve it."""
    ctx, sql = runtime.context, state["sql"]
    check = await ctx.db.explain_query(sql)
    if not check.get("ok"):
        error = str(check.get("error", "explain failed"))
        if is_db_down(error):
            raise _unavailable(check, "database unavailable")
        # Typically a wrong column or table name: exactly what the model can fix.
        return {"error": error, "failed_sql": sql}

    pii = list(check.get("pii_columns", []))
    est_rows = int(check.get("estimated_rows", 0))
    reasons = []
    if pii:
        reasons.append(f"it reads personal-data columns ({', '.join(pii)})")
    # An aggregate's result is small whatever the estimate says (see is_aggregate_query).
    if est_rows >= ctx.approval_rows and not is_aggregate_query(sql):
        reasons.append(
            f"it matches about {est_rows} rows and the result would be cut off at {ctx.max_rows}"
        )
    requires = bool(reasons)
    return {
        "error": "",
        "est_rows": est_rows,
        "pii_columns": pii,
        "requires_approval": requires,
        # An earlier yes covers only that exact string; a corrected query asks again.
        "needs_approval": requires and state.get("approved_sql") != sql,
        "approval_reason": "; ".join(reasons),
    }


def hitl_gate(state: SqlState) -> dict[str, Any]:
    """Pause for a human. Only interrupt() here: resuming re-runs this whole node (G22)."""
    decision = interrupt(
        {
            "sql": state["sql"],
            "reason": f"Approval needed: {state['approval_reason']}.",
            "est_rows": state.get("est_rows", 0),
            "pii_columns": state.get("pii_columns", []),
        }
    )
    approved = isinstance(decision, dict) and decision.get("approved") is True
    return {
        "approved_sql": state["sql"] if approved else state.get("approved_sql", ""),
        "rejected": not approved,
    }


async def execute(state: SqlState, runtime: Rt) -> dict[str, Any]:
    sql = state["sql"]
    if state.get("requires_approval") and state.get("approved_sql") != sql:
        # Defence in depth: the routing should make this impossible.
        raise AgentError("internal", "a query that needs approval was not approved")
    result = await runtime.context.db.run_select_query(sql, state["max_rows"])
    if not result.get("ok"):
        error = str(result.get("error", "query failed"))
        if is_db_down(error):
            raise _unavailable(result, "database unavailable")
        return {"error": error, "failed_sql": sql}
    return {
        "error": "",
        "columns": result["columns"],
        "rows": result["rows"],
        "row_count": result["row_count"],
        "truncated": bool(result["truncated"]),
        "pii_columns": sorted(
            {*state.get("pii_columns", []), *result.get("pii_columns", [])}
        ),
    }


async def explain_result(state: SqlState, runtime: Rt) -> dict[str, Any]:
    """Turn the rows into an answer. If the model fails, the rows are still the result."""
    sensitive = bool(state.get("pii_columns"))
    shown = state["rows"][:ROWS_IN_PROMPT]
    try:
        prompt = await aget_prompt("sql/explain")
        messages = prompt.compile(
            question=state["question"],
            sql=state["sql"],
            columns=", ".join(state["columns"]),
            rows=json.dumps(shown, default=str),
            row_count=str(state["row_count"]),
            truncated=", cut off" if state["truncated"] else "",
        )
        answer = await _generation(
            runtime.context.summarise,
            messages,
            prompt,
            "explain-result",
            sensitive=sensitive,
        )
        return {"answer": str(answer).strip(), "prompt_tags": _tag(state, prompt)}
    except Exception:  # noqa: BLE001 - the data is the result; the sentence is a bonus
        return {"answer": f"The query returned {state['row_count']} row(s)."}


async def write_sql_pair(state: SqlState, runtime: Rt) -> dict[str, Any]:
    """Remember a validated question -> SQL pair. Never for PII, and never fatal."""
    if (
        state.get("pii_columns")
        or state.get("requires_approval")
        or not state.get("row_count")
    ):
        return {"learned_pair": None}
    ctx = runtime.context
    domains = [ctx.table_domains.get(t.lower()) for t in state.get("tables_used", [])]
    domain = next((d for d in domains if d), state.get("domain"))
    try:
        pair_id = await ctx.pairs.write(
            state["question"],
            state["sql"],
            state.get("tables_used", []),
            domain,
            ctx.schema_version,
        )
    except Exception:  # noqa: BLE001 - learning must not fail an answered question
        return {"learned_pair": None}
    return {"learned_pair": pair_id}


def fail_end(state: SqlState) -> dict[str, Any]:
    return {
        "failure": {
            "code": "failed",
            "message": f"could not produce a working query after {state['attempts']} attempt(s): "
            f"{state.get('error', 'unknown error')}",
        }
    }


# ----------------------------------------------------------------------------- routing
def _retry_or_fail(state: SqlState) -> str:
    return "generate_sql" if state["attempts"] <= state["max_retries"] else "fail_end"


def after_generate(state: SqlState) -> str:
    return END if state.get("refused") else "validate"


def after_validate(state: SqlState) -> str:
    return _retry_or_fail(state) if state.get("error") else "approval_check"


def after_approval_check(state: SqlState) -> str:
    if state.get("error"):
        return _retry_or_fail(state)
    return "hitl_gate" if state.get("needs_approval") else "execute"


def after_hitl(state: SqlState) -> str:
    return END if state.get("rejected") else "execute"


def after_execute(state: SqlState) -> str:
    return _retry_or_fail(state) if state.get("error") else "explain_result"
