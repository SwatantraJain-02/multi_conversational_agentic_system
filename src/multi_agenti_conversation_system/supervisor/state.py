"""The supervisor's plan model and graph state."""

import operator
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """One unit of work for one agent."""

    id: str = Field(description='Short unique id, e.g. "s1"')
    agent: str = Field(description="The agent's name, exactly as listed")
    skill: str = Field(description="One of that agent's skill ids, exactly as listed")
    task: str = Field(
        description="The question for this agent, standalone: no pronouns or references to the conversation"
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="Ids of EARLIER steps whose results this step needs",
    )


class RouteDecision(BaseModel):
    """The planner's output (structured)."""

    intent: Literal["web", "sql", "hybrid", "compare", "direct"]
    # Required (no default): a model that sees it as optional leaves it out, which is
    # an empty plan. "direct" says so explicitly with [].
    plan: list[PlanStep] = Field(
        description="The steps, in order. An empty list only for intent direct"
    )
    reason: str = Field(
        default="", description="At most one short sentence. Written after the plan."
    )


class Contextual(BaseModel):
    """`contextualize`: the question made standalone, and the literals that change its answer."""

    standalone_question: str
    entities: list[
        str
    ]  # required: a field with a default is silently left out by free models


class Preference(BaseModel):
    key: str
    value: str
    lasting: (
        bool  # a standing instruction; False for a one-off ("write THIS one in Java")
    )


class NewFact(BaseModel):
    text: str
    attribute: str
    value: str
    temporary: bool
    valid_days: int  # 0 when unknown


class Extraction(BaseModel):
    """`write_memory`: what the user's own message says about them."""

    preferences: list[Preference]
    facts: list[NewFact]


def merge_dicts(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Reducer for results written by parallel steps. Reset with `Overwrite({})`."""
    return {**left, **right}


class AgentState(TypedDict, total=False):
    # --- conversation (checkpointed per thread)
    messages: Annotated[list[AnyMessage], add_messages]
    summary: str  # what older, trimmed messages said
    # --- this turn, reset by `graph.turn_input`
    question: str
    user_id: str
    thread_id: str
    selected_domain: str | None
    intent: str
    plan: list[dict[str, Any]]  # PlanStep dicts
    plan_error: str  # why no valid plan exists ("" when there is one)
    ready: list[str]  # ids of the steps to dispatch now
    dispatched: list[str]
    dispatch_count: int
    skipped: list[str]  # steps dropped because the dispatch budget ran out
    results: Annotated[dict[str, Any], merge_dicts]  # step id -> TaskResult dict
    timings: Annotated[dict[str, Any], merge_dicts]  # step id -> [start, end] (epoch s)
    hitl_decision: dict[str, Any] | None  # {"step_id", "approved"}
    approvals: Annotated[list[dict[str, Any]], operator.add]  # audit: who decided what
    # --- the input guard, this turn
    guard_blocked: bool
    guard_flagged: (
        bool  # suspicious or personal data in the message: nothing is learned
    )
    guard_reasons: list[str]
    input_pii: list[str]
    # --- memory, this turn (reset by `graph.turn_input`)
    standalone_question: str
    entity_key: str  # "" when the question has no entities (never None)
    standalone_ok: bool  # False when it needed history and contextualize failed
    memory_ok: bool  # False when the profile could not be loaded: then no cache either
    profile: dict[str, str]  # {key: value}
    profile_hash: str
    facts: list[str]  # relevant facts, dated, for the prompt
    episodes: list[str]  # relevant past conversations, dated
    cache_hit: bool
    memory_changes: list[
        str
    ]  # what this turn learned, e.g. "profile: code_language=java"
    # --- kept across turns by the checkpointer (not reset)
    turns: int
    # --- the answer
    answer: str
    citations: list[dict[str, Any]]
    sql_used: str | None
    agents_used: list[str]
    prompt_versions: dict[str, str]
    partial: bool
