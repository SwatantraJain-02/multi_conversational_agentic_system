"""State of one Synthesis run."""

from typing import Any, TypedDict


class SynthesisState(TypedDict, total=False):
    # --- input, reset on every task by `graph.build_input`
    question: str
    task_id: str  # this task's TaskRequest id: the parent id of a delegated call
    internal: dict[str, Any] | None  # a TaskResult as a dict, or None
    external: dict[str, Any] | None
    external_question: str | None
    # --- delegation
    delegate: bool  # no usable external result: ask the Web agent
    delegated: bool  # ... and we did
    delegation_error: str  # why the Web agent's answer isn't there (when it isn't)
    # --- result
    gaps: list[str]  # what is missing and why, in words
    answer: str
    citations: list[dict[str, Any]]
    prompt_tags: dict[str, str]
