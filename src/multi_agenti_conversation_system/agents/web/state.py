"""State of one Web agent run, and the model's structured output for planning."""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


class SearchPlan(BaseModel):
    """What to search for (structured output of the plan_queries step)."""

    queries: list[str] = Field(
        description="1 to 3 short, distinct web search queries that together cover the question"
    )
    recency: Literal["day", "week", "month", "year", "any"] = "any"
    topic: Literal["general", "news"] = "general"


# The fan-out nodes (search_one, fetch_one) each return a one-item list; `operator.add`
# merges them. Every such list must be reset with `Overwrite([])` when a new task starts
# (see graph.build_input), or tasks in one A2A context would see each other's results.
Accumulated = Annotated[list[Any], operator.add]


class WebState(TypedDict, total=False):
    question: str
    # --- planning and search
    queries: list[str]
    recency: str
    topic: str
    hits: Accumulated  # search results, one dict each
    search_skips: Accumulated  # queries skipped because the search was rate limited
    search_errors: Accumulated  # messages of searches that failed outright
    # --- fetching
    urls: list[dict[str, Any]]  # the pages chosen to read
    pages: Accumulated  # fetched pages
    fetch_skips: Accumulated  # URLs skipped because fetching was rate limited
    fetch_failures: Accumulated  # {"url", "error"} for pages that couldn't be read
    # --- reading
    chunks: list[dict[str, Any]]
    chunks_before: int
    chunks_after: int
    snippet_mode: bool  # no page could be read; the search snippets are all we have
    sources: list[dict[str, Any]]
    context_chars: int
    # --- answer
    answer: str
    citations: list[dict[str, Any]]
    confidence: Literal["high", "low", "none"]
    prompt_tags: dict[str, str]
