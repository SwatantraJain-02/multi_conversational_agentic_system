"""The only module that talks to Tavily: web search, and Extract as the fetch fallback."""

from functools import cache
from typing import Any, Literal

from tavily import TavilyClient

from multi_agenti_conversation_system.config import settings

Recency = Literal["day", "week", "month", "year", "any"]
Topic = Literal["general", "news"]


@cache
def _client() -> TavilyClient:
    if not settings.TAVILY_API_KEY:
        raise RuntimeError("TAVILY_API_KEY is not set")
    return TavilyClient(api_key=settings.TAVILY_API_KEY)


def search(
    query: str,
    max_results: int,
    recency: Recency,
    topic: Topic,
    include_domains: list[str] | None,
    exclude_domains: list[str] | None,
) -> list[dict[str, Any]]:
    """Results as [{title, url, snippet, published_date, score}], best first."""
    response = _client().search(
        query,
        search_depth="basic",
        topic=topic,
        time_range=None if recency == "any" else recency,
        max_results=max_results,
        include_domains=include_domains or None,
        exclude_domains=exclude_domains or None,
    )
    return [
        {
            "title": r.get("title"),
            "url": r.get("url"),
            "snippet": r.get("content"),
            "published_date": r.get("published_date"),
            "score": r.get("score"),
        }
        for r in response.get("results", [])
    ]


def extract(url: str) -> str | None:
    """Tavily's own extraction of one URL (renders JS-heavy pages), or None."""
    response = _client().extract(url, extract_depth="basic", format="text")
    for result in response.get("results", []):
        if result.get("raw_content"):
            return result["raw_content"]
    return None
