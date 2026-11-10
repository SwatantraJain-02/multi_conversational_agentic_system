"""The Web agent's only door to the internet: the Web MCP server's tools.

Replies are the tool's dict ({"ok": ..., ...}). A transport failure raises
`AgentError("unavailable")`. A rate-limited reply is *data* ({"ok": False, "error":
"rate_limited", "retry_after_s": ...}): `is_rate_limited` / `retry_after` read it.
"""

from typing import Any

from langchain_core.tools import BaseTool

from multi_agenti_conversation_system.a2a.contracts import AgentError
from multi_agenti_conversation_system.mcp.client import tool_result


def is_rate_limited(reply: dict[str, Any]) -> bool:
    return not reply.get("ok") and reply.get("error") == "rate_limited"


def retry_after(reply: dict[str, Any]) -> float:
    return float(reply.get("retry_after_s") or 0)


class WebTools:
    def __init__(self, tools: dict[str, BaseTool]) -> None:
        self._tools = tools

    async def _call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        try:
            raw = await self._tools[name].ainvoke(args)
        except Exception as exc:
            raise AgentError(
                "unavailable",
                f"web tools unavailable ({type(exc).__name__})",
                retryable=True,
            ) from exc
        return tool_result(raw)

    async def search_web(
        self, query: str, max_results: int, recency: str, topic: str
    ) -> dict[str, Any]:
        return await self._call(
            "search_web",
            {
                "query": query,
                "max_results": max_results,
                "recency": recency,
                "topic": topic,
            },
        )

    async def fetch_page(self, url: str, max_chars: int) -> dict[str, Any]:
        return await self._call("fetch_page", {"url": url, "max_chars": max_chars})
