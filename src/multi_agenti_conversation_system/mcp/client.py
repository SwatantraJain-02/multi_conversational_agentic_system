"""MCP client: connects to our MCP servers with their bearer tokens and returns LangChain tools.

    async with open_mcp(["db"]) as tools:      # {"db": [BaseTool, ...]}
        ...                                    # connections stay open inside the block

Each server gets its own adapter, so tools stay grouped per server: the SQL agent only sees
the DB tools and the Web agent only the web tools. One connection per server is held open
for the whole block (the caller's lifetime), not reopened per call.

`langchain.mcp` is in beta (G2): the import warning is silenced here, in one place.
"""

import warnings
from collections.abc import AsyncIterator, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass

from fastmcp.client.transports import StreamableHttpTransport
from langchain_core._api import LangChainBetaWarning
from langchain_core.tools import BaseTool

from multi_agenti_conversation_system.config import settings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", LangChainBetaWarning)
    from langchain.mcp import MCPAdapter


@dataclass(frozen=True)
class McpServer:
    name: str
    url: str
    token: str

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def __repr__(self) -> str:  # never print the token
        return f"McpServer(name={self.name!r}, url={self.url!r}, token={'***' if self.token else ''})"


def servers() -> dict[str, McpServer]:
    """The configured MCP servers, read from settings at call time."""
    return {
        "db": McpServer(
            "db",
            f"http://{settings.DB_MCP_HOST}:{settings.DB_MCP_PORT}/mcp",
            settings.DB_MCP_TOKEN,
        ),
        "web": McpServer(
            "web",
            f"http://{settings.WEB_MCP_HOST}:{settings.WEB_MCP_PORT}/mcp",
            settings.WEB_MCP_TOKEN,
        ),
    }


@asynccontextmanager
async def open_mcp(names: Sequence[str]) -> AsyncIterator[dict[str, list[BaseTool]]]:
    """Connect to the named servers and yield their tools, grouped per server.

    Raises ValueError for an unknown server name. A server that is down or rejects the
    token raises from here, at startup, so the caller fails fast instead of mid-conversation.
    """
    available = servers()
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(f"unknown MCP server(s) {unknown}; known: {sorted(available)}")

    async with AsyncExitStack() as stack:
        tools: dict[str, list[BaseTool]] = {}
        for name in names:
            server = available[name]
            transport = StreamableHttpTransport(server.url, headers=server.headers)
            adapter = await stack.enter_async_context(MCPAdapter(transport))
            tools[name] = await adapter.list_tools()
        yield tools
