import asyncio

import httpx
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamable_http_client

URL = "http://127.0.0.1:5555/mcp"


async def main():
    http = httpx.AsyncClient(
        headers={"Authorization": "Bearer read-this-from-settings"}
    )
    async with (
        streamable_http_client(URL, http_client=http) as (read, write, *_),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        print([t.name for t in (await session.list_tools()).tools])
        result = await session.call_tool("add", {"a": 2, "b": 3})
        print(result.is_error, result.content[0].text)


asyncio.run(main())
