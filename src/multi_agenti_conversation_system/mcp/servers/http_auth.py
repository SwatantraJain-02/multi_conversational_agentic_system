"""HTTP-layer pieces shared by the MCP servers (bearer-token auth, loopback detection)."""

import ipaddress
import secrets
from collections.abc import Iterable

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

OPEN_PATHS = frozenset({"/health", "/ready"})


class BearerAuth:
    """Pure ASGI middleware: rejects requests without the right token, except on open_paths.

    An empty token disables the check; callers must only allow that on a loopback address.
    """

    def __init__(
        self, app: ASGIApp, token: str, open_paths: Iterable[str] = OPEN_PATHS
    ):
        self.app, self.token = app, token
        self.open_paths = frozenset(open_paths)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            self.token
            and scope["type"] == "http"
            and scope["path"] not in self.open_paths
        ):
            got = dict(scope["headers"]).get(b"authorization", b"").decode()
            if not secrets.compare_digest(got, f"Bearer {self.token}"):
                await JSONResponse({"error": "unauthorized"}, status_code=401)(
                    scope, receive, send
                )
                return
        await self.app(scope, receive, send)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False
