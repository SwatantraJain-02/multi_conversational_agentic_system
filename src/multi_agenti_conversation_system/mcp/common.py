"""HTTP plumbing shared by the MCP servers: bearer-token auth and the loopback check.

Each server is its own process; this module holds only what they'd otherwise copy.
"""

import ipaddress
import secrets

from starlette.responses import JSONResponse


class BearerAuth:
    """Pure ASGI middleware: rejects requests without the right token (except /health)."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if self.token and scope["type"] == "http" and scope["path"] != "/health":
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


def require_token_off_loopback(token: str, host: str, env_name: str) -> None:
    """An empty token is only allowed when the server listens on loopback."""
    if not token and not is_loopback(host):
        raise SystemExit(
            f"Refusing to listen on a non-local address without {env_name} set."
        )
