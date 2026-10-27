import pytest
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from multi_agenti_conversation_system.mcp.servers.http_auth import (
    BearerAuth,
    is_loopback,
)


def _app(token: str) -> TestClient:
    async def ok(request):
        return PlainTextResponse("ok")

    inner = Starlette(routes=[Route(p, ok) for p in ("/health", "/ready", "/mcp")])
    return TestClient(BearerAuth(inner, token=token))


@pytest.mark.parametrize("path", ["/health", "/ready"])
def test_open_paths_need_no_token(path):
    assert _app("secret").get(path).status_code == 200


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "secret"},
        {"Authorization": "Bearer secret "},
    ],
)
def test_missing_or_wrong_token_is_401(headers):
    response = _app("secret").get("/mcp", headers=headers)
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}


def test_right_token_passes():
    assert (
        _app("secret")
        .get("/mcp", headers={"Authorization": "Bearer secret"})
        .status_code
        == 200
    )


def test_empty_token_disables_the_check():
    assert _app("").get("/mcp").status_code == 200


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("localhost", True),
        ("127.0.0.1", True),
        ("::1", True),
        ("0.0.0.0", False),
        ("10.0.0.5", False),
        ("example.com", False),
    ],
)
def test_is_loopback(host, expected):
    assert is_loopback(host) is expected
