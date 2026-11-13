import httpx
import pytest

from multi_agenti_conversation_system.mcp.servers.web_server import fetch
from multi_agenti_conversation_system.mcp.servers.web_server.rate_limit import (
    TokenBucket,
)

# Distinct paragraphs: trafilatura drops repeated ones.
ARTICLE = (
    "<html><head><title>Beverage market 2026</title></head><body><article>"
    + "".join(
        f"<p>Paragraph {i}: the global beverage market grew steadily in region {i}, "
        f"driven by demand for low-sugar drinks and premium coffee.</p>"
        for i in range(10)
    )
    + "</article></body></html>"
)

# Fake DNS: public names resolve to public addresses, internal ones to private ranges.
DNS = {
    "example.com": ["93.184.215.14"],
    "news.example.com": ["93.184.215.15"],
    "internal.example.com": ["10.0.0.5"],
    "mixed.example.com": ["93.184.215.16", "127.0.0.1"],
}


def resolver(host: str, port: int) -> list[str]:
    return DNS.get(host, [])


def transport(routes: dict[str, httpx.Response]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return routes.get(str(request.url), httpx.Response(404))

    return httpx.MockTransport(handler)


def html(body: str = ARTICLE, **headers) -> httpx.Response:
    return httpx.Response(
        200, text=body, headers={"content-type": "text/html", **headers}
    )


# ----------------------------------------------------------------- IP checks
@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "10.1.2.3",
        "172.16.0.1",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "0.0.0.0",
        "::1",
        "fe80::1",
        "fc00::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "224.0.0.1",
    ],
)
def test_non_public_addresses_are_blocked(address):
    assert not fetch.is_public_ip(address)


@pytest.mark.parametrize("address", ["93.184.215.14", "8.8.8.8", "2606:4700::1111"])
def test_public_addresses_are_allowed(address):
    assert fetch.is_public_ip(address)


# ----------------------------------------------------------------- URL checks
@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "http://user:pw@example.com/",
        "http:///nohost",
        "http://internal.example.com/",
        "http://mixed.example.com/",  # one bad address is enough
        "http://127.0.0.1/",
        "http://[::1]/",
    ],
)
def test_check_url_blocks(url):
    def literal_aware(host, port):
        return DNS.get(host) or [host]

    with pytest.raises(fetch.FetchBlocked):
        fetch.check_url(url, literal_aware)


def test_check_url_allows_public_https():
    fetch.check_url("https://example.com/report", resolver)


# ------------------------------------------------------------------- download
def test_fetch_extracts_main_text():
    page = fetch.fetch_page(
        "https://example.com/a",
        max_chars=20_000,
        resolver=resolver,
        transport=transport({"https://example.com/a": html()}),
    )
    assert "global beverage market" in page.text
    assert page.title == "Beverage market 2026"
    assert page.extracted_by == "trafilatura"
    assert not page.truncated


def test_redirect_to_public_host_is_followed():
    routes = {
        "https://example.com/old": httpx.Response(
            301, headers={"location": "https://news.example.com/new"}
        ),
        "https://news.example.com/new": html(),
    }
    page = fetch.fetch_page(
        "https://example.com/old",
        20_000,
        resolver=resolver,
        transport=transport(routes),
    )
    assert page.final_url == "https://news.example.com/new"


def test_redirect_to_private_host_is_blocked():
    routes = {
        "https://example.com/bounce": httpx.Response(
            302, headers={"location": "http://internal.example.com/admin"}
        )
    }
    with pytest.raises(fetch.FetchBlocked, match="non-public"):
        fetch.download("https://example.com/bounce", resolver, transport(routes))


def test_too_many_redirects():
    routes = {
        "https://example.com/loop": httpx.Response(
            302, headers={"location": "https://example.com/loop"}
        )
    }
    with pytest.raises(fetch.FetchBlocked, match="redirects"):
        fetch.download("https://example.com/loop", resolver, transport(routes))


def test_non_html_is_rejected():
    routes = {
        "https://example.com/file.pdf": httpx.Response(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )
    }
    with pytest.raises(fetch.FetchFailed, match="content type"):
        fetch.download("https://example.com/file.pdf", resolver, transport(routes))


def test_http_error_status_is_reported():
    with pytest.raises(fetch.FetchFailed, match="HTTP 404"):
        fetch.download("https://example.com/missing", resolver, transport({}))


def test_body_is_capped(monkeypatch):
    monkeypatch.setattr(fetch, "MAX_BYTES", 1000)
    routes = {"https://example.com/big": html("x" * 5000)}
    _, body, capped = fetch.download(
        "https://example.com/big", resolver, transport(routes)
    )
    assert capped and len(body) == 1000


def test_text_is_truncated_to_max_chars():
    page = fetch.fetch_page(
        "https://example.com/a",
        max_chars=100,
        resolver=resolver,
        transport=transport({"https://example.com/a": html()}),
    )
    assert len(page.text) == 100 and page.truncated


def test_thin_page_falls_back_to_tavily():
    thin = "<html><body><div id='app'></div><script>render()</script></body></html>"
    page = fetch.fetch_page(
        "https://example.com/spa",
        20_000,
        tavily_extract=lambda url: "Rendered text from Tavily. " * 20,
        resolver=resolver,
        transport=transport({"https://example.com/spa": html(thin)}),
    )
    assert page.extracted_by == "tavily"
    assert page.text.startswith("Rendered text from Tavily.")


def test_empty_page_without_fallback_fails():
    with pytest.raises(fetch.FetchFailed, match="no readable text"):
        fetch.fetch_page(
            "https://example.com/empty",
            20_000,
            resolver=resolver,
            transport=transport({"https://example.com/empty": html("<html></html>")}),
        )


# ----------------------------------------------------------------- rate limit
class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_bucket_allows_burst_then_limits():
    clock = FakeClock()
    bucket = TokenBucket(per_minute=60, burst=3, clock=clock)
    assert [bucket.try_acquire()[0] for _ in range(4)] == [True, True, True, False]


def test_bucket_reports_retry_after_and_refills():
    clock = FakeClock()
    bucket = TokenBucket(per_minute=60, burst=1, clock=clock)  # 1 token per second
    assert bucket.try_acquire() == (True, 0.0)
    allowed, retry_after = bucket.try_acquire()
    assert not allowed and retry_after == 1.0
    clock.now = 1.0
    assert bucket.try_acquire()[0]


def test_bucket_never_exceeds_capacity():
    clock = FakeClock()
    bucket = TokenBucket(per_minute=60, burst=2, clock=clock)
    clock.now = 1000.0
    assert [bucket.try_acquire()[0] for _ in range(3)] == [True, True, False]


def test_bucket_rejects_bad_config():
    with pytest.raises(ValueError):
        TokenBucket(per_minute=0, burst=1)
