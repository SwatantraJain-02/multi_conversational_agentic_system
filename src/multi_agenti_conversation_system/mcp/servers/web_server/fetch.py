"""Fetch a public web page safely and extract its main text.

SSRF guard (server-side request forgery): the URL must be http(s) without credentials,
and every address its host resolves to must be public. Redirects are followed by hand,
at most MAX_REDIRECTS, and each hop is checked again, so a public page can't bounce the
server to 127.0.0.1 or the cloud metadata address (169.254.169.254).

Residual risk (documented, not solved): DNS rebinding. The host is resolved here and
again by httpx when it connects, so a hostile DNS server could answer differently the
second time. Pinning the connection to the checked IP would close that gap.

Fetched text is *data*, never instructions: the calling agent's prompt says so.
"""

import ipaddress
import socket
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura

MAX_REDIRECTS = 3
TIMEOUT_S = 10.0
MAX_BYTES = 2 * 1024 * 1024
MIN_EXTRACTED_CHARS = 200  # below this, the page is probably JS-rendered
ALLOWED_TYPES = ("text/html", "text/plain", "application/xhtml+xml")
USER_AGENT = "multi-agent-assistant/0.1 (capstone research bot)"

Resolver = Callable[[str, int], list[str]]


class FetchBlocked(Exception):
    """The URL (or a redirect target) is not allowed. The message is safe to show."""


class FetchFailed(Exception):
    """The page couldn't be fetched or read. The message is safe to show."""


@dataclass
class Page:
    url: str
    final_url: str
    title: str | None
    published_date: str | None
    text: str
    truncated: bool
    extracted_by: str  # "trafilatura" or "tavily"


def system_resolver(host: str, port: int) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise FetchFailed(f"could not resolve host '{host}'") from exc
    return sorted({info[4][0] for info in infos})


def is_public_ip(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])  # drop an IPv6 zone id
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def check_url(url: str, resolver: Resolver = system_resolver) -> None:
    """Raise FetchBlocked unless the URL is http(s) and resolves only to public IPs."""
    try:
        parts = urlsplit(url)
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as exc:
        raise FetchBlocked("malformed URL") from exc
    if parts.scheme not in ("http", "https"):
        raise FetchBlocked("only http and https URLs are allowed")
    if not parts.hostname:
        raise FetchBlocked("URL has no host")
    if parts.username or parts.password:
        raise FetchBlocked("URLs with credentials are not allowed")

    addresses = resolver(parts.hostname, port)
    if not addresses:
        raise FetchFailed(f"could not resolve host '{parts.hostname}'")
    blocked = [a for a in addresses if not is_public_ip(a)]
    if blocked:
        raise FetchBlocked(
            f"host '{parts.hostname}' resolves to a non-public address ({blocked[0]})"
        )


def _read_capped(response: httpx.Response) -> tuple[bytes, bool]:
    body, truncated = bytearray(), False
    for chunk in response.iter_bytes():
        body.extend(chunk)
        if len(body) > MAX_BYTES:
            del body[MAX_BYTES:]
            truncated = True
            break
    return bytes(body), truncated


def download(
    url: str,
    resolver: Resolver = system_resolver,
    transport: httpx.BaseTransport | None = None,
) -> tuple[str, bytes, bool]:
    """GET the URL, following and re-checking redirects. Returns (final_url, body, capped)."""
    with httpx.Client(
        follow_redirects=False,
        timeout=TIMEOUT_S,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    ) as client:
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            check_url(current, resolver)
            try:
                with client.stream("GET", current) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise FetchFailed("redirect without a Location header")
                        current = urljoin(current, location)
                        continue
                    if response.status_code != 200:
                        raise FetchFailed(f"HTTP {response.status_code}")
                    content_type = response.headers.get("content-type", "").lower()
                    if not content_type.startswith(ALLOWED_TYPES):
                        raise FetchFailed(
                            f"unsupported content type '{content_type or 'unknown'}'"
                        )
                    body, capped = _read_capped(response)
                    return current, body, capped
            except httpx.TimeoutException as exc:
                raise FetchFailed("timed out") from exc
            except httpx.HTTPError as exc:
                raise FetchFailed(f"request failed: {type(exc).__name__}") from exc
        raise FetchBlocked(f"more than {MAX_REDIRECTS} redirects")


def extract(html: str, url: str) -> tuple[str, str | None, str | None]:
    """Main text, title and date via trafilatura ('' when nothing useful is found)."""
    text = trafilatura.extract(html, url=url, include_comments=False) or ""
    meta = trafilatura.extract_metadata(html, default_url=url)
    title = meta.title if meta else None
    date = meta.date if meta else None
    return text.strip(), title, date


def fetch_page(
    url: str,
    max_chars: int,
    tavily_extract: Callable[[str], str | None] | None = None,
    resolver: Resolver = system_resolver,
    transport: httpx.BaseTransport | None = None,
) -> Page:
    """Download, extract and truncate a page. Falls back to Tavily for JS-heavy pages."""
    final_url, body, capped = download(url, resolver, transport)
    html = body.decode("utf-8", errors="replace")
    text, title, date = extract(html, final_url)
    extracted_by = "trafilatura"

    if len(text) < MIN_EXTRACTED_CHARS and tavily_extract is not None:
        fallback = tavily_extract(final_url)
        if fallback and len(fallback.strip()) > len(text):
            text, extracted_by = fallback.strip(), "tavily"
    if not text:
        raise FetchFailed("no readable text on the page")

    truncated = capped or len(text) > max_chars
    return Page(
        url=url,
        final_url=final_url,
        title=title,
        published_date=date,
        text=text[:max_chars],
        truncated=truncated,
        extracted_by=extracted_by,
    )
