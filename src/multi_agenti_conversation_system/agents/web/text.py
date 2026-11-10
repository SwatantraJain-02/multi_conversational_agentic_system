"""Pure helpers of the Web agent: URLs, chunks, a context budget, citations. No I/O."""

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from langchain_text_splitters import RecursiveCharacterTextSplitter

# About 800 tokens per chunk with 100 overlap (roughly 4 characters per token).
CHUNK_CHARS = 3200
CHUNK_OVERLAP_CHARS = 400
MIN_CHUNK_CHARS = 80  # a page fragment shorter than this is noise
MIN_SNIPPET_CHARS = 30  # a search snippet is short by nature: only drop the empty ones

_TRACKING = re.compile(r"(?i)^(utm_.*|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|igshid)$")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def normalize_url(url: str) -> str:
    """One key for the same page: no fragment, tracking params or trailing slash, sorted query."""
    parts = urlsplit(url.strip())
    query = urlencode(
        sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k))
    )
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_CHARS, chunk_overlap=CHUNK_OVERLAP_CHARS
    )
    return [
        c.strip()
        for c in splitter.split_text(text)
        if len(c.strip()) >= MIN_CHUNK_CHARS
    ]


@dataclass
class Source:
    n: int
    url: str
    title: str
    published_date: str | None
    snippet_only: bool
    excerpts: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "url": self.url,
            "title": self.title,
            "published_date": self.published_date,
        }


def build_context(
    chunks: list[dict[str, Any]], budget_chars: int
) -> tuple[list[Source], int]:
    """Number the sources and fit their excerpts into the budget.

    `chunks` arrive best first. Sources are numbered by first appearance, each excerpt is
    kept whole while it fits, and the last one is cut to the space left (dropped if less
    than a useful remainder). Returns the sources and the characters used.
    """
    sources: dict[str, Source] = {}
    used = 0
    for chunk in chunks:
        room = budget_chars - used
        text = chunk["text"]
        if len(text) > room:
            if room < 200:
                continue
            text = text[:room].rsplit(" ", 1)[0] + " ..."
        source = sources.setdefault(
            chunk["url"],
            Source(
                n=len(sources) + 1,
                url=chunk["url"],
                title=chunk.get("title") or chunk["url"],
                published_date=chunk.get("published_date"),
                snippet_only=bool(chunk.get("snippet_only")),
                excerpts=[],
            ),
        )
        source.excerpts.append(text)
        used += len(text)
    return list(sources.values()), used


def strip_invalid_citations(answer: str, valid: set[int]) -> tuple[str, set[int]]:
    """Remove every [n] that doesn't point at a source we actually have.

    Done in code, not by asking the model: it will sometimes cite [7] when there are three
    sources. `[1, 9]` keeps the 1; a bracket with no valid number disappears. Returns the
    cleaned answer and the set of citation numbers that remain.
    """
    kept: set[int] = set()

    def fix(match: re.Match) -> str:
        numbers = [int(n) for n in re.split(r"\s*,\s*", match.group(1))]
        good = [n for n in numbers if n in valid]
        kept.update(good)
        return f"[{', '.join(map(str, good))}]" if good else ""

    cleaned = _CITATION.sub(fix, answer)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # "word [7] ." -> "word."
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    return cleaned.strip(), kept
