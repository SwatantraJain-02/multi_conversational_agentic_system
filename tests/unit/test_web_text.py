"""Pure helpers of the Web agent: URLs, chunks, the context budget, citations."""

import pytest

from multi_agenti_conversation_system.agents.web.text import (
    CHUNK_CHARS,
    CHUNK_OVERLAP_CHARS,
    build_context,
    chunk_text,
    normalize_url,
    strip_invalid_citations,
)


@pytest.mark.parametrize(
    "a,b",
    [
        ("https://Example.com/a/?utm_source=x&id=2", "https://example.com/a?id=2"),
        ("https://example.com/a#section", "https://example.com/a"),
        ("https://example.com/a?b=1&a=2", "https://example.com/a?a=2&b=1"),
        ("https://example.com", "https://example.com/"),
        ("https://example.com/a?fbclid=zzz", "https://example.com/a"),
    ],
)
def test_url_variants_collapse_to_one_key(a, b):
    assert normalize_url(a) == normalize_url(b)


def test_different_pages_stay_different():
    assert normalize_url("https://example.com/a?id=1") != normalize_url(
        "https://example.com/a?id=2"
    )
    assert normalize_url("http://example.com/a") != normalize_url(
        "https://example.com/a"
    )


def test_long_text_is_chunked_with_overlap():
    words = [f"word{i}" for i in range(2500)]
    chunks = chunk_text(" ".join(words))
    assert len(chunks) >= 3
    assert all(len(c) <= CHUNK_CHARS for c in chunks)
    # consecutive chunks share text (the overlap), so a fact split at a boundary survives
    tail = chunks[0][-CHUNK_OVERLAP_CHARS // 2 :]
    assert tail.split()[-1] in chunks[1]


def test_short_text_is_one_chunk_and_tiny_fragments_are_dropped():
    assert len(chunk_text("A sentence about beverages. " * 10)) == 1
    assert chunk_text("too short") == []


def chunk(url, text, **kw):
    return {"url": url, "text": text, "title": kw.get("title"), "snippet_only": False}


def test_sources_are_numbered_by_first_appearance_and_share_excerpts():
    chunks = [chunk("u2", "x" * 300), chunk("u1", "y" * 300), chunk("u2", "z" * 300)]
    sources, used = build_context(chunks, budget_chars=10_000)
    assert [(s.n, s.url, len(s.excerpts)) for s in sources] == [
        (1, "u2", 2),
        (2, "u1", 1),
    ]
    assert used == 900


def test_the_budget_is_respected_and_the_last_excerpt_is_cut():
    chunks = [chunk("u1", "word " * 300), chunk("u2", "word " * 300)]  # 1500 chars each
    sources, used = build_context(chunks, budget_chars=2000)
    assert used <= 2000 + len(" ...")
    assert sources[1].excerpts[0].endswith("...")


def test_a_remainder_too_small_to_be_useful_is_dropped():
    sources, used = build_context(
        [chunk("u1", "a" * 1000), chunk("u2", "b" * 500)], 1100
    )
    assert [s.url for s in sources] == ["u1"] and used == 1000


@pytest.mark.parametrize(
    "answer,valid,expected,kept",
    [
        (
            "Sales grew [1] and fell [2].",
            {1, 2},
            "Sales grew [1] and fell [2].",
            {1, 2},
        ),
        ("Sales grew [7].", {1, 2}, "Sales grew.", set()),
        ("Both agree [1, 9].", {1, 2}, "Both agree [1].", {1}),
        ("Mixed [3][1] here.", {1}, "Mixed [1] here.", {1}),
        (
            "Odd [x] and [] and [1.5] stay.",
            {1},
            "Odd [x] and [] and [1.5] stay.",
            set(),
        ),
        ("No citations at all.", {1}, "No citations at all.", set()),
        ("Zero [0] is not a source.", {1}, "Zero is not a source.", set()),
        ("Many [1,2,3] sources.", {2}, "Many [2] sources.", {2}),
    ],
)
def test_citations_without_a_source_are_removed_in_code(answer, valid, expected, kept):
    cleaned, found = strip_invalid_citations(answer, valid)
    assert cleaned == expected and found == kept
