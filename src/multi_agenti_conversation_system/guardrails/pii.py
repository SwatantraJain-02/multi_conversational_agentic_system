"""Redaction of secrets and personal data in free text.

First user: the Langfuse mask (`tracing.mask`), so nothing sensitive reaches a trace. The
Phase I guardrails (input flagging, output redaction, the memory write-gate) build on the
same patterns.

Regexes, not an NLP model, so this is fast and has no heavy dependency. It is deliberately
conservative about what looks like a phone number: dates (2024-07-04), ids, decimals
(1234567.89) and IPv4 addresses must survive, because traces full of `[REDACTED]` order
dates and prices would be useless. A phone number is only recognised when it has
separators, a `+`, or brackets, so a bare 11-digit run is *not* redacted (documented gap).
"""

import re
from collections.abc import Iterable
from functools import cache
from typing import Any

from multi_agenti_conversation_system.config import settings

# Short values would corrupt unrelated text when replaced literally (an empty one would
# match everywhere), so only substantial secrets are replaced by value.
MIN_SECRET_CHARS = 8
SECRET_FIELD_HINTS = ("TOKEN", "KEY", "PASSWORD", "SECRET")

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
SSN = re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)")
CARD = re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])")
# A candidate run of digits and phone separators; `_looks_like_phone` then filters it.
PHONE_CANDIDATE = re.compile(r"(?<![\w-])[+(]?\d[\d ()./-]{5,}\d(?![\w-])")
_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_IPV4 = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")

# Credentials by shape, whether or not they are configured here.
SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\b(?:sk|pk)-[A-Za-z0-9_-]{16,}"),  # OpenRouter, Langfuse
    re.compile(r"\btvly-[A-Za-z0-9_-]{10,}"),  # Tavily
    re.compile(r"\bpcsk_[A-Za-z0-9_]{10,}"),  # Pinecone
)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2:
            n = n * 2 - 9 if n > 4 else n * 2
        total += n
    return total % 10 == 0


def _redact_card(match: re.Match) -> str:
    digits = re.sub(r"\D", "", match.group())
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "[REDACTED:card]"
    return match.group()


def _looks_like_phone(text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if not 7 <= len(digits) <= 15:
        return False
    if _ISO_DATE.match(text) or _IPV4.fullmatch(text):
        return False
    # Pure digits and dots is a decimal or an id, not a phone number.
    return any(ch in text for ch in "+( -")


def _redact_phone(match: re.Match) -> str:
    return "[REDACTED:phone]" if _looks_like_phone(match.group()) else match.group()


@cache
def configured_secrets() -> list[str]:
    """Secret values from settings (tokens, keys, passwords), longest first.

    Cached: the mask runs on every string of every span, and settings don't change while
    the process runs (tests that change them call `configured_secrets.cache_clear()`).
    """
    values = {
        str(value)
        for name, value in settings.model_dump().items()
        if any(hint in name for hint in SECRET_FIELD_HINTS)
        and isinstance(value, str)
        and len(value) >= MIN_SECRET_CHARS
    }
    return sorted(values, key=len, reverse=True)


def redact_secrets(text: str, extra: Iterable[str] = ()) -> str:
    for secret in (*configured_secrets(), *extra):
        if len(secret) >= MIN_SECRET_CHARS:
            text = text.replace(secret, "[REDACTED:secret]")
    for pattern in SECRET_PATTERNS:
        text = pattern.sub("[REDACTED:secret]", text)
    return text


def redact_pii(text: str) -> str:
    text = EMAIL.sub("[REDACTED:email]", text)
    text = SSN.sub("[REDACTED:ssn]", text)
    text = CARD.sub(_redact_card, text)
    return PHONE_CANDIDATE.sub(_redact_phone, text)


def redact(text: str) -> str:
    return redact_pii(redact_secrets(text))


def mask(data: Any, **_: Any) -> Any:
    """Langfuse mask hook: redact every string inside `data`, never raise.

    A failure in here would lose the trace (or the whole export), so on any error the
    value is replaced by a placeholder instead.
    """
    try:
        return _walk(data)
    except Exception:  # noqa: BLE001 - masking must never break tracing
        return "[REDACTED:mask-error]"


def _walk(value: Any) -> Any:
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {key: _walk(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_walk(item) for item in value]
    return value
