"""The memory write-gate: what may be written to long-term memory.

Everything the extraction model proposes passes through here before a Pinecone or store write:

1. **No personal data or secrets** in the stored text or value (a user who types their email
   address while stating a preference must not get the address saved).
2. **Grounded in the user's own words.** Each significant word of the value must appear in
   the user's message. A free model that "infers" (the question mentions APAC, so the user
   must manage APAC) is caught here: only what the user said about themselves is stored.
3. **Nothing from a flagged turn** (suspected injection, PII in the message): the caller
   skips learning entirely.

Tool output is never an input to extraction (it only sees the user's message), so memory
cannot be poisoned through a web page the agent read.
"""

import re

from multi_agenti_conversation_system.guardrails.pii import redact_pii, redact_secrets

_WORD = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "and",
        "or",
        "is",
        "are",
        "am",
        "was",
        "be",
        "i",
        "my",
        "me",
        "we",
        "our",
        "with",
        "as",
        "by",
    ]
)


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def tokens(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower()) if w not in _STOP}


def grounded(candidate: str, message: str, minimum: float = 1.0) -> bool:
    """True when at least `minimum` of the candidate's significant words are in the message."""
    wanted = tokens(candidate)
    if not wanted:
        return False
    return len(wanted & tokens(message)) / len(wanted) >= minimum


def clean_of_pii(*texts: str) -> bool:
    return all(redact_pii(redact_secrets(t)) == t for t in texts)


def may_store(message: str, *candidates: str, text: str = "") -> tuple[bool, str]:
    """Whether a preference or fact (its value `candidates`, and its `text`) may be stored."""
    values = [c for c in candidates if c]
    if not clean_of_pii(*values, text):
        return False, "contains personal data or a secret"
    if not values or not all(grounded(v, message) for v in values):
        return False, "not stated by the user"
    return True, ""
