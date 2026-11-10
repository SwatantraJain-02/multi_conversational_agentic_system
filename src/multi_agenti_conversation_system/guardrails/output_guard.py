"""The output guard: the last look at an answer before the user (and the cache) gets it.

- **Secrets are always redacted** (API keys, bearer tokens, configured credentials): nothing
  legitimate ever needs to show one.
- **Personal data is redacted unless a human approved the query that fetched it.** An answer
  that lists contact names and phone numbers after the user approved exactly that query is
  the feature working; the same text from a web page, or from a model that made it up, is a
  leak. (Names have no regex: the real protection for names is the SQL agent's PII check and
  the approval gate; this guard catches the machine-readable kinds: email, phone, card, SSN.)
- **Citation markers need a source.** `[3]` with no citation number 3 would point at nothing,
  so the marker is removed. The agents already strip invalid markers; this checks again at
  the boundary, because Synthesis and the supervisor compose text from several of them.
"""

import re
from typing import Any

from multi_agenti_conversation_system.guardrails.pii import redact_pii, redact_secrets

MARKER = re.compile(r"\s?\[(\d{1,3})\]")


def strip_dangling_markers(answer: str, citations: list[dict[str, Any]]) -> str:
    known = {c.get("n") for c in citations}
    return MARKER.sub(lambda m: m.group() if int(m.group(1)) in known else "", answer)


def guard_output(
    answer: str, citations: list[dict[str, Any]], *, pii_approved: bool
) -> str:
    answer = redact_secrets(answer)
    if not pii_approved:
        answer = redact_pii(answer)
    return strip_dangling_markers(answer, citations)
