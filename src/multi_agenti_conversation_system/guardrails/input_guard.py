"""The input guard: the first thing a user message meets, before any memory or model call.

Three independent checks, cheapest first:

1. **Limits.** Empty or over-long messages are blocked (the A2A contract caps a task at 4000
   characters; a longer message could never be routed anyway). NUL bytes are stripped.
2. **Injection heuristics.** Regexes for the well-known shapes ("ignore previous
   instructions", "reveal your system prompt", "developer mode", "disable your guardrails",
   chat-template control tokens, asking for keys or passwords). A hit does NOT block by
   itself: heuristics have false positives ("ignore the previous query and show 2025").
   It sends the message to the model classifier (`guard/input` prompt), which decides.
   No hit, no model call: most messages cost nothing here.
3. **PII flagging.** Emails, SSNs, Luhn-valid card numbers and phone numbers are listed in the
   verdict. They never block (users may legitimately mention them) but they switch off
   memory learning for the turn, so personal data is not written to long-term memory.

If the classifier is unavailable, a heuristic hit is allowed but `flagged`: learning is off and
the turn is marked. Failing closed would turn a model outage into a refusal of every message
that happens to contain "ignore"; the layers behind this one (read-only SQL validator, the
read-only DB role, human approval, the A2A contracts) bound what an injected instruction
could do. What gets through a regex gap (obfuscated text) is exactly why they exist.
"""

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel

from multi_agenti_conversation_system.guardrails.pii import redact_pii

logger = logging.getLogger(__name__)

MAX_INPUT_CHARS = 4000
Label = Literal["safe", "injection", "jailbreak", "exfiltration"]

INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\b(ignore|disregard|forget|override)\b.{0,30}\b(previous|prior|above|earlier|all|your|these)\b.{0,30}\b(instructions?|prompts?|rules|messages|guidelines)\b",
        r"\b(reveal|show|print|repeat|output|display|leak)\b.{0,25}\b(system|hidden|initial|original|secret)\b.{0,15}\b(prompt|instructions?|message)\b",
        r"\byou are now\b",
        r"\b(developer|god|dan|jailbreak(ed)?|unrestricted) mode\b",
        r"\bdo anything now\b",
        r"\bjailbreak\b",
        r"\bact as\b.{0,20}\b(unrestricted|unfiltered|uncensored|evil)\b",
        r"\bpretend\b.{0,30}\bno (rules|restrictions|guardrails|limits)\b",
        r"\b(bypass|disable|turn off|ignore)\b.{0,15}\b(guardrails?|safety|safeguards?|filters?|restrictions?|validation)\b",
        r"<\|?\s*(system|im_start|im_end|endoftext)\s*\|?>",
        r"\b(begin|end) (of )?system prompt\b",
        r"\b(show|print|reveal|give|tell|send)\b.{0,20}\b(api[ _-]?keys?|passwords?|secrets?|tokens?|credentials)\b",
    )
]


class Classification(BaseModel):
    """What the model classifier returns. Both fields required (free models drop defaults)."""

    label: Label
    reason: str


Classifier = Callable[[list[dict[str, str]]], Awaitable[Classification]]


@dataclass(frozen=True)
class InputVerdict:
    action: Literal["allow", "block"]
    text: str  # the cleaned message
    flagged: bool = (
        False  # suspicious but allowed, or PII present: no learning this turn
    )
    reasons: list[str] = field(default_factory=list)
    pii: list[str] = field(default_factory=list)  # kinds: email, phone, card, ssn
    label: str = "safe"

    @property
    def refusal(self) -> str:
        if "empty" in self.reasons:
            return "I didn't get a message. What would you like to know?"
        if "too_long" in self.reasons:
            return (
                f"That message is too long (over {MAX_INPUT_CHARS} characters). "
                "Please shorten it or split it into smaller questions."
            )
        return (
            "I can't help with that request, because it asks me to ignore my instructions "
            "or reveal internal details. I can answer questions about our sales data and "
            "about public information from the web. What would you like to know?"
        )


def pii_kinds(text: str) -> list[str]:
    """Kinds of personal data in the text, e.g. ["email", "phone"]."""
    return sorted(set(re.findall(r"\[REDACTED:(\w+)\]", redact_pii(text))))


def heuristic_hits(text: str) -> list[str]:
    return [p.pattern[:40] for p in INJECTION_PATTERNS if p.search(text)]


async def check_input(
    text: str, classify: Classifier | None = None, prompt=None
) -> InputVerdict:
    """The verdict on one user message. Never raises."""
    clean = text.replace("\x00", "").strip()
    if not clean:
        return InputVerdict("block", clean, reasons=["empty"])
    if len(clean) > MAX_INPUT_CHARS:
        return InputVerdict("block", clean, reasons=["too_long"])

    pii = pii_kinds(clean)
    hits = heuristic_hits(clean)
    if not hits:
        return InputVerdict("allow", clean, flagged=bool(pii), pii=pii)

    reasons = [f"heuristic: {h}" for h in hits]
    if classify is None or prompt is None:
        return InputVerdict("allow", clean, flagged=True, reasons=reasons, pii=pii)
    try:
        result = await classify(prompt.compile(message=clean))
    except Exception as exc:  # noqa: BLE001 - see the module docstring: allow, flagged
        logger.warning("input classifier failed (%s): allowed, flagged", exc)
        return InputVerdict(
            "allow",
            clean,
            flagged=True,
            reasons=[*reasons, "classifier_unavailable"],
            pii=pii,
        )
    if result.label != "safe":
        return InputVerdict(
            "block",
            clean,
            flagged=True,
            reasons=[*reasons, f"classifier: {result.label}: {result.reason[:120]}"],
            pii=pii,
            label=result.label,
        )
    return InputVerdict(
        "allow",
        clean,
        flagged=bool(pii),
        reasons=[*reasons, "classifier: safe"],
        pii=pii,
    )
