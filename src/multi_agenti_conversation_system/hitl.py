"""Human-in-the-loop pauses, shared by the supervisor and every agent.

A pause is a LangGraph `interrupt(payload)`. Across A2A it becomes an `input-required` task
whose `TaskResult.data` is the payload; the supervisor passes it to the UI and sends the
human's reply back on the same task. Two kinds:

- approval: "may I run this?" (the SQL agent, for personal data or a large result).
  Payload {"kind": "approval", "reason", ...}; reply {"approved": bool}.
- clarify:  "I need one detail to answer this" (the planner, the SQL and Web agents; the
  Synthesis agent relays its Web agent's question). Payload {"kind": "clarify", "question",
  "options", "reason"}; reply {"answer": str}.

Rules every asker follows:
- the interrupt is alone in its node (resuming re-runs the whole node, G22);
- at most MAX_CLARIFICATIONS per task (per turn for the planner); after that, or when the
  reply is empty or declined, it answers anyway and says what it assumed;
- the reply is the user's text: the supervisor runs it through the input guard before any
  agent or prompt sees it, and it only enters a prompt through `with_clarification`.
"""

from typing import Any, Literal

APPROVAL: Literal["approval"] = "approval"
CLARIFY: Literal["clarify"] = "clarify"

MAX_CLARIFICATIONS = 1
MAX_QUESTION_CHARS = 300
MAX_OPTIONS = 5
MAX_OPTION_CHARS = 80
MAX_ANSWER_CHARS = 500

NO_ANSWER = (
    "No answer was given: make a reasonable assumption and state it in the answer."
)


def kind_of(payload: dict[str, Any]) -> str:
    """The kind of a pause. Payloads from before `kind` existed were approvals."""
    return CLARIFY if payload.get("kind") == CLARIFY else APPROVAL


def clarify_payload(question: str, options: list[str] | None = None) -> dict[str, Any]:
    """The interrupt payload of a clarifying question (sizes capped)."""
    question = " ".join(question.split())[:MAX_QUESTION_CHARS]
    cleaned = [" ".join(o.split())[:MAX_OPTION_CHARS] for o in options or []]
    return {
        "kind": CLARIFY,
        "question": question,
        "options": [o for o in dict.fromkeys(cleaned) if o][:MAX_OPTIONS],
        "reason": question,
    }


def answer_of(decision: Any) -> str:
    """The text of a clarify reply; "" for anything else (no reply, a decline, junk)."""
    if not isinstance(decision, dict):
        return ""
    answer = decision.get("answer")
    return (
        " ".join(answer.split())[:MAX_ANSWER_CHARS] if isinstance(answer, str) else ""
    )


def with_clarification(question: str, asked: str, answer: str) -> str:
    """The question with the exchange appended, for the next attempt's prompt."""
    if not answer:
        return f"{question}\n(You asked: {asked} {NO_ANSWER})"
    return f"{question}\n(You asked: {asked} The user answered: {answer})"
