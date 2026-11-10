"""LLM-as-judge: scores an answer on five criteria, 1 to 5, each with a reason.

Used for what deterministic checks cannot say (is this summary relevant, clear, grounded?).
The judge sees the question, optional reference notes, the cited sources and the answer; it
never sees the SQL or how the answer was produced. It uses `JUDGE_MODEL` (or the chat model),
and its prompt is the Langfuse prompt `eval/judge`, so judge changes are versioned too.
A failed judgement is skipped, not scored 0: a broken judge must not look like a bad answer.
"""

import logging
from typing import Any

from pydantic import BaseModel

from multi_agenti_conversation_system.agents.llm_calls import structured
from multi_agenti_conversation_system.config import settings
from multi_agenti_conversation_system.prompts import aget_prompt

logger = logging.getLogger(__name__)
CRITERIA = ("relevance", "correctness", "groundedness", "completeness", "clarity")


class Criterion(BaseModel):
    score: int
    reason: str


class Judgement(BaseModel):
    relevance: Criterion
    correctness: Criterion
    groundedness: Criterion
    completeness: Criterion
    clarity: Criterion


def judge_model() -> Any:
    from langchain_openrouter import ChatOpenRouter

    return ChatOpenRouter(
        api_key=settings.OPENROUTER_API_KEY,
        model=settings.JUDGE_MODEL or settings.CHAT_MODEL,
        temperature=0,
        max_retries=4,
    )


class Judge:
    def __init__(self, call=None) -> None:
        self._call = call

    @property
    def call(self):
        if self._call is None:
            self._call = structured(judge_model(), Judgement)
        return self._call

    async def judge(
        self,
        question: str,
        answer: str,
        sources: list[dict[str, Any]],
        expected: str = "",
    ) -> list[tuple[str, float, str]] | None:
        """[(criterion, score 1-5, reason)], or None when the judgement failed."""
        if not answer.strip():
            return None
        prompt = await aget_prompt("eval/judge")
        listed = "; ".join(
            f"[{s.get('n')}] {s.get('title') or s.get('url')}" for s in sources
        )
        try:
            result: Judgement = await self.call(
                prompt.compile(
                    question=question,
                    answer=answer[:6000],
                    sources=listed or "(none)",
                    expected=expected or "(none)",
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge failed (%s): item not judged", exc)
            return None
        return [
            (
                f"judge_{name}",
                float(min(5, max(1, getattr(result, name).score))),
                getattr(result, name).reason,
            )
            for name in CRITERIA
        ]
