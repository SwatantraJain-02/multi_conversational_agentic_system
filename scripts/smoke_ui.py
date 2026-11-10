"""Phase J live gate: drive the real Streamlit script with Streamlit's own test runner.

    uv run python scripts/run_all.py        # in another terminal: services must be up
    uv run python scripts/smoke_ui.py

`AppTest` runs `src/ui/streamlit_app.py` in-process, so this exercises the real page code
(the background event loop, progress streaming, the approval card, the memory panel, the
feedback buttons) against the real services, model, Pinecone, Aiven and Langfuse. It does not
check how the page looks: for that, open it (`run_all.py --ui`).

Flows: a SQL answer with its details; a PII question that pauses for approval and runs after
Approve; a stated preference and fact that show up in the memory panel; a New chat that still
recalls them; a thumbs-down. Everything the run wrote for its unique user is deleted at the end.
"""

import asyncio
import sys
import time
import uuid

from streamlit.testing.v1 import AppTest

from multi_agenti_conversation_system import tracing, vectorstore
from multi_agenti_conversation_system.memory.checkpointer import open_memory
from multi_agenti_conversation_system.memory.episodes import EpisodeStore
from multi_agenti_conversation_system.memory.facts import FactStore
from multi_agenti_conversation_system.memory.profile import ProfileStore

PAGE = "src/ui/streamlit_app.py"
USER = f"ui-smoke-{uuid.uuid4().hex[:6]}"
QUESTIONS = {
    "sql": "How many customers do we have in each country?",
    "pii": "List the phone numbers of our customers in France",
    "tell": "From now on please give me short answers. I manage the APAC region.",
    "recall": "Which region do I manage?",
}
POLL_S = 90
failures: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(
        f"  {'PASS' if condition else 'FAIL'}  {name}"
        + (f"  ({detail})" if detail else "")
    )
    if not condition:
        failures.append(name)


def last_answer(at: AppTest) -> str:
    return " ".join(m.value for m in at.chat_message[-1].markdown)


def labels(widgets) -> list[str]:
    return [w.label for w in widgets]


def main() -> None:
    at = AppTest.from_file(
        str(__import__("pathlib").Path(PAGE).resolve()), default_timeout=400
    )
    at.run()
    check(
        "the page starts without an exception",
        not at.exception,
        str([e.value for e in at.exception]),
    )
    at.sidebar.text_input[0].set_value(USER).run()

    print("\n1. a SQL question")
    at.chat_input[0].set_value(QUESTIONS["sql"]).run()
    check("no error on the page", not at.error and not at.exception)
    check(
        "an answer is shown",
        len(at.chat_message) == 2 and len(last_answer(at)) > 20,
        last_answer(at)[:80],
    )
    check(
        "its details are available",
        {"SQL used", "Prompts and trace"} <= set(labels(at.expander)),
        str(labels(at.expander)),
    )
    check("it has feedback buttons", {"👍", "👎"} <= set(labels(at.button)))

    print("\n2. a question that needs approval")
    at.chat_input[0].set_value(QUESTIONS["pii"]).run()
    check(
        "an approval card appears with the PII column",
        bool(at.warning) and "phone" in at.warning[0].value,
        str([w.value for w in at.warning]),
    )
    check(
        "Approve and Reject are offered",
        {"Approve", "Reject"} <= set(labels(at.button)),
    )
    check(
        "the SQL to approve is shown",
        any("phone" in c.value.lower() for c in at.code),
    )
    next(b for b in at.button if b.label == "Approve").click().run()
    check(
        "after Approve the answer arrives",
        "France" in last_answer(at) or "phone" in last_answer(at).lower(),
        last_answer(at)[:80],
    )

    print("\n3. memory: tell it something, see it in the panel")
    at.chat_input[0].set_value(QUESTIONS["tell"]).run()

    def panel() -> str:
        return " ".join(m.value for m in at.sidebar.markdown)

    deadline = time.monotonic() + POLL_S
    while time.monotonic() < deadline:  # a fetch by filter lags a write (G14)
        at.run()
        if "APAC" in panel():
            break
        time.sleep(3)
    check(
        "the preference is in the memory panel",
        "answer_length" in panel(),
        panel()[:100],
    )
    check(
        "the fact is in the memory panel, dated",
        "APAC" in panel() and "stated" in panel(),
        panel()[:160],
    )

    print("\n4. a new chat still recalls it")
    at.sidebar.button[labels(at.sidebar.button).index("New chat")].click().run()
    check("the new chat starts empty", len(at.chat_message) == 0)
    at.chat_input[0].set_value(QUESTIONS["recall"]).run()
    check("the region is recalled", "APAC" in last_answer(at), last_answer(at)[:80])

    print("\n5. feedback")
    next(b for b in at.button if b.label == "👎").click().run()
    check(
        "a thumbs-down is acknowledged",
        any("not be reused" in c.value for c in at.caption),
    )

    asyncio.run(cleanup())
    print(
        f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED: {failures}'}"
    )
    sys.exit(1 if failures else 0)


async def cleanup() -> None:
    await FactStore().delete_user(USER)
    await EpisodeStore().delete_user(USER)
    async with open_memory() as (_, store):
        await ProfileStore(store).reset(USER)
    for text in QUESTIONS.values():
        await asyncio.to_thread(
            vectorstore.delete_where,
            vectorstore.SEMANTIC_CACHE,
            {"text": {"$eq": text}},
        )
    await asyncio.to_thread(tracing.shutdown)
    print("\n  removed this run's facts, episodes, profile and cache entries")


if __name__ == "__main__":
    main()
