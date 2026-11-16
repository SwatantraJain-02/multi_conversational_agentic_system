"""The Streamlit page's chat history, driven with AppTest and a fake supervisor (no services).

A chat is saved and listed per user, New chat keeps it, reopening restores its messages and
thread, and switching user ends the open chat under the user it belongs to.
"""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from langgraph.store.memory import InMemoryStore
from streamlit.testing.v1 import AppTest

from multi_agenti_conversation_system.memory.chats import ChatLog
from multi_agenti_conversation_system.supervisor import service

APP = str(Path(__file__).parents[2] / "src" / "ui" / "streamlit_app.py")


class FakeSupervisor:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.chats = ChatLog(InMemoryStore())
        self.ended: list[tuple[str, str]] = []
        self.script: list[service.TurnResult] = []  # replies to ask / resume, in order
        self.resumes: list[dict] = []

    async def ask(self, thread_id, user_id, text, *, domain=None, on_event=None):
        if self.script:
            return self.script.pop(0)
        return service.TurnResult(
            thread_id=thread_id, status="answered", answer=f"echo: {text}"
        )

    async def resume(self, thread_id, approved=None, *, answer=None, **kw):
        self.resumes.append({"approved": approved, "answer": answer})
        return self.script.pop(0)

    async def what_i_know(self, user_id):
        return {"profile": {}, "facts": []}

    async def end_thread(self, thread_id, user_id):
        self.ended.append((thread_id, user_id))

    list_chats = service.Supervisor.list_chats
    load_chat = service.Supervisor.load_chat
    save_chat = service.Supervisor.save_chat


# The page caches its backend for the whole process (`st.cache_resource`), so every test's
# page talks to this one fake, reset per test. (Clearing the cache instead would discard a
# live backend and its event loop mid-session.)
FAKE = FakeSupervisor()


@asynccontextmanager
async def open_fake():
    yield FAKE


@pytest.fixture
def page(monkeypatch):
    FAKE.reset()
    monkeypatch.setattr(service, "open_supervisor", open_fake)
    monkeypatch.setattr(service.settings, "EVAL_MODE", False)
    return AppTest.from_file(APP, default_timeout=30).run(), FAKE


def chat_buttons(at):
    return [b for b in at.sidebar.button if b.key and b.key.startswith("chat-")]


def click(at, label):
    next(b for b in at.sidebar.button if b.label == label).click().run()


def test_a_chat_is_listed_and_reopened_with_its_messages_and_thread(page):
    at, fake = page
    at.chat_input[0].set_value("How many customers per country?").run()
    thread = at.session_state["thread_id"]
    assert [b.label for b in chat_buttons(at)] == ["▶ How many customers per country?"]

    click(at, "New chat")
    assert at.session_state["messages"] == []
    assert fake.ended[-1] == (thread, "demo-user")

    chat_buttons(at)[0].click().run()
    assert at.session_state["thread_id"] == thread
    assert [m["content"] for m in at.session_state["messages"]] == [
        "How many customers per country?",
        "echo: How many customers per country?",
    ]
    # leaving a reopened chat that got no new turn writes no second episode
    before = len(fake.ended)
    click(at, "New chat")
    assert len(fake.ended) == before
    assert not at.exception


def test_switching_user_ends_the_chat_for_its_own_user_and_hides_it(page):
    at, fake = page
    at.chat_input[0].set_value("Top products?").run()
    thread = at.session_state["thread_id"]
    at.sidebar.text_input[0].set_value("bob").run()
    # ended for demo-user, whose chat it is, not for bob (the episode would leak to bob)
    assert fake.ended[-1] == (thread, "demo-user")
    assert chat_buttons(at) == []
    at.sidebar.text_input[0].set_value("demo-user").run()
    assert len(chat_buttons(at)) == 1 and not at.exception


# ------------------------------------------------------------- a clarifying question
def asking(question="Which currency should I convert the revenue to?"):
    return service.TurnResult(
        thread_id="t",
        status="needs_clarification",
        clarification={
            "kind": "clarify",
            "question": question,
            "options": ["EUR", "GBP"],
            "agent": "supervisor",
            "step_id": "",
        },
    )


def answered(text="Revenue was EUR 1.2M."):
    return service.TurnResult(thread_id="t", status="answered", answer=text)


def test_a_question_shows_its_options_and_an_option_is_the_answer(page):
    at, fake = page
    fake.script = [asking(), answered()]
    at.chat_input[0].set_value("Convert that revenue").run()
    assert at.info[0].value == "Which currency should I convert the revenue to?"
    assert at.chat_input[0].disabled  # no new message while it waits
    next(b for b in at.button if b.label == "EUR").click().run()
    assert fake.resumes == [{"approved": None, "answer": "EUR"}]
    assert [m["content"] for m in at.session_state["messages"]] == [
        "Convert that revenue",
        "Which currency should I convert the revenue to?",
        "EUR",
        "Revenue was EUR 1.2M.",
    ]
    assert at.session_state["pending"] is None and not at.exception


def test_a_typed_answer_or_skip_is_sent_too(page):
    at, fake = page
    fake.script = [asking(), answered(), asking("Which year?"), answered("2025.")]
    at.chat_input[0].set_value("Convert that revenue").run()
    form_input = next(t for t in at.text_input if t.label == "Your answer")
    form_input.set_value("Indian rupees")
    next(b for b in at.button if b.label == "Send").click().run()
    assert fake.resumes[-1]["answer"] == "Indian rupees"

    at.chat_input[0].set_value("And for last year?").run()
    next(b for b in at.button if b.label == "Skip: let it assume").click().run()
    assert fake.resumes[-1]["answer"] == ""
    assert at.session_state["messages"][-2]["content"] == "(skipped)"
    assert not at.exception


def test_a_question_followed_by_an_approval_shows_the_approval_card(page):
    at, fake = page
    approval = service.TurnResult(
        thread_id="t",
        status="needs_approval",
        approval={
            "kind": "approval",
            "reason": "Approval needed: PII.",
            "sql": "SELECT 1",
        },
    )
    fake.script = [asking(), approval, answered("Done.")]
    at.chat_input[0].set_value("List contacts").run()
    next(b for b in at.button if b.label == "EUR").click().run()
    assert at.warning[0].value == "Approval needed: PII."
    next(b for b in at.button if b.label == "Approve").click().run()
    assert fake.resumes[-1] == {"approved": True, "answer": None}
    assert at.session_state["messages"][-1]["content"] == "Done." and not at.exception
