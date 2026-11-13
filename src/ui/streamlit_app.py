"""The Streamlit chat UI for the supervisor.

    uv run streamlit run src/ui/streamlit_app.py

Start the MCP servers and the three agents first (`scripts/run_all.py`).

Why there is a background thread (G6): Streamlit re-runs this whole script on every click,
and `asyncio.run()` on each rerun would create and destroy an event loop each time, which
breaks anything async that outlives one run (the Postgres pool, the A2A HTTP clients). So ONE
event loop runs in a daemon thread for the whole process (`Backend`, kept by
`st.cache_resource`); the script submits coroutines to it and waits. The supervisor
serialises turns of one thread itself (G13), so a double click cannot interleave two turns.

State: the conversation lives in the Postgres checkpoint (per thread id). `st.session_state`
holds only what the screen needs: the user, the thread id, the rendered messages, and a turn
that is waiting for approval.
"""

import asyncio
import queue
import threading
import time
import uuid
from concurrent.futures import Future
from dataclasses import asdict
from typing import Any

import streamlit as st

from multi_agenti_conversation_system.supervisor.service import (
    Supervisor,
    TurnResult,
    open_supervisor,
)

DOMAINS = ["any", "sales", "customers", "products", "hr", "logistics"]
NODE_LABELS = {
    "input_guard": "Checking the message",
    "contextualize": "Making the question standalone",
    "load_profile": "Loading your preferences",
    "check_cache": "Looking for a cached answer",
    "load_context": "Recalling relevant memory",
    "planner": "Planning which agents to ask",
    "run_step": "An agent answered",
    "hitl_gate": "Waiting for your approval",
    "resume_dispatch": "Running the approved query",
    "finalize": "Putting the answer together",
    "direct_answer": "Answering from the conversation",
    "write_memory": "Updating memory",
    "summarize_history": "Tidying the conversation",
    "refuse": "Declining the request",
}


class Backend:
    """One event loop in a daemon thread, with a supervisor open on it for the whole process."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()
        self.supervisor: Supervisor | None = None
        self.error: Exception | None = None
        self._ready = threading.Event()
        asyncio.run_coroutine_threadsafe(self._serve(), self.loop)
        self._ready.wait(timeout=120)

    async def _serve(self) -> None:
        try:
            async with open_supervisor() as supervisor:
                self.supervisor = supervisor
                self._ready.set()
                await asyncio.Event().wait()  # keep it open until the process ends
        except Exception as exc:  # noqa: BLE001 - shown on the page
            self.error = exc
            self._ready.set()

    def submit(self, coro) -> Future:
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def call(self, coro, timeout: float = 30) -> Any:
        return self.submit(coro).result(timeout)


@st.cache_resource(show_spinner="Connecting to the agents...")
def backend() -> Backend:
    return Backend()


def run_turn(be: Backend, make_coro) -> TurnResult | None:
    """Run one supervisor call, showing node progress while it works."""
    events: queue.Queue = queue.Queue()
    future = be.submit(make_coro(lambda event: events.put(event)))
    with st.status("Working...", expanded=True) as status:
        while not future.done():
            _drain(events, status)
            time.sleep(0.15)
        _drain(events, status)
        try:
            turn = future.result()
        except Exception as exc:  # noqa: BLE001
            status.update(label="Something went wrong", state="error")
            st.error(f"{type(exc).__name__}: {exc}")
            return None
        status.update(label="Done", state="complete", expanded=False)
    return turn


def _drain(events: queue.Queue, status) -> None:
    while not events.empty():
        event = events.get_nowait()
        if event["type"] == "plan" and event.get("plan"):
            steps = ", ".join(f"{s['agent']}" for s in event["plan"])
            status.write(f"Plan ({event.get('intent')}): {steps}")
        elif event["type"] == "node" and event["node"] in NODE_LABELS:
            status.write(NODE_LABELS[event["node"]])


# ----------------------------------------------------------------------------- the page
st.set_page_config(page_title="Business data assistant", page_icon="💬", layout="wide")
be = backend()
if be.supervisor is None:
    st.error(f"The supervisor could not start: {be.error}")
    st.stop()
supervisor: Supervisor = be.supervisor

ss = st.session_state
ss.setdefault("user_id", "demo-user")
ss.setdefault("thread_id", f"chat-{uuid.uuid4().hex[:8]}")
ss.setdefault("messages", [])
ss.setdefault("pending", None)  # a TurnResult dict waiting for approval
ss.setdefault("memory_version", 0)


def new_chat() -> None:
    """End the conversation (its summary becomes an episode) and start a fresh thread."""
    if ss.messages:
        be.call(supervisor.end_thread(ss.thread_id, ss.user_id), timeout=60)
    ss.thread_id, ss.messages, ss.pending = f"chat-{uuid.uuid4().hex[:8]}", [], None
    ss.memory_version += 1


with st.sidebar:
    st.header("Session")
    user = st.text_input("User", ss.user_id, help="Memory is kept per user.")
    if user != ss.user_id and user.strip():
        ss.user_id = user.strip()
        new_chat()
        st.rerun()
    domain = st.selectbox("Focus on", DOMAINS, help="Limits the SQL schema searched.")
    st.caption(f"Thread: `{ss.thread_id}`")
    if st.button("New chat", use_container_width=True):
        new_chat()
        st.rerun()

    st.header("Memory")
    st.caption("What the assistant has learned about you. Used only for you.")
    try:
        mine = be.call(supervisor.what_i_know(ss.user_id), timeout=30)
    except Exception as exc:  # noqa: BLE001 - the sidebar must not take the page down
        mine = None
        st.warning(f"Memory is unavailable: {type(exc).__name__}")
    if mine is not None:
        if mine["profile"]:
            st.subheader("Preferences")
            for key, entry in mine["profile"].items():
                older = (
                    f" (before: {', '.join(entry['history'])})"
                    if entry["history"]
                    else ""
                )
                st.markdown(f"- **{key}**: {entry['value']}{older}")
            if st.button("Reset preferences"):
                be.call(supervisor.reset_profile(ss.user_id))
                st.rerun()
        if mine["facts"]:
            st.subheader("Facts")
            for fact in mine["facts"]:
                left, right = st.columns([5, 1])
                left.markdown(f"- {fact['dated']}")
                if right.button(
                    "🗑",
                    key=f"forget-{fact['attribute'] or fact['text']}",
                    help="Forget this",
                ):
                    be.call(supervisor.forget_fact(ss.user_id, fact["attribute"]))
                    st.rerun()
        if not mine["profile"] and not mine["facts"]:
            st.write(
                "Nothing yet. Try: *I manage the APAC region* or *From now on, short answers*."
            )


def render_details(meta: dict[str, Any], key: str) -> None:
    badges = []
    if meta.get("cache_hit"):
        badges.append("⚡ cached answer")
    if meta.get("blocked"):
        badges.append("🛑 blocked by the input guard")
    if meta.get("partial"):
        badges.append("⚠️ partial")
    if meta.get("agents_used"):
        badges.append("agents: " + ", ".join(meta["agents_used"]))
    if meta.get("intent"):
        badges.append(f"route: {meta['intent']}")
    if badges:
        st.caption(" · ".join(badges))
    if meta.get("citations"):
        with st.expander(f"Sources ({len(meta['citations'])})"):
            for c in meta["citations"]:
                st.markdown(f"[{c['n']}] [{c.get('title') or c['url']}]({c['url']})")
    if meta.get("sql_used"):
        with st.expander("SQL used"):
            st.code(meta["sql_used"], language="sql")
    if meta.get("memory_changes"):
        with st.expander("What I learned this turn"):
            for change in meta["memory_changes"]:
                st.markdown(f"- {change}")
    if meta.get("prompt_versions") or meta.get("trace_id"):
        with st.expander("Prompts and trace"):
            st.json(meta.get("prompt_versions", {}))
            if meta.get("trace_id"):
                st.caption(f"Langfuse trace id: `{meta['trace_id']}`")
    if meta.get("status") == "answered" and not meta.get("blocked"):
        done = ss.get(f"fb-{key}")
        up, down, note = st.columns([1, 1, 8])
        if up.button("👍", key=f"up-{key}", disabled=bool(done)):
            be.call(supervisor.feedback(meta, positive=True))
            ss[f"fb-{key}"] = "up"
            st.rerun()
        if down.button("👎", key=f"down-{key}", disabled=bool(done)):
            be.call(supervisor.feedback(meta, positive=False))
            ss[f"fb-{key}"] = "down"
            st.rerun()
        if done:
            note.caption(
                "Thanks. Noted."
                if done == "up"
                else "Thanks. That answer will not be reused."
            )


def add_answer(turn: TurnResult) -> None:
    ss.messages.append(
        {"role": "assistant", "content": turn.answer, "meta": asdict(turn)}
    )
    ss.memory_version += 1


for i, m in enumerate(ss.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("meta"):
            render_details(m["meta"], str(i))

if ss.pending:
    request = ss.pending["approval"]
    with st.chat_message("assistant"):
        st.warning(request.get("reason") or "This query needs your approval.")
        st.code(request.get("sql", ""), language="sql")
        if request.get("pii_columns"):
            st.markdown(
                "**Personal data columns:** " + ", ".join(request["pii_columns"])
            )
        yes, no, _ = st.columns([1, 1, 6])
        decision = (
            True
            if yes.button("Approve", type="primary")
            else False
            if no.button("Reject")
            else None
        )
    if decision is not None:
        ss.pending = None
        turn = run_turn(
            be,
            lambda cb: supervisor.resume(
                ss.thread_id, approved=decision, user_id=ss.user_id, on_event=cb
            ),
        )
        if turn:
            add_answer(turn)
        st.rerun()

if prompt := st.chat_input(
    "Ask about our sales data, or about the world", disabled=bool(ss.pending)
):
    ss.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    turn = run_turn(
        be,
        lambda cb: supervisor.ask(
            ss.thread_id,
            ss.user_id,
            prompt,
            domain=None if domain == "any" else domain,
            on_event=cb,
        ),
    )
    if turn and turn.status == "needs_approval":
        ss.pending = asdict(turn)
    elif turn:
        add_answer(turn)
    st.rerun()
