"""Prompt loading, strict rendering and the Langfuse -> fallback -> local decision."""

from types import SimpleNamespace

import pytest

from multi_agenti_conversation_system import prompts, tracing
from multi_agenti_conversation_system.prompts import (
    PromptError,
    get_prompt,
    load_local,
    prompt_names,
)


def test_local_prompts_are_discovered_by_path():
    assert "plumbing/echo-reply" in prompt_names()
    local = load_local("plumbing/echo-reply")
    assert local.type == "text" and "{{task}}" in local.template


@pytest.mark.parametrize("name", ["nope/missing", "../pyproject", "/etc/passwd"])
def test_unknown_or_escaping_names_are_rejected(name):
    with pytest.raises(PromptError):
        load_local(name)


def test_compile_is_strict_about_variables():
    prompt = get_prompt("plumbing/echo-reply")
    assert prompt.compile(task="hi", unused=1) == "echo: hi"
    with pytest.raises(PromptError, match="task"):
        prompt.compile()


def test_chat_prompts_compile_each_message():
    prompt = prompts.Prompt(
        "x",
        "chat",
        [
            {"role": "system", "content": "You are {{role}}."},
            {"role": "user", "content": "{{q}}"},
        ],
        "local",
        "development",
    )
    assert prompt.compile(role="a DBA", q="how many?") == [
        {"role": "system", "content": "You are a DBA."},
        {"role": "user", "content": "how many?"},
    ]


def test_with_tracing_off_the_source_is_local():
    prompt = get_prompt("plumbing/echo-reply")
    assert (prompt.source, prompt.version, prompt.link) == ("local", None, None)
    assert prompt.tag == "plumbing/echo-reply@local"


class _FakeLangfuse:
    def __init__(self, fetched=None, error=None):
        self.fetched, self.error, self.calls = fetched, error, []

    def get_prompt(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if self.error:
            raise self.error
        return self.fetched


def _fetched(prompt="echo: {{task}} (v7)", version=7, is_fallback=False):
    return SimpleNamespace(
        prompt=prompt, version=version, config={"t": 1}, is_fallback=is_fallback
    )


def test_langfuse_result_carries_version_label_and_a_link(monkeypatch):
    fake = _FakeLangfuse(_fetched())
    monkeypatch.setattr(tracing, "get_langfuse", lambda: fake)
    prompt = get_prompt("plumbing/echo-reply", label="development")
    assert (prompt.source, prompt.version, prompt.label) == (
        "langfuse",
        7,
        "development",
    )
    assert prompt.tag == "plumbing/echo-reply@v7"
    assert prompt.link is fake.fetched
    assert prompt.compile(task="hi") == "echo: hi (v7)"
    kwargs = fake.calls[0][1]
    assert kwargs["label"] == "development"
    assert (
        kwargs["max_retries"] == 1 and kwargs["fetch_timeout_seconds"] <= 5
    )  # a bad key can't stall


def test_langfuse_fallback_result_is_reported_as_fallback(monkeypatch):
    monkeypatch.setattr(
        tracing, "get_langfuse", lambda: _FakeLangfuse(_fetched(is_fallback=True))
    )
    prompt = get_prompt("plumbing/echo-reply")
    assert (prompt.source, prompt.version, prompt.link) == ("fallback", None, None)
    assert prompt.compile(task="hi") == "echo: hi"  # the local copy's text


def test_a_langfuse_exception_falls_back_instead_of_raising(monkeypatch):
    monkeypatch.setattr(
        tracing, "get_langfuse", lambda: _FakeLangfuse(error=OSError("down"))
    )
    assert get_prompt("plumbing/echo-reply").source == "fallback"


def test_local_and_langfuse_templates_render_identically(monkeypatch):
    local = get_prompt("plumbing/echo-reply")
    template = load_local("plumbing/echo-reply").template
    monkeypatch.setattr(
        tracing, "get_langfuse", lambda: _FakeLangfuse(_fetched(prompt=template))
    )
    remote = get_prompt("plumbing/echo-reply")
    assert local.compile(task="same") == remote.compile(task="same")
