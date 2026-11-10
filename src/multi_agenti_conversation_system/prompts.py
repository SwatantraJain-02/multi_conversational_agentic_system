"""Prompts as a service: fetched from Langfuse by label, with a local YAML fallback.

    prompt = get_prompt("plumbing/echo-reply")
    text = prompt.compile(task="hello")          # strict {{variables}}
    prompt.tag                                   # "plumbing/echo-reply@v3": goes in TaskResult.prompt_versions

Where a prompt comes from (`Prompt.source`):
- "langfuse":  fetched (or served from the local cache, `PROMPT_CACHE_TTL_S`) for the label
               `PROMPT_LABEL`. This is the normal case and the only one with a version.
- "fallback":  Langfuse is configured but didn't answer (bad key, outage, label or prompt
               missing), so the local YAML is used. Say so loudly: it is NOT the version you
               think you are running. The fetch uses a short timeout and one retry, so a bad
               key doesn't stall an agent.
- "local":     tracing/Langfuse is off (no keys, or `TRACING_ENABLED=false`).

The local copy lives in `prompts/<name>.yaml` (the name is the path, so "sql/generate" is
`prompts/sql/generate.yaml`), is what `scripts/sync_prompts.py` pushes to Langfuse with the
label `development`, and is the fallback. Promotion to `production` is manual, in the
Langfuse UI.

Templates use `{{variable}}` in Langfuse and in YAML alike. `compile` is the single
renderer for both, and is strict: a variable you forget raises instead of leaking
"{{task}}" into a prompt. A `type: chat` prompt is a list of {role, content} messages.
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.config import BASE_DIR, settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = BASE_DIR / "prompts"
_VARIABLE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

PromptType = Literal["text", "chat"]
Source = Literal["langfuse", "fallback", "local"]
Template = str | list[dict[str, str]]


class PromptError(Exception):
    """A prompt file is missing or malformed, or a variable was not supplied."""


@dataclass(frozen=True)
class Prompt:
    name: str
    type: PromptType
    template: Template
    source: Source
    label: str
    version: int | None = None  # Langfuse version; None for fallback / local
    config: dict[str, Any] = field(default_factory=dict)
    client: Any = None  # the Langfuse prompt client, used to link generations

    @property
    def tag(self) -> str:
        """Short id for traces and results: 'name@v3', or 'name@fallback' / 'name@local'."""
        return (
            f"{self.name}@v{self.version}"
            if self.version
            else f"{self.name}@{self.source}"
        )

    @property
    def link(self) -> Any:
        """Pass as `prompt=` to a Langfuse generation to link it to this prompt version."""
        return self.client if self.source == "langfuse" else None

    def compile(self, **variables: Any) -> str | list[dict[str, str]]:
        if self.type == "chat":
            return [
                {
                    "role": m["role"],
                    "content": _render(m["content"], variables, self.name),
                }
                for m in self.template  # type: ignore[union-attr]
            ]
        return _render(self.template, variables, self.name)  # type: ignore[arg-type]


def _render(template: str, variables: dict[str, Any], name: str) -> str:
    missing = sorted({v for v in _VARIABLE.findall(template) if v not in variables})
    if missing:
        raise PromptError(f"prompt '{name}' needs variable(s) {missing}")
    return _VARIABLE.sub(lambda m: str(variables[m.group(1)]), template)


# -------------------------------------------------------------------------- local files
@dataclass(frozen=True)
class LocalPrompt:
    name: str
    type: PromptType
    template: Template
    config: dict[str, Any]
    commit_message: str | None


def prompt_names() -> list[str]:
    """Every local prompt, as its Langfuse name (the path under prompts/, no suffix)."""
    return sorted(
        p.relative_to(PROMPTS_DIR).with_suffix("").as_posix()
        for p in PROMPTS_DIR.rglob("*.yaml")
    )


def load_local(name: str) -> LocalPrompt:
    path: Path = (PROMPTS_DIR / f"{name}.yaml").resolve()
    if PROMPTS_DIR.resolve() not in path.parents or not path.is_file():
        raise PromptError(f"no local prompt '{name}' (expected prompts/{name}.yaml)")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    kind = data.get("type", "text")
    body = data.get("prompt")
    if kind == "text" and isinstance(body, str) and body.strip():
        template: Template = body.strip()
    elif (
        kind == "chat"
        and isinstance(body, list)
        and body
        and all(isinstance(m, dict) and {"role", "content"} <= set(m) for m in body)
    ):
        template = [
            {"role": str(m["role"]), "content": str(m["content"]).strip()} for m in body
        ]
    else:
        raise PromptError(
            f"prompts/{name}.yaml: 'type' must be text (prompt: string) or chat "
            "(prompt: list of {role, content})"
        )
    return LocalPrompt(
        name, kind, template, data.get("config") or {}, data.get("commit_message")
    )


# --------------------------------------------------------------------------------- fetch
def get_prompt(name: str, *, label: str | None = None) -> Prompt:
    """The prompt `name` for `label` (default `PROMPT_LABEL`). Never fails because of Langfuse."""
    label = label or settings.PROMPT_LABEL
    local = load_local(name)

    def from_local(source: Source) -> Prompt:
        return Prompt(
            name, local.type, local.template, source, label, None, local.config
        )

    client = tracing.get_langfuse()
    if client is None:
        return from_local("local")
    try:
        fetched = client.get_prompt(
            name,
            label=label,
            type=local.type,
            cache_ttl_seconds=settings.PROMPT_CACHE_TTL_S,
            fallback=local.template,  # type: ignore[arg-type]
            max_retries=1,
            fetch_timeout_seconds=settings.PROMPT_FETCH_TIMEOUT_S,
        )
    except Exception as exc:  # noqa: BLE001 - a prompt service outage must not stop an agent
        logger.warning(
            "prompt '%s': Langfuse failed (%s), using the local copy", name, exc
        )
        return from_local("fallback")
    if fetched.is_fallback:
        logger.warning(
            "prompt '%s' (label '%s') not available from Langfuse, using the local copy",
            name,
            label,
        )
        return from_local("fallback")
    return Prompt(
        name=name,
        type=local.type,
        template=_template_of(fetched),
        source="langfuse",
        label=label,
        version=fetched.version,
        config=dict(fetched.config or {}),
        client=fetched,
    )


def _template_of(fetched: Any) -> Template:
    body = fetched.prompt
    if isinstance(body, str):
        return body
    return [{"role": m["role"], "content": m["content"]} for m in body]


async def aget_prompt(name: str, *, label: str | None = None) -> Prompt:
    """`get_prompt` off the event loop: a cache miss is a blocking HTTP call."""
    return await asyncio.to_thread(get_prompt, name, label=label)
