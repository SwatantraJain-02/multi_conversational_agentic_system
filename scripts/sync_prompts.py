"""Push the local prompts (prompts/**/*.yaml) to Langfuse with the label `development`.

    uv run python scripts/sync_prompts.py            # push what changed
    uv run python scripts/sync_prompts.py --dry-run  # only say what would happen

Idempotent: a prompt whose content already matches the current `development` version is
skipped, so re-running doesn't mint new versions. Promotion to `production` is NOT done
here: it is a deliberate manual step in the Langfuse UI (Prompts -> the version -> add the
`production` label), which is the controlled-promotion demo.
"""

import argparse
import sys
from typing import Any
from urllib.parse import quote

from langfuse.api.core.api_error import ApiError

from multi_agenti_conversation_system import tracing
from multi_agenti_conversation_system.prompts import (
    LocalPrompt,
    load_local,
    prompt_names,
)

LABEL = "development"


def _normalised(body: Any) -> Any:
    """Text as a string, chat as a list of (role, content), so local and remote compare."""
    if isinstance(body, str):
        return body
    return [
        (m["role"], m["content"]) if isinstance(m, dict) else (m.role, m.content)
        for m in body
    ]


def _current_development(client, name: str):
    """The prompt's current `development` version, or None if there is none yet."""
    try:
        # A name like "sql/generate" must be URL-encoded as one path segment: the raw API
        # client doesn't do it, and an unencoded slash lands on the website's 404 page.
        return client.api.prompts.get(quote(name, safe=""), label=LABEL)
    except ApiError as exc:
        if exc.status_code == 404:
            return None
        raise


def _unchanged(remote, local: LocalPrompt) -> bool:
    return (
        remote.type == local.type
        and _normalised(remote.prompt) == _normalised(local.template)
        and (remote.config or {}) == local.config
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    dry_run = parser.parse_args().dry_run

    client = tracing.get_langfuse()
    if client is None:
        print(
            "Langfuse is off: set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and TRACING_ENABLED=true"
        )
        return 1
    if not client.auth_check():
        print("Langfuse rejected the keys (check LANGFUSE_* and LANGFUSE_BASE_URL)")
        return 1

    failures = 0
    for name in prompt_names():
        local = load_local(name)
        remote = _current_development(client, name)
        if remote is not None and _unchanged(remote, local):
            print(f"  unchanged  {name}  (v{remote.version}, label {LABEL})")
            continue
        action = "create" if remote is None else f"new version after v{remote.version}"
        if dry_run:
            print(f"  would {action}  {name}")
            continue
        try:
            created = client.create_prompt(
                name=name,
                prompt=local.template,  # type: ignore[arg-type]
                type=local.type,
                labels=[LABEL],
                config=local.config or None,
                commit_message=local.commit_message,
            )
            print(f"  pushed     {name}  -> v{created.version}, label {LABEL}")
        except Exception as exc:  # noqa: BLE001 - report every prompt, then fail at the end
            failures += 1
            print(f"  FAILED     {name}: {type(exc).__name__}: {exc}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
