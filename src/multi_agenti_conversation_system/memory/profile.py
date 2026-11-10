"""The user profile: how this *person* likes answers. Small, keyed, loaded in full every turn.

Stored in a LangGraph store (`AsyncPostgresStore`, no vector index): namespace
("profile", user_id), key "prefs". A preference is replaced BY KEY, never by similarity:
"code in Python" and "code in Java" embed almost identically, so a similarity-based dedupe
would treat Java as a re-statement of Python and keep the wrong value (G12).

Each entry: {value, source, updated_at, history: [last 5 earlier values]}.
`profile_hash` (a hash of the key/value pairs only) goes into the semantic-cache key, so a
changed preference stops old cached answers matching.

Not built (Should): inferred drift from repeated signals, and the confirm-once flow for an
ambiguous statement. Only explicit, lasting statements (and UI settings) change the profile.
"""

import hashlib
import re
import time
from typing import Any

HISTORY_KEEP = 5
MAX_KEYS = 20
MAX_VALUE_CHARS = 80
# The fixed fields. Anything else a user states becomes an `extra` key (e.g. code_language).
FIXED_KEYS = (
    "answer_length",
    "currency",
    "units",
    "date_format",
    "preferred_domain",
)
_KEY = re.compile(r"[^a-z0-9]+")


def namespace(user_id: str) -> tuple[str, str]:
    return ("profile", user_id)


def normalise_key(key: str) -> str:
    return _KEY.sub("_", key.lower()).strip("_")[:40]


def normalise_value(value: str) -> str:
    return " ".join(str(value).split())[:MAX_VALUE_CHARS]


def values(profile: dict[str, Any]) -> dict[str, str]:
    """`{key: value}` without the bookkeeping: what prompts and agents are given."""
    return {key: entry["value"] for key, entry in sorted(profile.items())}


def profile_hash(profile: dict[str, Any]) -> str:
    """Identifies the current preferences. An empty profile has a hash too (never None)."""
    canonical = "|".join(f"{k}={v.lower()}" for k, v in values(profile).items())
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def apply(
    profile: dict[str, Any],
    key: str,
    value: str,
    source: str = "explicit",
    now: float | None = None,
) -> tuple[dict[str, Any], dict[str, str] | None]:
    """The profile with one preference set, and what changed (None when nothing did).

    The old value moves to `history` (the last HISTORY_KEEP). A re-statement of the same
    value changes nothing. Returns a new dict; the input is not modified.
    """
    key, value = normalise_key(key), normalise_value(value)
    if not key or not value:
        return profile, None
    current = profile.get(key)
    if current and current["value"].lower() == value.lower():
        return profile, None
    if not current and len(profile) >= MAX_KEYS:
        return profile, None
    history = [*current["history"], current["value"]][-HISTORY_KEEP:] if current else []
    updated = {
        **profile,
        key: {
            "value": value,
            "source": source,
            "updated_at": int(now if now is not None else time.time()),
            "history": history,
        },
    }
    return updated, {
        "key": key,
        "old": current["value"] if current else "",
        "new": value,
    }


class ProfileStore:
    """Async access to the profile in a LangGraph store (tests pass an `InMemoryStore`)."""

    def __init__(self, store: Any) -> None:
        self.store = store

    async def load(self, user_id: str) -> dict[str, Any]:
        item = await self.store.aget(namespace(user_id), "prefs")
        return dict(item.value) if item else {}

    async def set(
        self,
        user_id: str,
        key: str,
        value: str,
        source: str = "explicit",
        now: float | None = None,
    ) -> dict[str, str] | None:
        profile, change = apply(await self.load(user_id), key, value, source, now)
        if change:
            await self.store.aput(namespace(user_id), "prefs", profile)
        return change

    async def reset(self, user_id: str) -> None:
        await self.store.adelete(namespace(user_id), "prefs")
