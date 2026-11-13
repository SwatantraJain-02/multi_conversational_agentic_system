"""The entity key of a question: the literals that change its answer.

"Show product 100" and "Show product 150" embed almost identically, so the semantic cache
must not rely on similarity alone. A cache entry carries an `entity_key`, and a lookup
filters on it EXACTLY: different ids can never match.

The key is built from two sources and merged:
- regexes (this module): numbers, years and dates, quoted strings, and capitalised words that
  are not the first word of a sentence (names of countries, customers, products);
- the model's own list of entities (`contextualize`), because a name typed in lowercase has no
  capital letter for a regex to find.

The regex half is the safety net: a free model that forgets "150" must still not produce a
cache hit, so the key never depends on the model alone. A missing key is the empty string
"", never None (a None would drop the filter clause, see `vectorstore.build_filter`).
"""

import re
from collections.abc import Iterable

MAX_ENTITIES = 12
MAX_ENTITY_CHARS = 40

_NUMBER = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)*(?![\w])")
_QUOTED = re.compile(r"""["“'‘]([^"”'’]{1,40})["”'’]""")
_WORD = re.compile(r"[A-Za-z][A-Za-z0-9_'-]*|\d[\d.,]*")
# Words that start with a capital for grammar, not because they name something.
_NOT_NAMES = frozenset(
    [
        "i",
        "i'm",
        "i've",
        "i'd",
        "i'll",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]
)


def normalise(entity: str) -> str:
    entity = re.sub(
        r"['’]s\b", "", " ".join(entity.lower().split())
    )  # Germany's -> germany
    return entity.strip(" .,;:!?\"'")[:MAX_ENTITY_CHARS]


def _numbers(text: str) -> set[str]:
    # "1,000" and "1000" are the same entity.
    return {m.group().replace(",", "") for m in _NUMBER.finditer(text)}


def _names(text: str) -> set[str]:
    """Capitalised words that do not open a sentence."""
    found = set()
    previous_end = None
    for match in _WORD.finditer(text):
        token = match.group()
        # A word opens a sentence if it is the first, or . ! ? lies between it and the last.
        opens = previous_end is None or bool(
            re.search(r"[.!?]", text[previous_end : match.start()])
        )
        if (
            not opens
            and token[0].isalpha()
            and token[0].isupper()
            and token.lower() not in _NOT_NAMES
        ):
            found.add(token)
        previous_end = match.end()
    return found


def extract_literals(text: str) -> set[str]:
    out = _numbers(text) | _names(text)
    out |= {m.group(1) for m in _QUOTED.finditer(text)}
    return {e for e in (normalise(x) for x in out) if e}


def entity_key(question: str, extra: Iterable[str] = ()) -> str:
    """Sorted, normalised, "|"-joined entities of the question ("" when it has none)."""
    entities = extract_literals(question)
    entities |= {e for e in (normalise(x) for x in extra) if e}
    return "|".join(sorted(entities)[:MAX_ENTITIES])
