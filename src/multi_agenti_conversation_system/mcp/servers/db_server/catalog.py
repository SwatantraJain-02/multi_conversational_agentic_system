"""Single place that reads the schema descriptions and few-shot examples from data/*.yaml.

The SQL validator (table allowlist, PII columns), the schema indexer (Chroma documents)
and the MCP tools (list_tables, get_schema) all read from here, so they can never disagree.
"""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

from multi_agenti_conversation_system.config import get_path_settings

SCHEMA_FILE = get_path_settings().DATA_DIR / "schema_descriptions.yaml"
FEW_SHOTS_FILE = get_path_settings().DATA_DIR / "few_shots.yaml"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@cache
def schema_tables() -> dict[str, dict[str, Any]]:
    """Every table as {name: {domain, description, columns}}. Loaded once, then cached."""
    return load_yaml(SCHEMA_FILE)["tables"]


@cache
def few_shots() -> list[dict[str, Any]]:
    """Seed examples: [{question, domain, sql, pii?}, ...]."""
    return load_yaml(FEW_SHOTS_FILE)["examples"]


def table_names() -> set[str]:
    """The validator's allowlist. Lower-case, because sqlglot keeps the case of quoted names."""
    return {name.lower() for name in schema_tables()}


def pii_columns() -> dict[str, set[str]]:
    """{table: {pii column names}} for tables that have at least one PII column."""
    result: dict[str, set[str]] = {}
    for table, info in schema_tables().items():
        cols = {col for col, meta in info["columns"].items() if meta.get("pii", False)}
        if cols:
            result[table.lower()] = {c.lower() for c in cols}
    return result


def table_domain(table: str) -> str:
    return schema_tables()[table]["domain"]


def table_document(table: str) -> str:
    """Plain-text description of one table, used both for embedding and for get_schema."""
    info = schema_tables()[table]
    lines = [
        f"Table: {table}",
        f"Domain: {info['domain']}",
        f"Description: {' '.join(info['description'].split())}",
        "Columns:",
    ]
    for col, meta in info["columns"].items():
        pii_tag = " [PII]" if meta.get("pii", False) else ""
        lines.append(f"- {col}{pii_tag}: {meta['description']}")
    return "\n".join(lines)
