"""The schema descriptions and few-shot examples from data/*.yaml, as one SchemaCatalog.

The SQL validator (table allowlist, PII columns), the schema indexer (Chroma documents)
and the MCP tools (list_tables, get_schema) all read from one catalog, so they can never
disagree. Pure: it reads YAML from the directory it is given and imports no configuration
or infrastructure, so both the agent and the DB MCP server can use it.
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import yaml

SCHEMA_FILE = "schema_descriptions.yaml"
FEW_SHOTS_FILE = "few_shots.yaml"


class CatalogError(Exception):
    """The catalog files are missing or malformed."""


def _load_section(path: Path, key: str) -> Any:
    try:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise CatalogError(f"Catalog file not found: {path}") from exc
    if not isinstance(data, dict) or key not in data:
        raise CatalogError(f"{path} has no top-level '{key}' section")
    return data[key]


@dataclass(frozen=True, eq=False)
class SchemaCatalog:
    tables: dict[str, dict[str, Any]]
    """Every table as {name: {domain, description, columns}}."""
    few_shots: list[dict[str, Any]]
    """Seed examples: [{question, domain, sql, pii?}, ...]."""

    @classmethod
    def load(cls, data_dir: Path) -> SchemaCatalog:
        return cls(
            tables=_load_section(data_dir / SCHEMA_FILE, "tables"),
            few_shots=_load_section(data_dir / FEW_SHOTS_FILE, "examples"),
        )

    def __len__(self) -> int:
        return len(self.tables)

    def table_names(self) -> set[str]:
        """The validator's allowlist. Lower-case, because sqlglot keeps the case of quoted names."""
        return {name.lower() for name in self.tables}

    def domains(self) -> list[str]:
        return sorted({info["domain"] for info in self.tables.values()})

    def pii_columns(self) -> dict[str, set[str]]:
        """{table: {pii column names}} for tables that have at least one PII column."""
        result: dict[str, set[str]] = {}
        for table, info in self.tables.items():
            cols = {
                col for col, meta in info["columns"].items() if meta.get("pii", False)
            }
            if cols:
                result[table.lower()] = {c.lower() for c in cols}
        return result

    def table_domain(self, table: str) -> str:
        return self.tables[table]["domain"]

    def table_description(self, table: str) -> str:
        """The table's description on one line (the YAML folds it over several)."""
        return " ".join(self.tables[table]["description"].split())

    def table_document(self, table: str) -> str:
        """Plain-text description of one table, used both for embedding and for get_schema."""
        info = self.tables[table]
        lines = [
            f"Table: {table}",
            f"Domain: {info['domain']}",
            f"Description: {self.table_description(table)}",
            "Columns:",
        ]
        for col, meta in info["columns"].items():
            pii_tag = " [PII]" if meta.get("pii", False) else ""
            lines.append(f"- {col}{pii_tag}: {meta['description']}")
        return "\n".join(lines)


@cache
def load_catalog(data_dir: Path) -> SchemaCatalog:
    """Load once per directory, then reuse."""
    return SchemaCatalog.load(data_dir)
