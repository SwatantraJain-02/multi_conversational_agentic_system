"""In-memory stand-ins for the DB server adapters (NorthwindReader, SchemaIndex)."""

from typing import Any


class FakeReader:
    def __init__(self, error: Exception | None = None, rows: int = 3):
        self.error = error
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []

    def fetch_rows(self, sql: str, max_rows: int) -> dict[str, Any]:
        self.calls.append(("fetch_rows", (sql, max_rows)))
        if self.error:
            raise self.error
        n = min(self.rows, max_rows)
        return {
            "columns": ["x"],
            "rows": [[i] for i in range(n)],
            "row_count": n,
            "truncated": self.rows > max_rows,
        }

    def explain(self, sql: str) -> int:
        self.calls.append(("explain", sql))
        if self.error:
            raise self.error
        return 42

    def ping(self, timeout_s: float) -> bool:
        return self.error is None

    def close(self) -> None:
        pass


class FakeIndex:
    def __init__(self, error: Exception | None = None):
        self.error = error

    def search_schema(self, question, k, domain):
        if self.error:
            raise self.error
        return [
            {
                "table": "orders",
                "domain": "sales",
                "has_pii": True,
                "document": "d",
                "similarity": 0.9,
            }
        ]

    def search_few_shots(self, question, k, domain):
        if self.error:
            raise self.error
        return [
            {
                "question": "q",
                "sql": "SELECT 1",
                "domain": "sales",
                "pii": False,
                "similarity": 0.8,
            }
        ]

    def ping(self) -> bool:
        return self.error is None

    def is_built(self) -> bool:
        return self.error is None
