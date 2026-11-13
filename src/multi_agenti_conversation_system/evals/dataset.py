"""Loading evals/dataset.yaml and splitting a case into what the task receives (`input`) and
what the scorers compare against (`expected`)."""

from pathlib import Path
from typing import Any

import yaml

DATASET_NAME = "multi-agent-assistant-eval"
FILE = Path(__file__).with_name("dataset.yaml")
INPUT_KEYS = ("id", "category", "question", "setup", "prime")


def load_cases(
    category: str | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    cases = yaml.safe_load(FILE.read_text(encoding="utf-8"))
    if category:
        cases = [c for c in cases if c["category"] == category]
    return cases[:limit] if limit else cases


def input_of(case: dict[str, Any]) -> dict[str, Any]:
    return {k: case[k] for k in INPUT_KEYS if k in case}


def expected_of(case: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in case.items() if k not in INPUT_KEYS}
