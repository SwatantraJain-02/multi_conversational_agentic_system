from pathlib import Path

import pytest

from multi_agenti_conversation_system.catalog import SchemaCatalog

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(scope="session")
def catalog() -> SchemaCatalog:
    """The real catalog from data/, loaded without any settings or .env."""
    return SchemaCatalog.load(DATA_DIR)
