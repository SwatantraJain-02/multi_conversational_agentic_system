from pathlib import Path

import pytest

from multi_agenti_conversation_system.catalog import CatalogError, SchemaCatalog


def test_loads_every_table_and_example(catalog: SchemaCatalog):
    assert len(catalog) == 14
    assert len(catalog.few_shots) == 20
    assert "orders" in catalog.table_names()


def test_domains_are_sorted_and_complete(catalog: SchemaCatalog):
    assert catalog.domains() == ["customers", "hr", "logistics", "products", "sales"]


def test_pii_map_is_lower_case_and_only_lists_tables_with_pii(catalog: SchemaCatalog):
    pii = catalog.pii_columns()
    assert "phone" in pii["customers"]
    assert {"ship_name", "ship_address"} <= pii["orders"]
    assert "order_details" not in pii
    assert all(
        t == t.lower() and all(c == c.lower() for c in cols) for t, cols in pii.items()
    )


def test_table_document_format(catalog: SchemaCatalog):
    doc = catalog.table_document("customers")
    lines = doc.splitlines()
    assert lines[0] == "Table: customers"
    assert lines[1] == "Domain: customers"
    assert lines[2].startswith(
        "Description: "
    ) and "\n" not in catalog.table_description("customers")
    assert lines[3] == "Columns:"
    assert any(line.startswith("- phone [PII]: ") for line in lines)


def test_missing_directory_raises_a_clear_error(tmp_path: Path):
    with pytest.raises(CatalogError, match="not found"):
        SchemaCatalog.load(tmp_path)


def test_malformed_file_raises_a_clear_error(tmp_path: Path):
    (tmp_path / "schema_descriptions.yaml").write_text("nothing: here\n")
    (tmp_path / "few_shots.yaml").write_text("examples: []\n")
    with pytest.raises(CatalogError, match="'tables'"):
        SchemaCatalog.load(tmp_path)
