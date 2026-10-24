"""Build the Chroma collections schema_docs and sql_seed_examples from data/*.yaml.

Run from the project root (needs Chroma and Ollama running):
    uv run python scripts/build_schema_index.py
Re-run whenever data/schema_descriptions.yaml or data/few_shots.yaml changes.
"""

from multi_agenti_conversation_system.mcp.servers.db_server import (
    catalog,
    schema_index,
)

EXPECTED = {
    schema_index.SCHEMA_COLLECTION: len(catalog.schema_tables()),
    schema_index.FEW_SHOT_COLLECTION: len(catalog.few_shots()),
}


def main() -> None:
    counts = schema_index.build_index()
    for name, count in counts.items():
        status = (
            "OK" if count == EXPECTED[name] else f"MISMATCH (expected {EXPECTED[name]})"
        )
        print(f"{name}: {count} documents  {status}")

    print("\nSample searches:")
    question = "How much revenue did we make per country?"
    print(f"  schema_docs for {question!r}:")
    for hit in schema_index.search_schema(question, k=3):
        print(f"    {hit['similarity']:.3f}  {hit['table']}  [{hit['domain']}]")
    print(f"  sql_seed_examples for {question!r}:")
    for hit in schema_index.search_few_shots(question, k=3):
        print(f"    {hit['similarity']:.3f}  {hit['question']}")


if __name__ == "__main__":
    main()
