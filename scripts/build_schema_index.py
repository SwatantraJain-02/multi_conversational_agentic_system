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

# Dense-friendly (paraphrase), keyword-friendly (exact column names), and mixed.
SAMPLE_QUESTIONS = [
    "How much revenue did we make per country?",
    "freight cost per shipper",
    "average unit_price by category",
]


def main() -> None:
    counts = schema_index.build_index()
    for name, count in counts.items():
        status = (
            "OK" if count == EXPECTED[name] else f"MISMATCH (expected {EXPECTED[name]})"
        )
        print(f"{name}: {count} documents  {status}")

    for question in SAMPLE_QUESTIONS:
        print(f"\n{question!r}")
        hits, degraded = schema_index.search_schema(question, k=4)
        _print_hits("schema_docs", hits, degraded, "table")
        hits, degraded = schema_index.search_few_shots(question, k=3)
        _print_hits("sql_seed_examples", hits, degraded, "question")


def _rank(value) -> str:
    return "-" if value is None else str(value)


def _print_hits(name: str, hits, degraded: bool, label: str) -> None:
    print(f"  {name}{'  (DEGRADED: BM25 only)' if degraded else ''}")
    print("    rrf  rrf_score  similarity  dense  bm25")
    for rank, hit in enumerate(hits, start=1):
        similarity = "-" if hit["similarity"] is None else f"{hit['similarity']:.3f}"
        print(
            f"    {rank:>3}  {hit['rrf_score']:.5f}  {similarity:>10}"
            f"  {_rank(hit['dense_rank']):>5}  {_rank(hit['bm25_rank']):>4}  {hit[label]}"
        )


if __name__ == "__main__":
    main()
