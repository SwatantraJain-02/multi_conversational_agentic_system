"""Measure the embedding model's similarity for pairs of questions, to pick CACHE_THRESHOLD.

    uv run python scripts/calibrate_cache.py

Uses only the embedding model (no Pinecone, no chat model). Three kinds of pairs:

- PARAPHRASES: same question, different words. These must HIT, so the threshold has to be
  at or below the lowest of them.
- DIFFERENT QUESTIONS about the same entity ("orders in Germany" vs "customers in Germany").
  These must MISS, so the threshold has to be above the highest of them.
- NEAR MISSES that differ only in an id or a name. Their similarity is high, and that is
  fine: the exact `entity_key` filter separates them, not the threshold. Shown to prove why
  similarity alone could not be trusted.

The recommendation is the midpoint of the gap between the lowest paraphrase and the highest
different-question score. A gap that is empty (the ranges overlap) means the model cannot
separate the two kinds, and the threshold must lean towards "miss" (a wrong hit is worse than
a miss). It prints the value; it does not change `.env`.
"""

import math

from multi_agenti_conversation_system.llm import get_embedding_model

PARAPHRASES = [
    (
        "How many customers do we have in each country?",
        "Show me the number of customers per country",
    ),
    ("What is our total revenue by year?", "Revenue per year, please"),
    ("List the top 5 products by sales", "Which five products sell the most?"),
    (
        "Show the details of the product with product_id 100",
        "What are the details for product_id 100?",
    ),
    (
        "What are the main trends in the global beverage market?",
        "Tell me about the biggest global beverage market trends",
    ),
    ("How many orders were placed in 2025?", "Number of orders placed during 2025"),
    (
        "Which employee handled the most orders?",
        "Who is the employee with the highest number of orders handled?",
    ),
    ("What is the latest news about coffee prices?", "Latest coffee price news"),
]
DIFFERENT = [
    ("How many orders did Germany place?", "How many customers do we have in Germany?"),
    ("What is our total revenue by year?", "How many orders were placed by year?"),
    ("List the top 5 products by sales", "List the top 5 customers by sales"),
    (
        "Show the details of the product with product_id 100",
        "Show the orders for the product with product_id 100",
    ),
    (
        "What is the latest news about coffee prices?",
        "What is the latest news about tea prices?",
    ),
    ("Average order value in 2025", "Total order value in 2025"),
]
NEAR_MISSES = [
    (
        "Show the details of the product with product_id 100",
        "Show the details of the product with product_id 150",
    ),
    (
        "How many orders did Germany place in 2025?",
        "How many orders did Germany place in 2024?",
    ),
    ("Show the orders of customer ALFKI", "Show the orders of customer ANATR"),
    ("Show employee 3's sales", "Show employee 7's sales"),
]


# (question, fact statement): relevant pairs must be recalled, unrelated ones must not.
FACT_RELEVANT = [
    ("Which region do I manage?", "Manages the APAC region"),
    ("Which region does the user manage?", "Manages the APAC region"),
    ("Show me sales for my region", "Manages the APAC region"),
    ("What am I preparing this quarter?", "Preparing the Q3 review"),
    ("Where is my team based?", "Team is based in Singapore"),
    ("What is my job title?", "Works as a regional sales director"),
]
FACT_UNRELATED = [
    ("How many customers do we have in Germany?", "Manages the APAC region"),
    ("What is the latest news about coffee prices?", "Team is based in Singapore"),
    ("List the top 5 products by sales", "Preparing the Q3 review"),
    ("How many orders were placed in 2025?", "Works as a regional sales director"),
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def score(pairs: list[tuple[str, str]]) -> list[tuple[float, str, str]]:
    emb = get_embedding_model()
    out = []
    for a, b in pairs:
        out.append((cosine(emb.embed_query(a), emb.embed_query(b)), a, b))
    return sorted(out)


def show(title: str, rows: list[tuple[float, str, str]]) -> None:
    print(f"\n{title}")
    for sim, a, b in rows:
        print(f"  {sim:.3f}  {a!r}  ~  {b!r}")


def main() -> None:
    para, diff, near = score(PARAPHRASES), score(DIFFERENT), score(NEAR_MISSES)
    show("PARAPHRASES (must hit)", para)
    show("DIFFERENT QUESTIONS, same entity (must miss)", diff)
    show("NEAR MISSES, differ in an id or name (the entity_key separates them)", near)
    rel, unrel = score(FACT_RELEVANT), score(FACT_UNRELATED)
    show("FACTS: relevant question ~ fact (must be recalled)", rel)
    show("FACTS: unrelated question ~ fact (must not be)", unrel)
    print(
        f"\nfacts: lowest relevant {rel[0][0]:.3f}, highest unrelated {unrel[-1][0]:.3f}"
    )
    if rel[0][0] > unrel[-1][0]:
        print(
            f"-> FACTS_MIN_SIMILARITY between them: {(rel[0][0] + unrel[-1][0]) / 2:.2f}"
        )
    else:
        print(f"-> overlap: use FACTS_MIN_SIMILARITY >= {unrel[-1][0] + 0.01:.2f}")
    lowest_hit, highest_miss = para[0][0], diff[-1][0]
    print(
        f"\nlowest paraphrase {lowest_hit:.3f}, highest different question {highest_miss:.3f}"
    )
    print(
        f"lowest near miss  {near[0][0]:.3f}  (all of these are above any sane threshold)"
    )
    if lowest_hit > highest_miss:
        print(
            f"-> a gap exists. Recommended CACHE_THRESHOLD = {(lowest_hit + highest_miss) / 2:.2f} "
            f"(current setting: see config.py)"
        )
    else:
        print(
            f"-> the ranges overlap. Use CACHE_THRESHOLD >= {highest_miss + 0.01:.2f}: "
            "a few paraphrases will miss, but no different question will hit."
        )


if __name__ == "__main__":
    main()
