import pytest

from multi_agenti_conversation_system.retrieval.hybrid import (
    DenseHit,
    candidate_depth,
    hybrid_search,
    keyword_query,
    rrf,
    tokenize,
)


# ------------------------------------------------------------------- tokenizer
@pytest.mark.parametrize(
    "text, expected",
    [
        ("unit_price", ["unit_price", "unit", "price"]),
        ("CustomerID", ["customerid", "customer", "id"]),
        ("unitPrice", ["unitprice", "unit", "price"]),
        ("orders", ["order"]),
        ("categories", ["category"]),
        ("address status", ["address", "status"]),
        ("Freight cost per shipper", ["freight", "cost", "shipper"]),
        (
            "order_details.product_id",
            ["order_details", "order", "detail", "product_id", "product", "id"],
        ),
        ("sales in 2025", ["sale", "2025"]),
    ],
)
def test_tokenize(text, expected):
    assert tokenize(text) == expected


def test_tokenize_only_stopwords_is_empty():
    assert tokenize("What is the") == []
    assert tokenize("  ,;  ") == []


# ------------------------------------------------------------------------- RRF
def test_rrf_arithmetic():
    fused = dict(rrf([["a", "b"], ["b", "c"]]))
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 62)


def test_rrf_order_and_missing_list_contributes_nothing():
    assert [d for d, _ in rrf([["a", "b"], ["b", "c"]])] == ["b", "a", "c"]


def test_rrf_tie_broken_by_dense_rank():
    # x: dense 1 + bm25 2 ; y: dense 2 + bm25 1 -> same score, dense rank wins.
    fused = rrf([["x", "y"], ["y", "x"]])
    assert fused[0][1] == pytest.approx(fused[1][1])
    assert [d for d, _ in fused] == ["x", "y"]


def test_rrf_tie_among_keyword_only_docs_is_deterministic():
    assert [d for d, _ in rrf([[], ["q", "p"]])] == ["q", "p"]
    # Equal score, no dense rank, same bm25 rank in different lists -> id decides.
    assert [d for d, _ in rrf([["m"], ["n"]])] == ["m", "n"]


def test_rrf_weights():
    fused = dict(rrf([["a"], ["b"]], weights=[1.0, 2.0]))
    assert fused["b"] == pytest.approx(2 * fused["a"])
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


# --------------------------------------------------------------- keyword_query
def test_keyword_query_joins_tokens_for_the_bm25_field():
    assert keyword_query("Average unit_price by category") == (
        "average unit_price unit price category"
    )


def test_keyword_query_of_only_stopwords_is_empty():
    assert keyword_query("what is the") == ""


# ----------------------------------------------------------------- hybrid_search
def _keyword(hits):
    return lambda n: hits[:n]


def test_hybrid_fuses_both_branches():
    result = hybrid_search(
        dense_fn=lambda n: [DenseHit("a", 0.9), DenseHit("b", 0.8)][:n],
        keyword_fn=_keyword([("b", 3.0), ("c", 1.0)]),
        k=3,
        depth=10,
    )
    assert not result.degraded
    assert [h.id for h in result.hits] == ["b", "a", "c"]
    b, a, c = result.hits
    assert (b.dense_rank, b.bm25_rank, b.similarity) == (2, 1, 0.8)
    assert (a.dense_rank, a.bm25_rank) == (1, None)
    assert (c.dense_rank, c.bm25_rank, c.similarity) == (None, 2, None)


def test_hybrid_keeps_top_k_and_passes_depth():
    seen = {}

    def dense(n):
        seen["dense"] = n
        return [DenseHit(x, 0.5) for x in "abcd"]

    def keyword(n):
        seen["keyword"] = n
        return []

    result = hybrid_search(dense, keyword, k=2, depth=7)
    assert seen == {"dense": 7, "keyword": 7}
    assert [h.id for h in result.hits] == ["a", "b"]


def test_hybrid_degrades_when_dense_fails():
    def broken(n):
        raise ConnectionError("embedding API is down")

    result = hybrid_search(broken, _keyword([("x", 2.0), ("y", 1.0)]), k=5, depth=10)
    assert result.degraded
    assert [h.id for h in result.hits] == ["x", "y"]
    assert all(h.similarity is None and h.dense_rank is None for h in result.hits)


def test_hybrid_keyword_failure_propagates():
    def broken(n):
        raise ConnectionError("vector store is down")

    with pytest.raises(ConnectionError):
        hybrid_search(lambda n: [DenseHit("a", 0.9)], broken, k=5, depth=10)


def test_hybrid_degraded_when_dense_index_empty():
    result = hybrid_search(lambda n: [], _keyword([("x", 1.0)]), k=5, depth=10)
    assert result.degraded
    assert [h.id for h in result.hits] == ["x"]


def test_hybrid_empty_filtered_corpus_is_not_degraded():
    result = hybrid_search(lambda n: [], _keyword([]), k=5, depth=0)
    assert result.hits == []
    assert not result.degraded


def test_candidate_depth():
    assert candidate_depth(14, k=4) == 14
    assert candidate_depth(50, k=4) == 50
    assert candidate_depth(500, k=4) == 20
    assert candidate_depth(500, k=10) == 40
