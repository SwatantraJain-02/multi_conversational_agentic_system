import pytest

from multi_agenti_conversation_system.retrieval.hybrid import (
    BM25Index,
    DenseHit,
    candidate_depth,
    hybrid_search,
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


# ------------------------------------------------------------------------ BM25
@pytest.fixture
def index():
    return BM25Index(
        ids=["orders", "order_details", "shippers", "employees"],
        documents=[
            "Table: orders. Columns: order_id, customer_id, freight, ship_via",
            "Table: order_details. Columns: order_id, product_id, unit_price, quantity",
            "Table: shippers. Columns: shipper_id, company_name, phone",
            "Table: employees. Columns: employee_id, last_name, region",
        ],
        metadatas=[
            {"domain": "sales"},
            {"domain": "sales"},
            {"domain": "logistics"},
            {"domain": "hr"},
        ],
    )


def test_bm25_finds_exact_identifier(index):
    assert index.search("unit_price", k=3)[0][0] == "order_details"


def test_bm25_drops_non_matches(index):
    hits = index.search("freight", k=4)
    assert [d for d, _ in hits] == ["orders"]
    assert all(score > 0 for _, score in hits)
    assert index.search("zebra", k=4) == []
    assert index.search("what is the", k=4) == []


def test_bm25_domain_filter_applies_after_scoring(index):
    hits = index.search("order_id", k=4, where={"domain": "sales"})
    assert {d for d, _ in hits} == {"orders", "order_details"}
    assert index.search("shipper", k=4, where={"domain": "sales"}) == []
    assert index.count({"domain": "sales"}) == 2
    assert index.count() == 4


def test_bm25_two_doc_corpus_still_returns_keyword_hits():
    # "order" is in both docs: BM25Okapi's IDF would be <= 0 here (G11); Lucene's stays > 0.
    tiny = BM25Index(
        ids=["a", "b"],
        documents=["orders freight", "orders unit_price"],
        metadatas=[{}, {}],
    )
    assert {d for d, _ in tiny.search("orders", k=2)} == {"a", "b"}
    assert tiny.search("freight", k=2)[0][0] == "a"


def test_bm25_filter_leaving_two_docs_still_returns_keyword_hits(index):
    # Both sales docs contain "order_id"; scored over the full corpus, the IDF stays > 0.
    hits = index.search("order", k=4, where={"domain": "sales"})
    assert len(hits) == 2


def test_bm25_empty_corpus():
    empty = BM25Index(ids=[], documents=[], metadatas=[])
    assert empty.search("anything", k=3) == []
    assert empty.count() == 0


def test_bm25_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        BM25Index(ids=["a"], documents=[], metadatas=[])


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
        raise ConnectionError("chroma is down")

    result = hybrid_search(broken, _keyword([("x", 2.0), ("y", 1.0)]), k=5, depth=10)
    assert result.degraded
    assert [h.id for h in result.hits] == ["x", "y"]
    assert all(h.similarity is None and h.dense_rank is None for h in result.hits)


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
