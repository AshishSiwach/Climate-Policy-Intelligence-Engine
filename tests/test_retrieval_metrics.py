"""
Unit tests for retrieval metric functions.

All fixtures are synthetic — no real BM25, dense, or ground truth needed.
Tests verify metric math and edge cases against hand-computed expected values.
"""

from __future__ import annotations

import math

import pytest

from evaluation.retrieval_metrics import (
    aggregate_metrics,
    evaluate_query,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    page_in_range_rate,
    precision_at_k,
    recall_at_k,
    retrieved_doc_ids,
)


def _chunk(doc_id: str, page: int = 1) -> dict:
    """Minimal chunk dict — only fields the metric functions read."""
    return {"doc_id": doc_id, "page_number": page}


# ---------------------------------------------------------------------------
# retrieved_doc_ids
# ---------------------------------------------------------------------------

def test_retrieved_doc_ids_dedupes_preserving_order():
    chunks = [_chunk("A"), _chunk("A"), _chunk("B"), _chunk("A"), _chunk("C")]
    assert retrieved_doc_ids(chunks, k=5) == ["A", "B", "C"]


def test_retrieved_doc_ids_respects_k():
    chunks = [_chunk("A"), _chunk("B"), _chunk("C"), _chunk("D")]
    assert retrieved_doc_ids(chunks, k=2) == ["A", "B"]


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def test_recall_at_k_full_hit_single_doc():
    chunks = [_chunk("A"), _chunk("B"), _chunk("C")]
    assert recall_at_k(chunks, {"A"}, k=5) == 1.0


def test_recall_at_k_partial_hit_cross_doc():
    chunks = [_chunk("A"), _chunk("A"), _chunk("A"), _chunk("A"), _chunk("A")]
    # Expected 2 docs but only A retrieved → 0.5
    assert recall_at_k(chunks, {"A", "B"}, k=5) == 0.5


def test_recall_at_k_no_hit():
    chunks = [_chunk("X"), _chunk("Y")]
    assert recall_at_k(chunks, {"A"}, k=5) == 0.0


def test_recall_at_k_empty_expected_raises():
    with pytest.raises(ValueError, match="undefined"):
        recall_at_k([_chunk("A")], set(), k=5)


# ---------------------------------------------------------------------------
# precision_at_k
# ---------------------------------------------------------------------------

def test_precision_at_k_all_relevant():
    chunks = [_chunk("A"), _chunk("A"), _chunk("B")]
    # Unique docs = [A, B], both expected → precision = 1.0
    assert precision_at_k(chunks, {"A", "B"}, k=5) == 1.0


def test_precision_at_k_half_relevant():
    chunks = [_chunk("A"), _chunk("X")]
    # Unique docs = [A, X], one expected → 0.5
    assert precision_at_k(chunks, {"A"}, k=5) == 0.5


def test_precision_at_k_empty_retrieved_returns_zero():
    assert precision_at_k([], {"A"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# mrr_at_k
# ---------------------------------------------------------------------------

def test_mrr_at_k_top_1_hit():
    chunks = [_chunk("A"), _chunk("B")]
    assert mrr_at_k(chunks, {"A"}, k=5) == 1.0


def test_mrr_at_k_rank_3_hit():
    chunks = [_chunk("X"), _chunk("Y"), _chunk("A"), _chunk("B")]
    assert mrr_at_k(chunks, {"A"}, k=5) == pytest.approx(1 / 3)


def test_mrr_at_k_no_hit_in_topk():
    chunks = [_chunk("X"), _chunk("Y")]
    assert mrr_at_k(chunks, {"A"}, k=5) == 0.0


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

def test_ndcg_at_k_perfect_ranking():
    """All relevant at the top → nDCG = 1.0."""
    chunks = [_chunk("A"), _chunk("A"), _chunk("X"), _chunk("Y")]
    # rels = [1, 1, 0, 0]; ideal same → nDCG = 1.0
    assert ndcg_at_k(chunks, {"A"}, k=4) == pytest.approx(1.0)


def test_ndcg_at_k_reversed_ranking():
    """All relevant at the bottom → nDCG < 1.0."""
    chunks = [_chunk("X"), _chunk("Y"), _chunk("A"), _chunk("A")]
    # rels = [0, 0, 1, 1]; ideal = [1, 1, 0, 0]
    # dcg = 1/log2(4) + 1/log2(5) = 0.5 + 0.4307 = 0.9307
    # idcg = 1/log2(2) + 1/log2(3) = 1.0 + 0.6309 = 1.6309
    expected = (1 / math.log2(4) + 1 / math.log2(5)) / (1 / math.log2(2) + 1 / math.log2(3))
    assert ndcg_at_k(chunks, {"A"}, k=4) == pytest.approx(expected)


def test_ndcg_at_k_no_relevant_returns_zero():
    chunks = [_chunk("X"), _chunk("Y")]
    assert ndcg_at_k(chunks, {"A"}, k=2) == 0.0


# ---------------------------------------------------------------------------
# hit_at_k
# ---------------------------------------------------------------------------

def test_hit_at_k_one_of_two_expected():
    chunks = [_chunk("A"), _chunk("X")]
    assert hit_at_k(chunks, {"A", "B"}, k=5) == 1


def test_hit_at_k_zero_when_none():
    chunks = [_chunk("X"), _chunk("Y")]
    assert hit_at_k(chunks, {"A"}, k=5) == 0


# ---------------------------------------------------------------------------
# page_in_range_rate
# ---------------------------------------------------------------------------

def test_page_in_range_all_hit():
    chunks = [_chunk("A", page=25), _chunk("A", page=30)]
    expected_sources = [{"doc_id": "A", "page_range": [20, 35]}]
    assert page_in_range_rate(chunks, expected_sources, k=5) == 1.0


def test_page_in_range_half_hit():
    chunks = [_chunk("A", page=25), _chunk("A", page=50)]   # 25 in range, 50 out
    expected_sources = [{"doc_id": "A", "page_range": [20, 35]}]
    assert page_in_range_rate(chunks, expected_sources, k=5) == 0.5


def test_page_in_range_returns_none_when_no_range_available():
    """Cross-doc queries with all null page_ranges → metric not applicable."""
    chunks = [_chunk("A", page=25)]
    expected_sources = [{"doc_id": "A", "page_range": None}]
    assert page_in_range_rate(chunks, expected_sources, k=5) is None


def test_page_in_range_ignores_chunks_from_unexpected_docs():
    """Only chunks whose doc_id has a range in expected_sources count in the denominator."""
    chunks = [_chunk("A", page=25), _chunk("Z", page=999)]
    expected_sources = [{"doc_id": "A", "page_range": [20, 35]}]
    # Only chunk from A counts; Z has no range so excluded → 1/1 = 1.0
    assert page_in_range_rate(chunks, expected_sources, k=5) == 1.0


# ---------------------------------------------------------------------------
# evaluate_query
# ---------------------------------------------------------------------------

def test_evaluate_query_returns_all_metrics_at_all_ks():
    chunks = [_chunk("A", page=25), _chunk("B", page=1), _chunk("A", page=30)]
    sources = [{"doc_id": "A", "page_range": [20, 35]}]
    result = evaluate_query(chunks, sources, ks=(3, 5))
    for k in (3, 5):
        for m in ("recall", "precision", "mrr", "ndcg", "hit"):
            assert f"{m}@{k}" in result
        assert f"page_in_range@{k}" in result
    assert result["n_expected_docs"] == 1


def test_evaluate_query_skips_page_metric_when_no_ranges():
    chunks = [_chunk("A", page=25)]
    sources = [{"doc_id": "A", "page_range": None}]
    result = evaluate_query(chunks, sources, ks=(5,))
    assert "page_in_range@5" not in result
    assert "recall@5" in result   # doc-level still computed


def test_evaluate_query_negative_raises():
    with pytest.raises(ValueError, match="negatives"):
        evaluate_query([_chunk("A")], expected_sources=[], ks=(5,))


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

def test_aggregate_mean_of_numeric_metrics():
    per_query = [
        {"recall@5": 1.0, "mrr@5": 1.0},
        {"recall@5": 0.5, "mrr@5": 0.5},
    ]
    agg = aggregate_metrics(per_query)
    assert agg["recall@5"] == pytest.approx(0.75)
    assert agg["mrr@5"] == pytest.approx(0.75)


def test_aggregate_handles_partial_missing_keys():
    """Some queries may lack page_in_range@k (cross-doc null ranges) — aggregate skips missing."""
    per_query = [
        {"recall@5": 1.0, "page_in_range@5": 0.8},
        {"recall@5": 0.5},   # no page metric
    ]
    agg = aggregate_metrics(per_query)
    assert agg["recall@5"] == pytest.approx(0.75)
    assert agg["page_in_range@5"] == pytest.approx(0.8)   # single-value mean


def test_aggregate_empty_returns_empty():
    assert aggregate_metrics([]) == {}
