"""
Unit tests for retrieval — BM25Retriever and HybridRetriever RRF fusion.

DenseRetriever is not unit-tested here because it requires the BAAI/bge-base
model and Chroma; those are covered by scripts/validate_pipeline_e2e.py.
We mock the dense retriever's interface where hybrid tests need it.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever, _rrf_score

# ---------------------------------------------------------------------------
# BM25Retriever
# ---------------------------------------------------------------------------


def test_bm25_build_then_query_returns_results(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    results = bm25.query("Ofgem load control licensing", top_k=3)
    assert len(results) >= 1
    assert all("bm25_score" in r for r in results)
    assert all("bm25_rank" in r for r in results)


def test_bm25_scores_strictly_positive(sample_chunks):
    """BM25 excludes zero-score results (term overlap required)."""
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    results = bm25.query("Ofgem licensing", top_k=5)
    assert all(r["bm25_score"] > 0 for r in results)


def test_bm25_ranks_are_1_indexed_and_ordered(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    results = bm25.query("Ofgem", top_k=3)
    ranks = [r["bm25_rank"] for r in results]
    assert ranks == list(range(1, len(results) + 1))


def test_bm25_metadata_carried_through(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    results = bm25.query("Ofgem", top_k=1)
    r = results[0]
    for key in ("doc_id", "institution", "publication_date", "page_number", "chunk_index"):
        assert key in r, f"metadata field {key} lost through BM25"


def test_bm25_query_before_build_raises():
    bm25 = BM25Retriever()
    with pytest.raises(RuntimeError, match="Index not built"):
        bm25.query("anything")


def test_bm25_build_empty_list_raises():
    bm25 = BM25Retriever()
    with pytest.raises(ValueError, match="empty"):
        bm25.build([])


def test_bm25_save_load_roundtrip(sample_chunks, tmp_path):
    """A saved index reloaded from disk must produce identical query results."""
    path = tmp_path / "bm25.pkl"
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)
    original = bm25.query("Ofgem licensing", top_k=3)
    bm25.save(path)

    loaded = BM25Retriever.load(path)
    reloaded = loaded.query("Ofgem licensing", top_k=3)

    assert len(original) == len(reloaded)
    for a, b in zip(original, reloaded):
        assert a["doc_id"] == b["doc_id"]
        assert a["chunk_index"] == b["chunk_index"]
        assert abs(a["bm25_score"] - b["bm25_score"]) < 1e-9


# ---------------------------------------------------------------------------
# RRF fusion math
# ---------------------------------------------------------------------------


def test_rrf_score_formula():
    """RRF: 1/(k + rank). k=60 default. Verify formula."""
    assert _rrf_score(1, 60) == pytest.approx(1 / 61)
    assert _rrf_score(2, 60) == pytest.approx(1 / 62)
    assert _rrf_score(10, 30) == pytest.approx(1 / 40)


def test_rrf_score_monotonic_decreasing_with_rank():
    """Higher rank = worse position = lower RRF score."""
    scores = [_rrf_score(r, 60) for r in range(1, 10)]
    for i in range(len(scores) - 1):
        assert scores[i] > scores[i + 1]


# ---------------------------------------------------------------------------
# HybridRetriever — with mocked dense retriever
# ---------------------------------------------------------------------------


def test_hybrid_fuses_bm25_and_dense_results(sample_chunks):
    """Chunks appearing in both retrievers accumulate RRF from both sources."""
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)

    # Mock dense retriever — returns same chunks with a fake dense_rank
    def fake_dense_query(text, top_k=20):
        return [{**c, "dense_score": 0.9 - 0.1 * i, "dense_rank": i + 1} for i, c in enumerate(sample_chunks[:top_k])]

    dense_mock = MagicMock()
    dense_mock.query.side_effect = fake_dense_query

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    results = hybrid.retrieve("Ofgem", top_k=5)

    assert len(results) >= 1
    assert all("rrf_score" in r for r in results)
    # Fused results should be sorted by rrf_score descending
    scores = [r["rrf_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_hybrid_dedupes_chunks_by_id(sample_chunks):
    """Same chunk from BM25 and dense should appear ONCE in results."""
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)

    def fake_dense_query(text, top_k=20):
        return [{**c, "dense_score": 0.9, "dense_rank": i + 1} for i, c in enumerate(sample_chunks)]

    dense_mock = MagicMock()
    dense_mock.query.side_effect = fake_dense_query

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    results = hybrid.retrieve("Ofgem", top_k=10)

    seen_ids = set()
    for r in results:
        cid = (r["doc_id"], r["chunk_index"])
        assert cid not in seen_ids, f"chunk {cid} duplicated in hybrid output"
        seen_ids.add(cid)


def test_hybrid_returns_top_k_at_most(sample_chunks):
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)

    def fake_dense_query(text, top_k=20):
        return [{**c, "dense_score": 0.9, "dense_rank": i + 1} for i, c in enumerate(sample_chunks)]

    dense_mock = MagicMock()
    dense_mock.query.side_effect = fake_dense_query

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    results = hybrid.retrieve("Ofgem", top_k=2)
    assert len(results) <= 2


def test_hybrid_locked_rrf_k_default():
    """CLAUDE.md locks RRF k=60."""
    bm25 = MagicMock()
    dense = MagicMock()
    hybrid = HybridRetriever(bm25=bm25, dense=dense)
    assert hybrid.rrf_k == 60


# ---------------------------------------------------------------------------
# HybridRetriever — metadata filter (Week 5)
# ---------------------------------------------------------------------------


def _make_chunks(specs):
    """Helper: turn [(doc_id, chunk_idx, institution, text)] into chunk dicts."""
    return [{"doc_id": did, "chunk_index": ci, "institution": inst, "text": text} for did, ci, inst, text in specs]


def test_hybrid_filters_dense_via_native_where():
    """When institutions is non-empty, dense.query must receive institutions kwarg."""
    chunks = _make_chunks(
        [
            ("A", 0, "Ofgem", "ofgem load control"),
            ("B", 0, "IEA", "iea global fossil"),
        ]
    )
    bm25 = BM25Retriever()
    bm25.build(chunks)

    dense_mock = MagicMock()
    dense_mock.query.return_value = [
        {**chunks[0], "dense_score": 0.9, "dense_rank": 1},
    ]

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    hybrid.retrieve("Ofgem load control", top_k=5, institutions=["Ofgem"])

    # dense.query must have been called with institutions=['Ofgem']
    _, kwargs = dense_mock.query.call_args
    assert kwargs.get("institutions") == ["Ofgem"]


def test_hybrid_post_filters_bm25_by_institution():
    """BM25 chunks not matching the filter must be dropped from fusion."""
    chunks = _make_chunks(
        [
            ("A", 0, "Ofgem", "ofgem load control licensing"),
            ("B", 0, "IEA", "iea peak fossil demand"),
            ("C", 0, "CCC", "ccc progress report"),
        ]
    )
    bm25 = BM25Retriever()
    bm25.build(chunks)

    dense_mock = MagicMock()
    dense_mock.query.return_value = []  # dense returns nothing, force reliance on BM25

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    # Query matches all three doc texts (single common token would); filter to Ofgem+IEA
    results = hybrid.retrieve("load control fossil ccc", top_k=5, institutions=["Ofgem", "IEA"])

    returned_institutions = {r["institution"] for r in results}
    assert "CCC" not in returned_institutions, (
        "CCC chunks must be dropped by the metadata filter even though the BM25 query matched them"
    )
    assert returned_institutions.issubset({"Ofgem", "IEA"})


def test_hybrid_falls_back_to_full_corpus_on_zero_filter_matches():
    """If the filter matches nothing, retrieval must fall back to unfiltered results."""
    chunks = _make_chunks(
        [
            ("A", 0, "Ofgem", "ofgem load control"),
            ("B", 0, "IEA", "iea projections"),
        ]
    )
    bm25 = BM25Retriever()
    bm25.build(chunks)

    # Dense mock: when called with institutions=['DESNZ'] → return empty (filter matches nothing)
    #             when called WITHOUT institutions → return the Ofgem chunk (fallback)
    def dense_query(query, top_k=20, institutions=None):
        if institutions:
            return []
        return [{**chunks[0], "dense_score": 0.9, "dense_rank": 1}]

    dense_mock = MagicMock()
    dense_mock.query.side_effect = dense_query

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    # Query mentions DESNZ but our test corpus has no DESNZ chunks → filter matches zero
    results = hybrid.retrieve("DESNZ non-existent topic", top_k=5, institutions=["DESNZ"])

    # Must have fallen back — dense called twice (filter attempt + fallback)
    assert dense_mock.query.call_count == 2
    assert len(results) >= 1, "fallback path should return unfiltered results, not empty"


def test_hybrid_no_institutions_kwarg_matches_v1_behaviour(sample_chunks):
    """institutions=None (or omitted) must NOT pass institutions to dense.query."""
    bm25 = BM25Retriever()
    bm25.build(sample_chunks)

    dense_mock = MagicMock()
    dense_mock.query.return_value = [{**sample_chunks[0], "dense_score": 0.9, "dense_rank": 1}]

    hybrid = HybridRetriever(bm25=bm25, dense=dense_mock, rrf_k=60)
    hybrid.retrieve("Ofgem", top_k=5)  # no institutions kwarg

    _, kwargs = dense_mock.query.call_args
    assert "institutions" not in kwargs, "no-filter path must not pass institutions to dense.query"
