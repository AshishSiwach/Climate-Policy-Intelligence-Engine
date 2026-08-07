"""
Retrieval quality metrics — pure functions, no I/O.

Computes standard IR metrics at document level from retrieved chunks:
  - Recall@k          — did we retrieve any chunk from each expected doc?
  - Precision@k       — what fraction of retrieved docs are relevant?
  - MRR@k             — how high up the first expected doc appears
  - nDCG@k            — position-weighted quality with binary relevance
  - Hit@k             — coarse flag: any expected doc in top-k?
  - page_in_range     — bonus: fraction of retrieved chunks whose page fell in
                        the expected page range for their doc (only computed
                        when at least one expected_source has a non-null range)

Convention: `k` is the CHUNK cutoff. That's what the pipeline actually returns
to the synthesiser. If top-k chunks all come from 1 doc, that's 1 unique doc,
and the doc-level metrics reflect that.

Negatives (expected_sources=[]) are NOT scored here — the runner handles them
via a "correctly_rejected" check against the pipeline's RRF threshold.
"""

from __future__ import annotations

import math
from typing import Iterable

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unique_ordered(items: Iterable[str]) -> list[str]:
    """Deduplicate preserving first-appearance order."""
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def retrieved_doc_ids(retrieved_chunks: list[dict], k: int) -> list[str]:
    """Extract unique doc_ids from top-k chunks, preserving rank order."""
    return _unique_ordered(c["doc_id"] for c in retrieved_chunks[:k])


# ---------------------------------------------------------------------------
# Core metrics — all doc-level, cutoff = k chunks
# ---------------------------------------------------------------------------

def recall_at_k(retrieved_chunks: list[dict], expected_docs: set[str], k: int) -> float:
    """Fraction of expected docs represented by any chunk in top-k."""
    if not expected_docs:
        raise ValueError("Recall is undefined for empty expected_docs (negatives).")
    retrieved = set(retrieved_doc_ids(retrieved_chunks, k))
    return len(retrieved & expected_docs) / len(expected_docs)


def precision_at_k(retrieved_chunks: list[dict], expected_docs: set[str], k: int) -> float:
    """Of the unique docs represented by top-k chunks, what fraction are expected?"""
    if not expected_docs:
        raise ValueError("Precision is undefined for empty expected_docs (negatives).")
    retrieved = retrieved_doc_ids(retrieved_chunks, k)
    if not retrieved:
        return 0.0
    hits = sum(1 for d in retrieved if d in expected_docs)
    return hits / len(retrieved)


def mrr_at_k(retrieved_chunks: list[dict], expected_docs: set[str], k: int) -> float:
    """Reciprocal rank of the first chunk from any expected doc. 0 if none in top-k."""
    if not expected_docs:
        raise ValueError("MRR is undefined for empty expected_docs (negatives).")
    for rank, chunk in enumerate(retrieved_chunks[:k], start=1):
        if chunk["doc_id"] in expected_docs:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_chunks: list[dict], expected_docs: set[str], k: int) -> float:
    """Normalised DCG with binary relevance at the chunk level."""
    if not expected_docs:
        raise ValueError("nDCG is undefined for empty expected_docs (negatives).")

    def dcg(rels: list[int]) -> float:
        return sum(r / math.log2(i + 2) for i, r in enumerate(rels))

    rels = [1 if c["doc_id"] in expected_docs else 0 for c in retrieved_chunks[:k]]
    # Ideal ordering: all relevant chunks first, up to what's actually retrievable.
    n_relevant = sum(rels)
    ideal_rels = [1] * n_relevant + [0] * (len(rels) - n_relevant)

    idcg = dcg(ideal_rels)
    if idcg == 0.0:
        return 0.0
    return dcg(rels) / idcg


def hit_at_k(retrieved_chunks: list[dict], expected_docs: set[str], k: int) -> int:
    """1 if any expected doc appears in top-k chunks, else 0."""
    if not expected_docs:
        raise ValueError("Hit is undefined for empty expected_docs (negatives).")
    top_docs = {c["doc_id"] for c in retrieved_chunks[:k]}
    return 1 if top_docs & expected_docs else 0


# ---------------------------------------------------------------------------
# chunk_type slice — added Week 5 Step 2b (CLAUDE.md) for table-retrieval eval
# ---------------------------------------------------------------------------

def table_fraction_at_k(retrieved_chunks: list[dict], k: int) -> float:
    """
    Fraction of top-k retrieved chunks that carry `chunk_type == "table"`.

    Purely descriptive — does not need `expected_docs`. Applies to every query
    (positive or negative). Higher = table chunks are surfacing; near-zero on
    numeric queries would signal that Tier 2 heading injection isn't helping
    the retriever find table pages.
    """
    top = retrieved_chunks[:k]
    if not top:
        return 0.0
    return sum(1 for c in top if c.get("chunk_type") == "table") / len(top)


# ---------------------------------------------------------------------------
# Bonus: page-level in-range rate
# ---------------------------------------------------------------------------

def page_in_range_rate(
    retrieved_chunks: list[dict],
    expected_sources: list[dict],   # from ground_truth.json enriched schema
    k: int,
) -> float | None:
    """
    Of the top-k chunks that came from an expected doc AND that doc has a
    non-null page_range, what fraction fall inside the expected range?

    Returns None if no expected_source has a page_range (metric not applicable).
    """
    # Build lookup: doc_id -> [start, end] (only for sources with a range)
    ranges: dict[str, list[int]] = {
        s["doc_id"]: s["page_range"]
        for s in expected_sources
        if s.get("page_range") is not None
    }
    if not ranges:
        return None

    hits = 0
    total = 0
    for chunk in retrieved_chunks[:k]:
        doc_id = chunk["doc_id"]
        if doc_id not in ranges:
            continue
        total += 1
        start, end = ranges[doc_id]
        if start <= chunk["page_number"] <= end:
            hits += 1

    return hits / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Per-query aggregation
# ---------------------------------------------------------------------------

def evaluate_query(
    retrieved_chunks: list[dict],
    expected_sources: list[dict],
    ks: tuple[int, ...] = (5, 10),
) -> dict:
    """
    Compute all applicable metrics for one query at each k in `ks`.

    Skips negatives (empty expected_sources) — those are handled separately
    by the runner via the pipeline's out-of-corpus threshold.
    """
    expected_docs = {s["doc_id"] for s in expected_sources}
    if not expected_docs:
        raise ValueError(
            "evaluate_query does not handle negatives. "
            "Runner should check the out-of-corpus short-circuit behaviour instead."
        )

    result: dict = {"n_expected_docs": len(expected_docs)}
    for k in ks:
        result[f"recall@{k}"] = recall_at_k(retrieved_chunks, expected_docs, k)
        result[f"precision@{k}"] = precision_at_k(retrieved_chunks, expected_docs, k)
        result[f"mrr@{k}"] = mrr_at_k(retrieved_chunks, expected_docs, k)
        result[f"ndcg@{k}"] = ndcg_at_k(retrieved_chunks, expected_docs, k)
        result[f"hit@{k}"] = hit_at_k(retrieved_chunks, expected_docs, k)
        result[f"table_fraction@{k}"] = table_fraction_at_k(retrieved_chunks, k)
        page_rate = page_in_range_rate(retrieved_chunks, expected_sources, k)
        if page_rate is not None:
            result[f"page_in_range@{k}"] = page_rate
    return result


def aggregate_metrics(per_query_results: list[dict]) -> dict[str, float]:
    """Mean of each numeric metric across queries. Missing keys ignored per query."""
    if not per_query_results:
        return {}
    keys: set[str] = set()
    for r in per_query_results:
        keys.update(k for k, v in r.items() if isinstance(v, (int, float)))
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [r[key] for r in per_query_results if key in r]
        if values:
            aggregated[key] = sum(values) / len(values)
    return aggregated
