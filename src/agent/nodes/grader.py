"""
Grader node — Phase 2 Cross-Document Route.

Deterministic coverage grading via cross-encoder (ms-marco-MiniLM-L-6-v2).

For each sub-question, scores every retrieved passage with the cross-encoder
and takes the maximum score. That single number drives the coverage decision:

  score >= COVERED_THRESHOLD  → "covered"
  score >= PARTIAL_THRESHOLD  → "partial"
  else                        → "not_covered"

Thresholds were chosen for the CPIE climate-policy corpus and should be
re-calibrated if the corpus or retriever changes — run a small labelled set
through _score_subquestion() and inspect the raw scores.

No LLM call. No langgraph imports.
"""

from __future__ import annotations

import logging

from src.evidence.claims import Coverage, SubQuestion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds — tune against your corpus
# ---------------------------------------------------------------------------

# Calibrated against CPIE corpus (scripts/calibrate_grader_threshold.py):
#   SHOULD_BE_COVERED:  n=4  mean=+4.97  std=0.84 → t-interval lower bound = 3.63
#   LIKELY_NOT_COVERED: n=3  mean=-6.77  std=5.33 → t-interval upper bound unreliable
#     (high std driven by near-miss outlier; add more probe queries before trusting it)
#   PARTIAL_THRESHOLD held at midpoint-of-means until not_covered group has n≥10.
COVERED_THRESHOLD = 3.63  # t-based 95% lower bound of covered distribution
PARTIAL_THRESHOLD = -2.3  # midpoint(mean_covered, mean_ooc) — pending larger probe set

# ---------------------------------------------------------------------------
# Cross-encoder singleton — loaded once per process
# ---------------------------------------------------------------------------

_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("Grader: loading cross-encoder %s", _MODEL_NAME)
        _encoder = CrossEncoder(_MODEL_NAME)
        logger.info("Grader: cross-encoder ready")
    return _encoder


# ---------------------------------------------------------------------------
# Node entry point
# ---------------------------------------------------------------------------


def run_grader(state: dict) -> dict:
    """Grader node: score coverage for each sub-question via cross-encoder.

    Returns partial dict with "coverage" and incremented "steps_used".
    Falls back to all-covered on any error (fail-permissive).
    """
    sub_questions: list = state.get("sub_questions", [])
    retrievals: dict = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)

    if not sub_questions:
        return {"coverage": {}, "steps_used": steps_used + 1}

    try:
        coverage = _grade_all(sub_questions, retrievals)
    except Exception as exc:
        logger.warning("Grader: cross-encoder failed (%s) — defaulting all to 'covered'", exc)
        coverage = _default_coverage(sub_questions)

    for sq_id, cov in coverage.items():
        if cov.status != "covered":
            logger.info("Grader gap [%s] status=%s reason=%r", sq_id, cov.status, cov.gap_reason)

    return {"coverage": coverage, "steps_used": steps_used + 1}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _grade_all(sub_questions: list, retrievals: dict) -> dict[str, Coverage]:
    encoder = _get_encoder()
    coverage: dict[str, Coverage] = {}

    for sq in sub_questions:
        sq_id = sq.id if isinstance(sq, SubQuestion) else sq.get("id", "sq_?")
        question = sq.question if isinstance(sq, SubQuestion) else sq.get("question", "")
        chunks = retrievals.get(sq_id, [])

        max_score, status, gap_reason = _score_subquestion(encoder, question, chunks)

        # Coverage decisions directly control the retry loop, so keep the raw
        # score visible at the normal production log level.  Without this it is
        # impossible to distinguish a threshold false-negative from a stale
        # workflow deployment by looking at container logs.
        logger.info(
            "Grader score [%s] max_score=%.2f status=%s question=%r",
            sq_id,
            max_score,
            status,
            question,
        )
        coverage[sq_id] = Coverage(
            sub_question_id=sq_id,
            status=status,
            gap_reason=gap_reason,
        )

    return coverage


def _score_subquestion(
    encoder,
    question: str,
    chunks: list[dict],
) -> tuple[float, str, str | None]:
    """Return (max_score, status, gap_reason) for one sub-question."""
    if not chunks:
        return -999.0, "not_covered", "No passages retrieved for this sub-question"

    passages = [(chunk.get("text") or chunk.get("passage") or "").strip() for chunk in chunks]
    passages = [p for p in passages if p]

    if not passages:
        return -999.0, "not_covered", "All retrieved passages are empty"

    pairs = [(question, p) for p in passages]
    scores = encoder.predict(pairs)
    max_score = float(max(scores))

    covered_thresh = COVERED_THRESHOLD
    partial_thresh = PARTIAL_THRESHOLD

    if max_score >= covered_thresh:
        return max_score, "covered", None
    elif max_score >= partial_thresh:
        return max_score, "partial", (
            f"Best passage relevance score {max_score:.1f} is below the coverage "
            f"threshold ({covered_thresh}); key details may be missing"
        )
    else:
        return max_score, "not_covered", (
            f"No retrieved passage scored above {partial_thresh} "
            f"(best: {max_score:.1f}); topic not in corpus"
        )


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def _default_coverage(sub_questions: list) -> dict[str, Coverage]:
    coverage: dict[str, Coverage] = {}
    for sq in sub_questions:
        sq_id = sq.id if isinstance(sq, SubQuestion) else sq.get("id", "sq_?")
        coverage[sq_id] = Coverage(sub_question_id=sq_id, status="covered")
    return coverage
