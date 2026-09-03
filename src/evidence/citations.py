"""
Hardened citation verifier — Phase 0 Baseline Hardening.

Fail-closed rules (all must pass to keep a citation):
1. chunk_id must be present on the LLMCitation.
2. chunk_id must match exactly one chunk in the retrieved list.
3. The normalised passage anchor (first 60 chars) must appear in THAT
   specific chunk's text, not any chunk.
4. doc_id and page are pulled from the MATCHED chunk, not from the model —
   the model cannot forge metadata.
5. If the passage anchor appears in ≥2 retrieved chunks (boilerplate /
   ambiguous), the citation is dropped (cannot be pinned to one source).
"""

from __future__ import annotations

import logging

from synthesis.output_schema import Citation, LLMCitation

logger = logging.getLogger(__name__)

_ANCHOR_LEN = 60  # characters — long enough to be distinctive


def verify_citations(
    citations: list[LLMCitation],
    chunks: list[dict],
) -> list[Citation]:
    """Fail-closed: drops any citation that can't be unambiguously bound to a chunk.

    Args:
        citations: LLM-produced citations (may have chunk_id=None for backwards-compat).
        chunks: Retrieved chunks as returned by HybridRetriever.  Each chunk
                must have ``chunk_id``, ``doc_id``, ``page_number``, and ``text``.

    Returns:
        List of verified, metadata-enriched Citation objects.
    """
    verified: list[Citation] = []

    # Pre-compute normalised text per chunk for efficient membership tests.
    normalised: list[tuple[str, str, dict]] = [(c.get("chunk_id", ""), _normalise(c["text"]), c) for c in chunks]

    for cit in citations:
        # Rule 1 — chunk_id required
        if not cit.chunk_id:
            logger.info(
                "Dropped citation (no chunk_id): doc_id=%s page=%s passage=%r",
                cit.doc_id,
                cit.page,
                (cit.passage or "")[:80],
            )
            continue

        target = _normalise(cit.passage)
        if not target:
            continue

        anchor = target[:_ANCHOR_LEN]

        # Rule 2 — chunk_id must resolve to exactly one retrieved chunk
        matched_chunk: dict | None = None
        for chunk_id, _norm_text, chunk in normalised:
            if chunk_id == cit.chunk_id:
                matched_chunk = chunk
                break

        if matched_chunk is None:
            logger.info(
                "Dropped citation: chunk_id=%r not in retrieved chunks (doc_id=%s)",
                cit.chunk_id,
                cit.doc_id,
            )
            continue

        # Rule 3 — passage must appear in THAT chunk
        matched_norm_text = _normalise(matched_chunk["text"])
        if anchor not in matched_norm_text:
            logger.info(
                "Dropped citation: passage anchor not found in chunk_id=%r",
                cit.chunk_id,
            )
            continue

        # Rule 5 — ambiguity check across ALL chunks
        matches_count = sum(1 for _cid, norm_text, _c in normalised if anchor in norm_text)
        if matches_count >= 2:
            logger.info(
                "Dropped citation: ambiguous anchor appears in %d chunks (chunk_id=%r)",
                matches_count,
                cit.chunk_id,
            )
            continue

        # Rule 4 — pull doc_id and page from the matched chunk, ignore model values
        verified.append(
            Citation(
                doc_id=matched_chunk["doc_id"],
                passage=cit.passage,
                page=matched_chunk["page_number"],
                publication_date=matched_chunk.get("publication_date"),
            )
        )

    return verified


def _normalise(text: str) -> str:
    """Lowercase + collapse whitespace — matches synthesiser._normalise exactly."""
    return " ".join(text.lower().split())
