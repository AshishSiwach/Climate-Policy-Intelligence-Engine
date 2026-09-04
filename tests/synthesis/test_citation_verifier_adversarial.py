"""
Adversarial tests for the hardened citation verifier (evidence.citations).

Six attack / edge cases — all must result in either a clean keep or a clean
drop as specified in Phase 0 Baseline Hardening.
"""

from __future__ import annotations

from evidence.citations import verify_citations
from synthesis.output_schema import LLMCitation

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(doc_id: str, chunk_index: int, text: str, page: int = 1, pub: str = "2024") -> dict:
    return {
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_{chunk_index}",
        "text": text,
        "page_number": page,
        "publication_date": pub,
        "chunk_type": "prose",
    }


def _cite(doc_id: str, passage: str, page: int, chunk_id: str | None) -> LLMCitation:
    return LLMCitation(doc_id=doc_id, passage=passage, page=page, chunk_id=chunk_id)


# ---------------------------------------------------------------------------
# Test 1 — Right passage, wrong chunk_id → dropped
# ---------------------------------------------------------------------------


def test_right_passage_wrong_chunk_id_dropped():
    """Passage exists in chunk A but citation claims chunk B → dropped."""
    chunk_a = _chunk("ofgem", 0, "Ofgem proposes new load control licensing requirements for 2026.")
    chunk_b = _chunk("ofgem", 1, "The consultation closes in February 2026 and responses will be published.")

    # passage is from chunk A, but chunk_id points at chunk B
    cit = _cite("ofgem", "Ofgem proposes new load control licensing requirements for 2026.", 1, chunk_id="ofgem_1")

    result = verify_citations([cit], [chunk_a, chunk_b])
    assert result == [], "Citation should be dropped: passage in chunk A but chunk_id=ofgem_1 (chunk B)"


# ---------------------------------------------------------------------------
# Test 2 — Correct chunk_id, passage not found in that chunk → dropped
# ---------------------------------------------------------------------------


def test_correct_chunk_id_passage_not_in_chunk_dropped():
    """chunk_id resolves to the right chunk, but the passage text isn't there."""
    chunk = _chunk("boe", 0, "UK banks faced aggregate losses under the CBES scenario.")
    # passage references something completely different
    cit = _cite("boe", "The moon is made of green cheese according to regulators.", 1, chunk_id="boe_0")

    result = verify_citations([cit], [chunk])
    assert result == [], "Citation should be dropped: passage not found in the claimed chunk"


# ---------------------------------------------------------------------------
# Test 3 — Ambiguous anchor (passage in ≥2 chunks) → dropped
# ---------------------------------------------------------------------------


def test_ambiguous_anchor_dropped():
    """Boilerplate phrase appearing in multiple chunks → citation is ambiguous → dropped."""
    boilerplate = "This document is subject to copyright and may not be reproduced without permission."
    chunk_a = _chunk("doc_a", 0, boilerplate + " See section 3 for details.")
    chunk_b = _chunk("doc_b", 0, boilerplate + " All rights reserved worldwide.")

    # Model correctly identifies chunk_a as the source
    cit = _cite("doc_a", boilerplate, 1, chunk_id="doc_a_0")

    result = verify_citations([cit], [chunk_a, chunk_b])
    assert result == [], "Ambiguous anchor (appears in 2 chunks) must be dropped"


# ---------------------------------------------------------------------------
# Test 4 — Fabricated passage (not in any chunk) → dropped
# ---------------------------------------------------------------------------


def test_fabricated_passage_dropped():
    """LLM hallucinated a passage that doesn't appear anywhere → dropped."""
    chunk = _chunk("ofgem", 0, "Ofgem proposes new load control licensing requirements for 2026.")
    cit = _cite("ofgem", "Carbon taxes will triple by 2030 under new Ofgem rules.", 1, chunk_id="ofgem_0")

    result = verify_citations([cit], [chunk])
    assert result == [], "Fabricated passage must be dropped"


# ---------------------------------------------------------------------------
# Test 5 — Happy path: exact match on chunk_id and passage → kept
# ---------------------------------------------------------------------------


def test_happy_path_kept_with_chunk_metadata():
    """Correct chunk_id + passage found → citation kept; doc_id/page from chunk."""
    chunk = _chunk("ofgem", 2, "Smart meter rollout targets have been revised for 2027.", page=7, pub="2025")
    cit = _cite(
        "wrong_doc_id_from_model",  # model supplies wrong doc_id — must be overridden
        "Smart meter rollout targets have been revised for 2027.",
        99,  # model supplies wrong page — must be overridden
        chunk_id="ofgem_2",
    )

    result = verify_citations([cit], [chunk])
    assert len(result) == 1, "Citation should be kept"
    assert result[0].doc_id == "ofgem", "doc_id must come from the matched chunk, not the model"
    assert result[0].page == 7, "page must come from the matched chunk, not the model"
    assert result[0].publication_date == "2025"
    assert "Smart meter rollout targets" in result[0].passage


# ---------------------------------------------------------------------------
# Test 6 — Model supplied wrong doc_id but chunk_id is correct → kept with chunk's doc_id
# ---------------------------------------------------------------------------


def test_wrong_doc_id_overridden_by_chunk():
    """chunk_id matches chunk A (doc_id='ofgem'), model claims doc_id='boe' → kept with doc_id='ofgem'."""
    ofgem_chunk = _chunk("ofgem", 5, "Electricity grid balancing costs reached £2.1 billion in 2024.", page=12)
    boe_chunk = _chunk("boe", 0, "Bank stress tests showed resilience to a 4°C warming scenario.", page=3)

    cit = _cite(
        doc_id="boe",  # model hallucinated wrong doc_id
        passage="Electricity grid balancing costs reached £2.1 billion in 2024.",
        page=3,  # model hallucinated wrong page
        chunk_id="ofgem_5",  # chunk_id correctly identifies ofgem_5
    )

    result = verify_citations([cit], [ofgem_chunk, boe_chunk])
    assert len(result) == 1, "Citation should be kept despite wrong doc_id from model"
    assert result[0].doc_id == "ofgem", "doc_id must be pulled from the matched chunk (ofgem), not the model (boe)"
    assert result[0].page == 12, "page must be pulled from chunk (12), not model (3)"
