"""
Tests for the verifier node — Phase 2 Cross-Document Route.

No LLM mock needed — the verifier is purely deterministic.
"""

from __future__ import annotations

from src.agent.nodes.verifier import run_verifier
from src.evidence.claims import Claim

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claim(
    id_: str,
    text: str,
    evidence_ids: list[str],
    source_doc_id: str = "doc",
) -> Claim:
    return Claim(id=id_, text=text, evidence_ids=evidence_ids, source_doc_id=source_doc_id)


def _make_chunk(chunk_id: str, doc_id: str, text: str, page: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "page_number": page,
        "publication_date": "2024",
        "chunk_type": "prose",
    }


def _make_state(claims: list, retrievals: dict, steps_used: int = 4) -> dict:
    return {
        "request_id": "test-req-verifier",
        "query": "test query",
        "task_type": "cross_doc",
        "sub_questions": [],
        "retrievals": retrievals,
        "coverage": {},
        "claims": claims,
        "verified_claims": [],
        "steps_used": steps_used,
        "cost_used_usd": 0.01,
        "time_used_s": 8.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }


# ---------------------------------------------------------------------------
# Test 1: Claim with valid chunk_id and matching passage → kept
# ---------------------------------------------------------------------------


def test_valid_claim_kept():
    """Claim whose evidence_id resolves to a unique chunk → claim is kept."""
    chunk = _make_chunk("boe_0", "boe", "The BoE runs annual climate stress tests under CBES.")
    claim = _make_claim("c_0", "BoE runs stress tests.", ["boe_0"], "boe")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    assert result["verified_claims"][0].id == "c_0"
    assert "boe_0" in result["verified_claims"][0].evidence_ids


# ---------------------------------------------------------------------------
# Test 2: Claim with chunk_id not in retrievals → dropped
# ---------------------------------------------------------------------------


def test_claim_with_missing_chunk_id_dropped():
    """evidence_id not found in any retrieval chunk → claim dropped entirely."""
    chunk = _make_chunk("boe_0", "boe", "The BoE runs annual climate stress tests under CBES.")
    # evidence_id "nonexistent_42" does not appear in retrievals
    claim = _make_claim("c_0", "Some claim.", ["nonexistent_42"], "boe")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk]}))

    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Test 3: Claim with chunk_id found but chunk text is ambiguous (boilerplate) → dropped
# ---------------------------------------------------------------------------


def test_claim_with_ambiguous_boilerplate_chunk_dropped():
    """Chunk text that appears in 2+ chunks triggers the ambiguity rule → dropped."""
    boilerplate = "This document is subject to copyright and may not be reproduced without permission."
    chunk_a = _make_chunk("doc_a_0", "doc_a", boilerplate + " See section 3.", page=1)
    chunk_b = _make_chunk("doc_b_0", "doc_b", boilerplate + " All rights reserved.", page=1)

    # Claim references chunk_a whose first 60 chars appear in both chunks
    claim = _make_claim("c_0", "Copyright statement.", ["doc_a_0"], "doc_a")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk_a, chunk_b]}))

    # The anchor (first 60 chars of boilerplate) appears in 2 chunks → ambiguous → dropped
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Test 4: Claim with zero surviving evidence → dropped entirely
# ---------------------------------------------------------------------------


def test_claim_with_zero_surviving_evidence_dropped():
    """All evidence_ids fail verification → claim dropped entirely."""
    chunk = _make_chunk("boe_0", "boe", "Valid unique text about BoE stress testing scenarios.")
    # Both evidence_ids point at non-existent chunks
    claim = _make_claim("c_0", "Fabricated claim.", ["ghost_1", "ghost_2"], "boe")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk]}))

    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Test 5: 2 evidence_ids; one verifies, one doesn't → claim kept with only verified id
# ---------------------------------------------------------------------------


def test_claim_partial_evidence_kept_with_verified_subset():
    """Claim with 2 evidence_ids: one valid, one missing → kept with only the valid id."""
    chunk = _make_chunk("boe_0", "boe", "The Bank of England published its climate stress results for 2024.")
    # "ghost_99" does not exist in retrievals
    claim = _make_claim("c_0", "BoE published stress results.", ["boe_0", "ghost_99"], "boe")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    vc = result["verified_claims"][0]
    assert vc.id == "c_0"
    assert vc.evidence_ids == ["boe_0"]
    assert "ghost_99" not in vc.evidence_ids


# ---------------------------------------------------------------------------
# Test 6: Multiple claims — some kept, some dropped
# ---------------------------------------------------------------------------


def test_multiple_claims_mixed_outcome():
    """Mix of valid and invalid claims: only valid ones survive."""
    chunk_a = _make_chunk("ofg_0", "ofgem", "Ofgem proposes new licensing for load controllers in 2026.")
    chunk_b = _make_chunk("boe_0", "boe", "BoE published the Climate Biennial Exploratory Scenario results.")

    valid_claim = _make_claim("c_0", "Ofgem licensing proposal.", ["ofg_0"], "ofgem")
    invalid_claim = _make_claim("c_1", "Ghost reference.", ["ghost_x"], "boe")
    another_valid = _make_claim("c_2", "BoE CBES results.", ["boe_0"], "boe")

    retrievals = {"sq_0": [chunk_a], "sq_1": [chunk_b]}
    result = run_verifier(_make_state([valid_claim, invalid_claim, another_valid], retrievals))

    surviving_ids = {vc.id for vc in result["verified_claims"]}
    assert "c_0" in surviving_ids
    assert "c_1" not in surviving_ids
    assert "c_2" in surviving_ids
    assert len(result["verified_claims"]) == 2


# ---------------------------------------------------------------------------
# Test 7: steps_used incremented
# ---------------------------------------------------------------------------


def test_steps_incremented():
    """steps_used is incremented by 1."""
    claim = _make_claim("c_0", "BoE stress tests.", ["boe_0"], "boe")
    chunk = _make_chunk("boe_0", "boe", "Unique BoE text about climate scenarios for 2024.")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk]}, steps_used=5))

    assert result["steps_used"] == 6


# ---------------------------------------------------------------------------
# Test 8: Empty claims list → empty verified claims, no error
# ---------------------------------------------------------------------------


def test_empty_claims_returns_empty():
    """No claims to verify → empty verified_claims."""
    result = run_verifier(_make_state([], {}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Test 9: source_doc_id corrected to chunk's actual doc_id
# ---------------------------------------------------------------------------


def test_source_doc_id_corrected_to_chunk_doc():
    """When evidence_id resolves to a chunk from a different doc, source_doc_id is corrected."""
    # Chunk is from 'ofgem' but claim claims 'boe'
    chunk = _make_chunk("ofg_5", "ofgem", "Electricity grid balancing costs reached two billion pounds in 2024.")
    claim = _make_claim("c_0", "Grid balancing costs.", ["ofg_5"], source_doc_id="boe")

    result = run_verifier(_make_state([claim], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    assert result["verified_claims"][0].source_doc_id == "ofgem"


# ---------------------------------------------------------------------------
# Test 10: Claim dict (not Claim object) is accepted
# ---------------------------------------------------------------------------


def test_claim_dict_accepted():
    """run_verifier accepts claim as plain dict (Claim coercion)."""
    chunk = _make_chunk("boe_0", "boe", "Unique text: BoE climate exploratory scenario findings were published.")
    claim_dict = {
        "id": "c_0",
        "text": "BoE scenario findings.",
        "evidence_ids": ["boe_0"],
        "source_doc_id": "boe",
    }

    result = run_verifier(_make_state([claim_dict], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    assert isinstance(result["verified_claims"][0], Claim)
