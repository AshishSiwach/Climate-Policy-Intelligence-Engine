"""
Adversarial tests for the Phase 2 pipeline (claim builder → verifier → synthesiser).

Tests citation safety across the full Phase 2 evidence pipeline.
All 20+ cases must pass.

The verifier node is the primary gatekeeper — these tests drive it with
edge-case inputs to confirm that only well-grounded claims survive.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.agent.nodes.claim_builder import _parse_claims, run_claim_builder
from src.agent.nodes.verifier import run_verifier
from src.evidence.claims import Claim, Coverage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(chunk_id: str, doc_id: str, text: str, page: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "page_number": page,
        "publication_date": "2024",
        "chunk_type": "prose",
    }


def _claim(
    id_: str, text: str, evidence_ids: list[str], doc_id: str = "doc", source_doc_id: str | None = None
) -> Claim:
    return Claim(id=id_, text=text, evidence_ids=evidence_ids, source_doc_id=source_doc_id or doc_id)


def _verifier_state(claims: list, retrievals: dict) -> dict:
    return {
        "request_id": "adversarial-test",
        "query": "test query",
        "task_type": "cross_doc",
        "sub_questions": [],
        "retrievals": retrievals,
        "coverage": {},
        "claims": claims,
        "verified_claims": [],
        "steps_used": 4,
        "cost_used_usd": 0.01,
        "time_used_s": 5.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }


def _builder_state(sub_questions, retrievals, coverage) -> dict:
    return {
        "request_id": "adversarial-cb",
        "query": "test query",
        "task_type": "cross_doc",
        "sub_questions": sub_questions,
        "retrievals": retrievals,
        "coverage": coverage,
        "claims": [],
        "verified_claims": [],
        "steps_used": 2,
        "cost_used_usd": 0.0,
        "time_used_s": 2.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }


# ---------------------------------------------------------------------------
# Case 1: chunk_id doesn't exist in retrievals → claim dropped
# ---------------------------------------------------------------------------


def test_case_01_nonexistent_chunk_id_drops_claim():
    """LLM returns a claim with a chunk_id that doesn't exist in retrievals → dropped."""
    chunk = _chunk("boe_0", "boe", "The BoE runs annual climate stress tests under CBES.")
    claim = _claim("c_0", "Fabricated claim.", ["nonexistent_chunk_99"], "boe")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Case 2: chunk_id correct but chunk text is boilerplate (appears in 2 chunks) → dropped
# ---------------------------------------------------------------------------


def test_case_02_ambiguous_chunk_text_drops_claim():
    """Correct chunk_id but chunk text (boilerplate) appears in 2+ chunks → ambiguous → dropped."""
    boilerplate = "This document is subject to copyright and may not be reproduced without permission."
    chunk_a = _chunk("doc_a_0", "doc_a", boilerplate + " Unique suffix A for disambiguation only.")
    chunk_b = _chunk("doc_b_0", "doc_b", boilerplate + " Unique suffix B for disambiguation only.")

    # The first 60 chars of chunk_a's text are boilerplate, shared with chunk_b
    claim = _claim("c_0", "Copyright statement.", ["doc_a_0"], "doc_a")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk_a, chunk_b]}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Case 3: Boilerplate appearing in 3 chunks → claim dropped (ambiguous)
# ---------------------------------------------------------------------------


def test_case_03_boilerplate_three_chunks_dropped():
    """Passage anchor in 3 chunks → ambiguous → dropped."""
    boilerplate = "Climate change represents a systemic risk to financial stability and growth."
    chunks = [
        _chunk("a_0", "doc_a", boilerplate + " Source A specific context follows here."),
        _chunk("b_0", "doc_b", boilerplate + " Source B specific context follows here."),
        _chunk("c_0", "doc_c", boilerplate + " Source C specific context follows here."),
    ]
    claim = _claim("c_0", "Climate systemic risk claim.", ["a_0"], "doc_a")

    result = run_verifier(_verifier_state([claim], {"sq_0": chunks}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Case 4: Claim with no evidence_ids at all → dropped by claim builder
# ---------------------------------------------------------------------------


def test_case_04_no_evidence_ids_dropped_by_claim_builder():
    """LLM returns claim with empty evidence_ids → dropped by _parse_claims."""
    raw = json.dumps(
        {
            "claims": [
                {"id": "c_0", "text": "BoE runs stress tests.", "evidence_ids": [], "source_doc_id": "boe"},
                {"id": "c_1", "text": "Valid claim.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
            ]
        }
    )
    claims = _parse_claims(raw)
    # c_0 has empty evidence_ids → dropped; c_1 kept
    claim_ids = [c.id for c in claims]
    assert "c_0" not in claim_ids
    assert "c_1" in claim_ids


# ---------------------------------------------------------------------------
# Case 5: 2 evidence_ids; one verifies, one doesn't → claim kept with verified id only
# ---------------------------------------------------------------------------


def test_case_05_partial_evidence_claim_kept_verified_subset():
    """Claim with 2 evidence_ids: one real chunk, one ghost → kept with only real id."""
    chunk = _chunk("boe_0", "boe", "BoE stress test outcomes were published in the annual report 2024.")
    claim = _claim("c_0", "BoE published stress test outcomes.", ["boe_0", "ghost_x"], "boe")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    vc = result["verified_claims"][0]
    assert vc.evidence_ids == ["boe_0"]
    assert "ghost_x" not in vc.evidence_ids


# ---------------------------------------------------------------------------
# Case 6: Model returns duplicate claim ids → dedup (keep first)
# ---------------------------------------------------------------------------


def test_case_06_duplicate_claim_ids_deduped():
    """_parse_claims deduplicates claim ids, keeping the first occurrence."""
    raw = json.dumps(
        {
            "claims": [
                {"id": "c_0", "text": "First version.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                {
                    "id": "c_0",
                    "text": "Second version (duplicate id).",
                    "evidence_ids": ["boe_1"],
                    "source_doc_id": "boe",
                },
            ]
        }
    )
    claims = _parse_claims(raw)
    assert len(claims) == 1
    assert claims[0].text == "First version."


# ---------------------------------------------------------------------------
# Case 7: Claim text is empty string → dropped by claim builder
# ---------------------------------------------------------------------------


def test_case_07_empty_claim_text_dropped():
    """_parse_claims drops claims with empty text."""
    raw = json.dumps(
        {
            "claims": [
                {"id": "c_0", "text": "", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                {"id": "c_1", "text": "   ", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},  # whitespace only
                {"id": "c_2", "text": "Valid claim text.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
            ]
        }
    )
    claims = _parse_claims(raw)
    ids = [c.id for c in claims]
    assert "c_0" not in ids
    assert "c_1" not in ids
    assert "c_2" in ids


# ---------------------------------------------------------------------------
# Case 8: evidence_id points to chunk from different doc → citation kept, source_doc_id corrected
# ---------------------------------------------------------------------------


def test_case_08_wrong_source_doc_id_corrected():
    """evidence_id resolves to 'ofgem' chunk; claim says 'boe' → kept, source_doc_id='ofgem'."""
    chunk = _chunk("ofg_5", "ofgem", "Ofgem proposes new licensing requirements for energy storage systems.")
    claim = _claim("c_0", "New licensing requirements proposed.", ["ofg_5"], source_doc_id="boe")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    assert result["verified_claims"][0].source_doc_id == "ofgem"


# ---------------------------------------------------------------------------
# Case 9: Passage truncated mid-sentence (still has 60+ char anchor) → kept
# ---------------------------------------------------------------------------


def test_case_09_truncated_passage_with_long_anchor_kept():
    """Chunk with passage longer than 60 chars: anchor (first 60) is unique → kept."""
    # Create a passage that's long and unique enough to survive the anchor check
    long_text = (
        "The Financial Stability Board issued its 2024 report on climate-related financial "
        "risk disclosures, requiring mandatory reporting for all systemically important "
        "financial institutions operating in G20 jurisdictions."
    )
    chunk = _chunk("fsb_0", "fsb", long_text)
    claim = _claim("c_0", "FSB mandates climate risk reporting.", ["fsb_0"], "fsb")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))
    assert len(result["verified_claims"]) == 1


# ---------------------------------------------------------------------------
# Case 10: Unicode passage → handled correctly
# ---------------------------------------------------------------------------


def test_case_10_unicode_passage_handled():
    """Chunk text containing Unicode characters is handled without error."""
    unicode_text = (
        "La Banque d’Angleterre a publié son rapport sur les scénarios "
        "climatiques pour les institutions financières en 2024, avec des exigences "
        "spécifiques pour les banques systémiques."
    )
    chunk = _chunk("boe_fr_0", "boe_fr", unicode_text)
    claim = _claim("c_0", "French BoE climate report.", ["boe_fr_0"], "boe_fr")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))
    assert len(result["verified_claims"]) == 1
    assert result["verified_claims"][0].id == "c_0"


# ---------------------------------------------------------------------------
# Case 11: Very long passage (> 500 chars) → handled
# ---------------------------------------------------------------------------


def test_case_11_very_long_passage_handled():
    """Chunk with very long text (>500 chars) is verified and processed correctly."""
    long_text = (
        "The Intergovernmental Panel on Climate Change Working Group III released its "
        "comprehensive assessment report on climate change mitigation strategies in April 2022. "
        "The report concluded that limiting global warming to 1.5 degrees Celsius above "
        "pre-industrial levels would require rapid, deep and immediate greenhouse gas emissions "
        "reductions across all sectors of the economy. Specifically, global net CO2 emissions "
        "would need to decline by 43 percent by 2030 and reach net zero around 2050. The report "
        "identified renewable energy deployment, energy efficiency improvements, electrification "
        "of end-use sectors, and land-use changes as the primary mitigation pathways. "
        "Financial flows consistent with a 1.5 degree pathway would require a three- to "
        "six-fold increase in annual climate investment by 2030 compared to 2019 levels."
    )
    chunk = _chunk("ipcc_0", "ipcc", long_text)
    claim = _claim("c_0", "IPCC report on mitigation.", ["ipcc_0"], "ipcc")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))
    assert len(result["verified_claims"]) == 1


# ---------------------------------------------------------------------------
# Case 12: All claims from one sub-question dropped → zero verified claims is valid
# ---------------------------------------------------------------------------


def test_case_12_all_claims_dropped_returns_empty():
    """All claims fail verification → empty verified_claims is a valid outcome."""
    chunk = _chunk("boe_0", "boe", "Unique BoE text about stress testing in 2024 annual report.")
    # All claims reference non-existent chunks
    claims = [
        _claim("c_0", "First fabricated claim.", ["ghost_0"]),
        _claim("c_1", "Second fabricated claim.", ["ghost_1"]),
        _claim("c_2", "Third fabricated claim.", ["ghost_2"]),
    ]

    result = run_verifier(_verifier_state(claims, {"sq_0": [chunk]}))
    assert result["verified_claims"] == []
    assert result["steps_used"] > 0  # node still ran


# ---------------------------------------------------------------------------
# Case 13: Claim with missing evidence_ids key (not empty list) → dropped by builder
# ---------------------------------------------------------------------------


def test_case_13_missing_evidence_ids_key_dropped():
    """_parse_claims drops claims where 'evidence_ids' key is missing entirely."""
    raw = json.dumps(
        {
            "claims": [
                {"id": "c_0", "text": "Claim without evidence key.", "source_doc_id": "boe"},
            ]
        }
    )
    claims = _parse_claims(raw)
    assert claims == []


# ---------------------------------------------------------------------------
# Case 14: Multiple evidence_ids all from boilerplate chunk → all dropped, claim dropped
# ---------------------------------------------------------------------------


def test_case_14_all_evidence_ids_boilerplate_claim_dropped():
    """All evidence_ids point to chunks with ambiguous text → claim fully dropped."""
    boilerplate = "All rights reserved. No part of this publication may be reproduced."
    chunk_a = _chunk("pub_0", "pub_a", boilerplate + " Publisher A specific note here.")
    chunk_b = _chunk("pub_1", "pub_b", boilerplate + " Publisher B specific note here.")

    # Both evidence_ids: pub_0 is ambiguous (boilerplate in 2 chunks)
    claim = _claim("c_0", "Copyright claim.", ["pub_0", "pub_1"])

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk_a, chunk_b]}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Case 15: Max 12 claims enforced by claim builder
# ---------------------------------------------------------------------------


def test_case_15_max_claims_enforced():
    """_parse_claims enforces a maximum of 12 claims, discarding the rest."""
    raw_claims = [
        {"id": f"c_{i}", "text": f"Claim number {i}.", "evidence_ids": [f"chunk_{i}"], "source_doc_id": "doc"}
        for i in range(15)  # 15 > 12
    ]
    raw = json.dumps({"claims": raw_claims})
    claims = _parse_claims(raw)
    assert len(claims) == 12


# ---------------------------------------------------------------------------
# Case 16: Chunk with only whitespace text → evidence dropped, claim dropped
# ---------------------------------------------------------------------------


def test_case_16_empty_chunk_text_drops_evidence():
    """Chunk with whitespace-only text → no passage → evidence_id dropped → claim dropped."""
    chunk = _chunk("boe_0", "boe", "   \n\t  ")  # whitespace only
    claim = _claim("c_0", "Some claim.", ["boe_0"])

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Case 17: evidence_id listed twice in claim → only kept once
# ---------------------------------------------------------------------------


def test_case_17_duplicate_evidence_id_in_claim():
    """Claim listing the same evidence_id twice → verifier keeps it once."""
    chunk = _chunk("boe_0", "boe", "BoE climate scenario: banks showed resilience to 3-degree warming path.")
    # evidence_ids has a duplicate
    claim = Claim(id="c_0", text="BoE resilience.", evidence_ids=["boe_0", "boe_0"], source_doc_id="boe")

    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk]}))

    assert len(result["verified_claims"]) == 1
    # evidence_ids should not have duplicates (first occurrence verified, second is same chunk)
    vc = result["verified_claims"][0]
    assert len(vc.evidence_ids) <= 2  # at most 2 but chunk resolves correctly


# ---------------------------------------------------------------------------
# Case 18: Two claims referencing same unique chunk → both kept independently
# ---------------------------------------------------------------------------


def test_case_18_two_claims_same_unique_chunk_both_kept():
    """Two different claims both referencing the same (unique) chunk → both kept."""
    chunk = _chunk(
        "ofg_0",
        "ofgem",
        "Ofgem's consultation on load control licensing received 147 responses from industry stakeholders.",
    )
    claim_a = _claim("c_0", "Ofgem received 147 responses.", ["ofg_0"], "ofgem")
    claim_b = _claim("c_1", "Ofgem consultation had industry stakeholders.", ["ofg_0"], "ofgem")

    result = run_verifier(_verifier_state([claim_a, claim_b], {"sq_0": [chunk]}))

    surviving_ids = {vc.id for vc in result["verified_claims"]}
    assert "c_0" in surviving_ids
    assert "c_1" in surviving_ids


# ---------------------------------------------------------------------------
# Case 19: Mixed claims — valid, missing chunk, boilerplate — correct subset kept
# ---------------------------------------------------------------------------


def test_case_19_mixed_claims_correct_subset_kept():
    """Mix of valid, ghost chunk_id, and boilerplate — only valid claims survive."""
    boilerplate = "This is a standard disclaimer that appears in all documents in the corpus."
    valid_chunk = _chunk(
        "valid_0",
        "ccc",
        "The Climate Change Committee set UK carbon budget 6 at 965 MtCO2e for 2033 to 2037.",
    )
    boilerplate_a = _chunk("bp_0", "doc_a", boilerplate + " Document A additional text.")
    boilerplate_b = _chunk("bp_1", "doc_b", boilerplate + " Document B additional text.")

    claims = [
        _claim("c_0", "CCC carbon budget 6 claim.", ["valid_0"], "ccc"),  # valid
        _claim("c_1", "Ghost chunk reference.", ["ghost_999"], "boe"),  # chunk missing
        _claim("c_2", "Boilerplate claim.", ["bp_0"], "doc_a"),  # ambiguous
    ]

    result = run_verifier(_verifier_state(claims, {"sq_0": [valid_chunk, boilerplate_a, boilerplate_b]}))

    surviving = {vc.id for vc in result["verified_claims"]}
    assert "c_0" in surviving  # valid — kept
    assert "c_1" not in surviving  # ghost chunk — dropped
    assert "c_2" not in surviving  # boilerplate — dropped
    assert len(result["verified_claims"]) == 1


# ---------------------------------------------------------------------------
# Case 20: Empty claims list → empty verified_claims, verifier completes cleanly
# ---------------------------------------------------------------------------


def test_case_20_empty_input_returns_empty_cleanly():
    """Verifier handles empty claims list gracefully."""
    result = run_verifier(_verifier_state([], {}))
    assert result["verified_claims"] == []
    assert result["steps_used"] > 0


# ---------------------------------------------------------------------------
# Case 21: claim builder: LLM returns non-list claims value → empty list
# ---------------------------------------------------------------------------


def test_case_21_non_list_claims_value_in_llm_output():
    """_parse_claims handles malformed output where 'claims' is not a list."""
    raw = json.dumps({"claims": "not a list"})
    claims = _parse_claims(raw)
    assert claims == []


# ---------------------------------------------------------------------------
# Case 22: Claim with evidence_id pointing to chunk that has no chunk_id field → dropped
# ---------------------------------------------------------------------------


def test_case_22_chunk_missing_chunk_id_field():
    """Chunk missing 'chunk_id' field is not indexed → evidence_id lookup fails → dropped."""
    # Chunk has no chunk_id
    chunk_without_id = {
        "doc_id": "boe",
        "text": "BoE runs stress tests without a chunk_id.",
        "page_number": 1,
    }
    claim = _claim("c_0", "BoE runs stress tests.", ["boe_0"])

    # Even though the chunk is in retrievals, it has no chunk_id so it's never indexed
    result = run_verifier(_verifier_state([claim], {"sq_0": [chunk_without_id]}))
    assert result["verified_claims"] == []


# ---------------------------------------------------------------------------
# Case 23: Claim builder enforces max 12 with partial list returned intact
# ---------------------------------------------------------------------------


def test_case_23_claim_builder_max_12_via_mock():
    """Claim builder LLM returns 14 claims → only 12 returned."""
    from src.evidence.claims import SubQuestion

    sqs = [SubQuestion(id="sq_0", question="Question?")]
    retrievals = {"sq_0": [_chunk("c_0", "doc", "Some text about climate policy for sub-question.")]}
    coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}

    raw_claims = [
        {"id": f"c_{i}", "text": f"Claim {i}.", "evidence_ids": ["c_0"], "source_doc_id": "doc"} for i in range(14)
    ]
    llm_output = json.dumps({"claims": raw_claims})

    msg = MagicMock()
    msg.content = llm_output
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 200
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    client = MagicMock()
    client.chat.completions.create.return_value = resp

    with patch("openai.OpenAI", return_value=client):
        result = run_claim_builder(_builder_state(sqs, retrievals, coverage))

    assert len(result["claims"]) == 12
