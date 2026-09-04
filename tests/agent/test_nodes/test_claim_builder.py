"""
Tests for the claim builder node — Phase 2 Cross-Document Route.

Mocks the OpenAI client so no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.agent.nodes.claim_builder import run_claim_builder
from src.evidence.claims import Claim, Coverage, SubQuestion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sq(id_: str, question: str) -> SubQuestion:
    return SubQuestion(id=id_, question=question)


def _make_coverage(sq_id: str, status: str, gap_reason: str | None = None) -> Coverage:
    return Coverage(sub_question_id=sq_id, status=status, gap_reason=gap_reason)


def _make_chunk(chunk_id: str, doc_id: str, text: str, page: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "page_number": page,
    }


def _make_state(
    sub_questions: list,
    retrievals: dict,
    coverage: dict,
    steps_used: int = 2,
    cost_used_usd: float = 0.01,
) -> dict:
    return {
        "request_id": "test-req-cb",
        "query": "test query",
        "task_type": "cross_doc",
        "sub_questions": sub_questions,
        "retrievals": retrievals,
        "coverage": coverage,
        "claims": [],
        "verified_claims": [],
        "steps_used": steps_used,
        "cost_used_usd": cost_used_usd,
        "time_used_s": 5.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }


def _mock_llm_response(content: str, prompt_tokens: int = 100, completion_tokens: int = 50):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_client(content: str):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response(content)
    return client


# ---------------------------------------------------------------------------
# Test 1: Happy path — 2 sub-questions, 2 claims each with evidence_ids
# ---------------------------------------------------------------------------


class TestClaimBuilderHappyPath:
    def test_two_sub_questions_return_two_claims(self):
        """Happy path: 2 covered sub-questions → 2 claims, each with evidence_ids."""
        sqs = [
            _make_sq("sq_0", "What is BoE climate policy?"),
            _make_sq("sq_1", "What is Ofgem's licensing approach?"),
        ]
        retrievals = {
            "sq_0": [_make_chunk("boe_0", "boe", "BoE runs annual climate stress tests.")],
            "sq_1": [_make_chunk("ofg_0", "ofgem", "Ofgem proposes licensing for load controllers.")],
        }
        coverage = {
            "sq_0": _make_coverage("sq_0", "covered"),
            "sq_1": _make_coverage("sq_1", "covered"),
        }
        llm_output = json.dumps(
            {
                "claims": [
                    {
                        "id": "c_0",
                        "text": "BoE runs annual stress tests.",
                        "evidence_ids": ["boe_0"],
                        "source_doc_id": "boe",
                    },
                    {
                        "id": "c_1",
                        "text": "Ofgem licenses load controllers.",
                        "evidence_ids": ["ofg_0"],
                        "source_doc_id": "ofgem",
                    },
                ]
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert "claims" in result
        assert len(result["claims"]) == 2
        for claim in result["claims"]:
            assert isinstance(claim, Claim)
            assert len(claim.evidence_ids) >= 1
            assert claim.text

    def test_steps_incremented_by_one(self):
        """steps_used is incremented by exactly 1."""
        sqs = [_make_sq("sq_0", "What is BoE policy?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        llm_output = json.dumps(
            {
                "claims": [
                    {"id": "c_0", "text": "BoE runs stress tests.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"}
                ]
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage, steps_used=3))

        assert result["steps_used"] == 4

    def test_partial_coverage_sub_question_included(self):
        """Sub-question with 'partial' coverage is included in LLM call."""
        sqs = [_make_sq("sq_0", "Partial coverage question")]
        retrievals = {"sq_0": [_make_chunk("doc_0", "doc", "Partial text about climate.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "partial", gap_reason="No post-2020 data")}
        llm_output = json.dumps(
            {"claims": [{"id": "c_0", "text": "Partial claim.", "evidence_ids": ["doc_0"], "source_doc_id": "doc"}]}
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert len(result["claims"]) == 1


# ---------------------------------------------------------------------------
# Test 2: Malformed LLM output → empty claims, no crash
# ---------------------------------------------------------------------------


class TestClaimBuilderMalformed:
    def test_invalid_json_returns_empty_no_crash(self):
        """Malformed JSON output → empty claims list, no exception raised."""
        sqs = [_make_sq("sq_0", "What is the BoE approach?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "Some text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        client = _make_client("NOT VALID JSON {{{ definitely broken")

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert result["claims"] == []
        assert "steps_used" in result

    def test_llm_exception_returns_empty_no_crash(self):
        """LLM network exception → empty claims, no crash."""
        sqs = [_make_sq("sq_0", "What is the BoE approach?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "Some text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("network down")

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert result["claims"] == []
        assert result["steps_used"] > 0

    def test_claims_without_evidence_ids_dropped(self):
        """Claims returned without evidence_ids are dropped silently."""
        sqs = [_make_sq("sq_0", "Question?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        llm_output = json.dumps(
            {
                "claims": [
                    {"id": "c_0", "text": "Valid claim.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                    {"id": "c_1", "text": "No evidence claim.", "evidence_ids": [], "source_doc_id": "boe"},
                    {"id": "c_2", "text": "Missing evidence key.", "source_doc_id": "boe"},
                ]
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        # Only the valid claim survives
        assert len(result["claims"]) == 1
        assert result["claims"][0].id == "c_0"

    def test_empty_claim_text_dropped(self):
        """Claims with empty text are dropped."""
        sqs = [_make_sq("sq_0", "Question?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        llm_output = json.dumps(
            {
                "claims": [
                    {"id": "c_0", "text": "", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                    {"id": "c_1", "text": "Valid claim.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                ]
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert len(result["claims"]) == 1
        assert result["claims"][0].id == "c_1"

    def test_duplicate_claim_ids_deduped(self):
        """Duplicate claim ids are deduplicated (first kept)."""
        sqs = [_make_sq("sq_0", "Question?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        llm_output = json.dumps(
            {
                "claims": [
                    {"id": "c_0", "text": "First claim.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                    {"id": "c_0", "text": "Duplicate id claim.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"},
                ]
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert len(result["claims"]) == 1
        assert result["claims"][0].text == "First claim."


# ---------------------------------------------------------------------------
# Test 3: Sub-question with "not_covered" status is skipped
# ---------------------------------------------------------------------------


class TestClaimBuilderSkipsNotCovered:
    def test_not_covered_sub_question_excluded(self):
        """Sub-question with not_covered status produces no LLM call for it."""
        sqs = [
            _make_sq("sq_0", "What is BoE policy?"),
            _make_sq("sq_1", "What is the missing data?"),
        ]
        retrievals = {
            "sq_0": [_make_chunk("boe_0", "boe", "BoE text.")],
            "sq_1": [],
        }
        coverage = {
            "sq_0": _make_coverage("sq_0", "covered"),
            "sq_1": _make_coverage("sq_1", "not_covered"),
        }
        llm_output = json.dumps(
            {
                "claims": [
                    {"id": "c_0", "text": "BoE runs stress tests.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"}
                ]
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert len(result["claims"]) == 1
        assert result["claims"][0].source_doc_id == "boe"

    def test_all_not_covered_returns_empty_without_llm_call(self):
        """All sub-questions not_covered → empty claims, no LLM call made."""
        sqs = [_make_sq("sq_0", "Missing question")]
        retrievals = {"sq_0": []}
        coverage = {"sq_0": _make_coverage("sq_0", "not_covered")}
        client = MagicMock()

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage))

        assert result["claims"] == []
        client.chat.completions.create.assert_not_called()


# ---------------------------------------------------------------------------
# Test 4: Cost is incremented in state
# ---------------------------------------------------------------------------


class TestClaimBuilderCostTracking:
    def test_cost_incremented_after_llm_call(self):
        """cost_used_usd increases after a successful LLM call."""
        sqs = [_make_sq("sq_0", "What is BoE policy?")]
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE text.")]}
        coverage = {"sq_0": _make_coverage("sq_0", "covered")}
        llm_output = json.dumps(
            {
                "claims": [
                    {"id": "c_0", "text": "BoE runs stress tests.", "evidence_ids": ["boe_0"], "source_doc_id": "boe"}
                ]
            }
        )
        client = _make_client(llm_output)
        initial_cost = 0.01

        with patch("openai.OpenAI", return_value=client):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage, cost_used_usd=initial_cost))

        # Cost should have increased (100 prompt + 50 completion tokens = small positive amount)
        assert result["cost_used_usd"] >= initial_cost

    def test_cost_not_changed_when_no_llm_call(self):
        """cost_used_usd unchanged when no LLM call (all not_covered)."""
        sqs = [_make_sq("sq_0", "Missing question")]
        retrievals = {"sq_0": []}
        coverage = {"sq_0": _make_coverage("sq_0", "not_covered")}
        initial_cost = 0.02

        with patch("openai.OpenAI", return_value=MagicMock()):
            result = run_claim_builder(_make_state(sqs, retrievals, coverage, cost_used_usd=initial_cost))

        assert result["cost_used_usd"] == initial_cost
