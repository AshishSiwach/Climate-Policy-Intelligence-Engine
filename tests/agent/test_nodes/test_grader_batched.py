"""
Tests for the grader node (batched) — Phase 1 Foundations.

Mocks the OpenAI client so no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from src.agent.nodes.grader import run_grader
from src.evidence.claims import Coverage, SubQuestion


def _make_sq(id_: str, question: str, required_source: str | None = None) -> SubQuestion:
    return SubQuestion(id=id_, question=question, required_source=required_source)


def _make_state(sub_questions: list, retrievals: dict, steps_used: int = 2) -> dict:
    return {
        "request_id": "test-req-grader",
        "query": "test query",
        "task_type": "cross_doc",
        "sub_questions": sub_questions,
        "retrievals": retrievals,
        "coverage": {},
        "claims": [],
        "verified_claims": [],
        "steps_used": steps_used,
        "cost_used_usd": 0.01,
        "time_used_s": 5.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }


def _mock_llm_response(content: str):
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_client(content: str):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_llm_response(content)
    return client


SQ_LIST = [
    _make_sq("sq_0", "What is the BoE climate stress test?", "BoE"),
    _make_sq("sq_1", "What is Ofgem's carbon pricing approach?", "Ofgem"),
]
CHUNKS = {
    "sq_0": [{"text": "The BoE runs annual climate stress tests.", "doc_id": "boe_001", "page_number": 3}],
    "sq_1": [{"text": "Ofgem uses a price cap mechanism.", "doc_id": "ofg_002", "page_number": 7}],
}


class TestGraderAllCovered:
    def test_all_covered_returns_covered_status(self):
        """LLM says all covered → all Coverage objects have status='covered'."""
        llm_output = json.dumps(
            {
                "sq_0": {"status": "covered", "gap_reason": None},
                "sq_1": {"status": "covered", "gap_reason": None},
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state(SQ_LIST, CHUNKS))

        assert "coverage" in result
        coverage = result["coverage"]
        assert coverage["sq_0"].status == "covered"
        assert coverage["sq_1"].status == "covered"
        assert coverage["sq_0"].gap_reason is None
        assert coverage["sq_1"].gap_reason is None

    def test_steps_incremented(self):
        """steps_used is incremented by 1."""
        llm_output = json.dumps({"sq_0": {"status": "covered", "gap_reason": None}})
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state([SQ_LIST[0]], {"sq_0": CHUNKS["sq_0"]}, steps_used=3))

        assert result["steps_used"] == 4


class TestGraderPartial:
    def test_partial_coverage_has_gap_reason(self):
        """One partial → that Coverage has status='partial' with gap_reason."""
        llm_output = json.dumps(
            {
                "sq_0": {"status": "covered", "gap_reason": None},
                "sq_1": {"status": "partial", "gap_reason": "No post-2020 data found"},
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state(SQ_LIST, CHUNKS))

        coverage = result["coverage"]
        assert coverage["sq_0"].status == "covered"
        assert coverage["sq_1"].status == "partial"
        assert coverage["sq_1"].gap_reason == "No post-2020 data found"

    def test_not_covered_has_gap_reason(self):
        """not_covered status preserved with gap_reason."""
        llm_output = json.dumps(
            {
                "sq_0": {"status": "not_covered", "gap_reason": "No relevant chunks retrieved"},
                "sq_1": {"status": "covered", "gap_reason": None},
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state(SQ_LIST, CHUNKS))

        assert result["coverage"]["sq_0"].status == "not_covered"
        assert result["coverage"]["sq_0"].gap_reason == "No relevant chunks retrieved"


class TestGraderFailPermissive:
    def test_unparseable_json_defaults_all_covered(self):
        """Unparseable grader output → all sub-questions treated as 'covered'."""
        client = _make_client("THIS IS NOT JSON AT ALL !@#$")

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state(SQ_LIST, CHUNKS))

        coverage = result["coverage"]
        assert len(coverage) == len(SQ_LIST)
        for sq_id, cov in coverage.items():
            assert cov.status == "covered"

    def test_llm_exception_defaults_all_covered(self):
        """LLM call exception → all sub-questions treated as 'covered'."""
        client = MagicMock()
        client.chat.completions.create.side_effect = ConnectionError("network down")

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state(SQ_LIST, CHUNKS))

        coverage = result["coverage"]
        assert len(coverage) == len(SQ_LIST)
        for cov in coverage.values():
            assert cov.status == "covered"

    def test_empty_sub_questions_returns_empty_coverage(self):
        """No sub-questions → empty coverage dict (no LLM call needed)."""
        result = run_grader(_make_state([], {}))

        assert result["coverage"] == {}

    def test_coverage_objects_are_coverage_instances(self):
        """All returned values are Coverage instances."""
        llm_output = json.dumps(
            {
                "sq_0": {"status": "covered", "gap_reason": None},
                "sq_1": {"status": "partial", "gap_reason": "Gap"},
            }
        )
        client = _make_client(llm_output)

        with patch("openai.OpenAI", return_value=client):
            result = run_grader(_make_state(SQ_LIST, CHUNKS))

        for cov in result["coverage"].values():
            assert isinstance(cov, Coverage)
