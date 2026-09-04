"""
Tests for the planner node — Phase 1 Foundations.

Mocks the OpenAI client so no real API calls are made.
Tests retry logic by patching _call_planner for run_planner tests,
and patching openai.OpenAI for lower-level LLM call tests.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from src.agent.nodes.planner import _parse_sub_questions, run_planner
from src.evidence.claims import SubQuestion


def _make_state(**overrides) -> dict:
    base = {
        "request_id": "test-req-001",
        "query": "How does the Bank of England's climate stress test compare to Ofgem's approach?",
        "task_type": "cross_doc",
        "sub_questions": [],
        "retrievals": {},
        "coverage": {},
        "claims": [],
        "verified_claims": [],
        "steps_used": 0,
        "cost_used_usd": 0.0,
        "time_used_s": 0.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }
    base.update(overrides)
    return base


_VALID_SQS = [
    SubQuestion(id="sq_0", question="What is the BoE climate stress test methodology?", required_source="BoE"),
    SubQuestion(id="sq_1", question="What is Ofgem's approach to carbon pricing?", required_source="Ofgem"),
]

_VALID_SQ_JSON = json.dumps(
    [
        {
            "id": "sq_0",
            "question": "What is the BoE climate stress test methodology?",
            "required_source": "BoE",
            "task_type": "factual",
        },
        {
            "id": "sq_1",
            "question": "What is Ofgem's approach to carbon pricing?",
            "required_source": "Ofgem",
            "task_type": "factual",
        },
    ]
)


class TestRunPlannerViaCallPlanner:
    """Tests for run_planner — patch _call_planner to isolate retry logic."""

    def test_valid_sub_questions_returned(self):
        """First _call_planner attempt succeeds → sub_questions in result."""
        with patch("src.agent.nodes.planner._call_planner", return_value=_VALID_SQS):
            result = run_planner(_make_state())

        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 2
        for sq in result["sub_questions"]:
            assert isinstance(sq, SubQuestion)

    def test_steps_used_incremented(self):
        """steps_used is incremented by 1 on success."""
        with patch("src.agent.nodes.planner._call_planner", return_value=_VALID_SQS):
            result = run_planner(_make_state(steps_used=3))

        assert result["steps_used"] == 4

    def test_required_source_preserved(self):
        """required_source is correctly mapped to SubQuestion."""
        with patch("src.agent.nodes.planner._call_planner", return_value=_VALID_SQS):
            result = run_planner(_make_state())

        assert result["sub_questions"][0].required_source == "BoE"

    def test_malformed_json_triggers_retry(self):
        """First attempt returns None → retry (second call) is made."""
        call_count = [0]

        def fake_call_planner(query, strict):
            call_count[0] += 1
            if call_count[0] == 1:
                return None  # first attempt fails
            return _VALID_SQS  # retry succeeds

        with patch("src.agent.nodes.planner._call_planner", side_effect=fake_call_planner):
            result = run_planner(_make_state())

        assert call_count[0] == 2, "Expected exactly two _call_planner calls (first fail + retry)"
        assert "sub_questions" in result
        assert len(result["sub_questions"]) == 2

    def test_both_attempts_fail_returns_fallback(self):
        """Both _call_planner attempts return None → termination_reason='fallback_to_fast'."""
        with patch("src.agent.nodes.planner._call_planner", return_value=None):
            result = run_planner(_make_state())

        assert result.get("termination_reason") == "fallback_to_fast"
        assert "sub_questions" not in result

    def test_fallback_still_increments_steps(self):
        """Even on fallback, steps_used is incremented."""
        with patch("src.agent.nodes.planner._call_planner", return_value=None):
            result = run_planner(_make_state(steps_used=2))

        assert result["steps_used"] == 3


class TestParseSubQuestions:
    """Unit tests for the JSON parsing helper."""

    def test_valid_json_list_parsed(self):
        """Valid JSON array → list of SubQuestion."""
        result = _parse_sub_questions(_VALID_SQ_JSON)
        assert result is not None
        assert len(result) == 2
        assert isinstance(result[0], SubQuestion)

    def test_dict_wrapping_list_parsed(self):
        """JSON object wrapping a list is unpacked."""
        wrapped = json.dumps(
            {"sub_questions": [{"id": "sq_0", "question": "Q?", "required_source": None, "task_type": "factual"}]}
        )
        result = _parse_sub_questions(wrapped)
        assert result is not None
        assert len(result) == 1

    def test_bad_json_returns_none(self):
        """Malformed JSON → None."""
        assert _parse_sub_questions("not valid json {{") is None

    def test_empty_list_returns_none(self):
        """Empty JSON array → None (no usable sub-questions)."""
        assert _parse_sub_questions("[]") is None

    def test_max_six_sub_questions(self):
        """More than 6 items are truncated to 6."""
        many = json.dumps(
            [
                {"id": f"sq_{i}", "question": f"Q{i}?", "required_source": None, "task_type": "factual"}
                for i in range(10)
            ]
        )
        result = _parse_sub_questions(many)
        assert result is not None
        assert len(result) <= 6

    def test_id_auto_assigned_when_missing(self):
        """Items missing 'id' get a generated id."""
        no_id = json.dumps([{"question": "What is X?", "required_source": None, "task_type": "factual"}])
        result = _parse_sub_questions(no_id)
        assert result is not None
        assert result[0].id == "sq_0"
