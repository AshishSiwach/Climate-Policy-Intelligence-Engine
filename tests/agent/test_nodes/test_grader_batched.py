"""Unit tests for deterministic cross-encoder coverage grading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.agent.nodes.grader import (
    COVERED_THRESHOLD,
    PARTIAL_THRESHOLD,
    _score_subquestion,
    run_grader,
)
from src.evidence.claims import Coverage, SubQuestion


def _make_state(sub_questions: list, retrievals: dict, steps_used: int = 2) -> dict:
    return {
        "sub_questions": sub_questions,
        "retrievals": retrievals,
        "steps_used": steps_used,
    }


def _encoder_with(scores: list[float]) -> MagicMock:
    encoder = MagicMock()
    encoder.predict.return_value = scores
    return encoder


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (COVERED_THRESHOLD, "covered"),
        (PARTIAL_THRESHOLD, "partial"),
        (PARTIAL_THRESHOLD - 0.01, "not_covered"),
    ],
)
def test_score_boundaries(score: float, expected: str):
    encoder = _encoder_with([score])

    max_score, status, gap_reason = _score_subquestion(
        encoder,
        "What is the target?",
        [{"text": "The target is net zero by 2050."}],
    )

    assert max_score == score
    assert status == expected
    assert (gap_reason is None) is (expected == "covered")


def test_uses_maximum_passage_score():
    encoder = _encoder_with([-8.0, COVERED_THRESHOLD + 0.5, 0.0])

    max_score, status, _ = _score_subquestion(
        encoder,
        "What is the target?",
        [{"text": "irrelevant"}, {"text": "relevant"}, {"text": "partial"}],
    )

    assert max_score == COVERED_THRESHOLD + 0.5
    assert status == "covered"
    encoder.predict.assert_called_once_with(
        [
            ("What is the target?", "irrelevant"),
            ("What is the target?", "relevant"),
            ("What is the target?", "partial"),
        ]
    )


def test_empty_retrieval_is_not_covered_without_model_call():
    encoder = MagicMock()

    max_score, status, gap_reason = _score_subquestion(encoder, "Question?", [])

    assert max_score == -999.0
    assert status == "not_covered"
    assert gap_reason == "No passages retrieved for this sub-question"
    encoder.predict.assert_not_called()


def test_blank_passages_are_not_covered_without_model_call():
    encoder = MagicMock()

    max_score, status, gap_reason = _score_subquestion(
        encoder,
        "Question?",
        [{"text": "  "}, {"passage": ""}],
    )

    assert max_score == -999.0
    assert status == "not_covered"
    assert gap_reason == "All retrieved passages are empty"
    encoder.predict.assert_not_called()


def test_run_grader_returns_coverage_and_increments_steps():
    sqs = [
        SubQuestion(id="sq_0", question="UK target?"),
        SubQuestion(id="sq_1", question="Paris commitment?"),
    ]
    retrievals = {
        "sq_0": [{"text": "Net zero by 2050."}],
        "sq_1": [{"text": "Nationally determined contributions."}],
    }
    encoder = MagicMock()
    encoder.predict.side_effect = [
        [COVERED_THRESHOLD + 1],
        [PARTIAL_THRESHOLD + 1],
    ]

    with patch("src.agent.nodes.grader._get_encoder", return_value=encoder):
        result = run_grader(_make_state(sqs, retrievals, steps_used=3))

    assert result["steps_used"] == 4
    assert result["coverage"]["sq_0"].status == "covered"
    assert result["coverage"]["sq_1"].status == "partial"
    assert all(isinstance(value, Coverage) for value in result["coverage"].values())


def test_encoder_failure_defaults_all_subquestions_to_covered():
    sqs = [SubQuestion(id="sq_0", question="UK target?")]
    encoder = MagicMock()
    encoder.predict.side_effect = RuntimeError("model unavailable")

    with patch("src.agent.nodes.grader._get_encoder", return_value=encoder):
        result = run_grader(_make_state(sqs, {"sq_0": [{"text": "evidence"}]}))

    assert result["coverage"]["sq_0"].status == "covered"


def test_no_subquestions_returns_empty_coverage_without_loading_encoder():
    with patch("src.agent.nodes.grader._get_encoder") as get_encoder:
        result = run_grader(_make_state([], {}, steps_used=4))

    assert result == {"coverage": {}, "steps_used": 5}
    get_encoder.assert_not_called()
