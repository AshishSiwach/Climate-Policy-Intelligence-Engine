"""
Unit tests for LLMJudge — schema, prompt construction, scoring path, refusal handling.

All OpenAI calls mocked; no network.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from evaluation.judge import (
    JudgeScore,
    LLMJudge,
    _build_judge_message,
    _estimate_cost,
)


# ---------------------------------------------------------------------------
# JudgeScore Pydantic schema
# ---------------------------------------------------------------------------

def _valid_score(**overrides) -> dict:
    """Base valid payload — override fields to test bounds."""
    return {
        "correctness": 5, "correctness_rationale": "matches reference",
        "faithfulness": 5, "faithfulness_rationale": "supported by chunks",
        "completeness": 5, "completeness_rationale": "all key points covered",
        "refusal_appropriateness": 5, "refusal_appropriateness_rationale": "answered appropriately",
        **overrides,
    }


def test_judge_score_accepts_valid_payload():
    JudgeScore(**_valid_score())   # must not raise


def test_judge_score_rejects_score_below_1():
    with pytest.raises(ValidationError):
        JudgeScore(**_valid_score(correctness=0))


def test_judge_score_rejects_score_above_5():
    with pytest.raises(ValidationError):
        JudgeScore(**_valid_score(faithfulness=6))


def test_judge_score_requires_all_dimensions():
    payload = _valid_score()
    del payload["completeness"]
    with pytest.raises(ValidationError):
        JudgeScore(**payload)


# ---------------------------------------------------------------------------
# _build_judge_message — prompt construction
# ---------------------------------------------------------------------------

def _mk_chunk(doc_id="OFGEM_TEST", page=25, text="Sample text."):
    return {"doc_id": doc_id, "page_number": page, "text": text}


def test_build_judge_message_includes_all_sections():
    msg = _build_judge_message(
        question="Q?",
        query_type="factual",
        expected_answer="Reference answer.",
        retrieved_chunks=[_mk_chunk()],
        generated_answer="Generated answer.",
    )
    assert "QUESTION:" in msg and "Q?" in msg
    assert "QUERY_TYPE: factual" in msg
    assert "REFERENCE_ANSWER:" in msg and "Reference answer." in msg
    assert "RETRIEVED_CHUNKS:" in msg and "OFGEM_TEST" in msg and "page=25" in msg
    assert "GENERATED_ANSWER:" in msg and "Generated answer." in msg


def test_build_judge_message_empty_chunks_shows_short_circuit_note():
    msg = _build_judge_message(
        question="Q?",
        query_type="negative",
        expected_answer="Corpus does not contain...",
        retrieved_chunks=[],
        generated_answer="Corpus does not contain...",
    )
    assert "(no chunks" in msg


def test_build_judge_message_renders_multiple_chunks_in_order():
    msg = _build_judge_message(
        question="Q?",
        query_type="factual",
        expected_answer="ref",
        retrieved_chunks=[
            _mk_chunk("A", page=1, text="alpha"),
            _mk_chunk("B", page=2, text="beta"),
        ],
        generated_answer="gen",
    )
    assert "[Chunk 1]" in msg
    assert "[Chunk 2]" in msg
    assert msg.index("[Chunk 1]") < msg.index("[Chunk 2]")


# ---------------------------------------------------------------------------
# LLMJudge.score — happy path (mocked LLM)
# ---------------------------------------------------------------------------

def _stub_judge() -> LLMJudge:
    return LLMJudge(api_key="test-key-not-used")


def _mock_response(scores_kwargs: dict, prompt_tokens=500, completion_tokens=200):
    """Build a fake OpenAI response wrapping a JudgeScore."""
    parsed = JudgeScore(**_valid_score(**scores_kwargs))
    message = MagicMock()
    message.refusal = None
    message.parsed = parsed
    response = MagicMock()
    response.choices = [MagicMock(message=message)]
    response.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return response


def test_score_happy_path_returns_validated_scores():
    judge = _stub_judge()
    fake_resp = _mock_response({"correctness": 4, "faithfulness": 5})

    with patch.object(judge._client.beta.chat.completions, "parse", return_value=fake_resp):
        result = judge.score(
            question="q", query_type="factual", expected_answer="ref",
            retrieved_chunks=[_mk_chunk()], generated_answer="gen",
        )

    assert result["scores"].correctness == 4
    assert result["scores"].faithfulness == 5
    assert result["prompt_tokens"] == 500
    assert result["completion_tokens"] == 200


def test_score_returns_zero_cost_for_unknown_model():
    """gpt-5.4-mini pricing is unknown until confirmed — cost logged as 0 with warning."""
    judge = LLMJudge(model="gpt-5.4-mini", api_key="test-key-not-used")
    fake_resp = _mock_response({}, prompt_tokens=1000, completion_tokens=200)

    with patch.object(judge._client.beta.chat.completions, "parse", return_value=fake_resp):
        result = judge.score(
            question="q", query_type="factual", expected_answer="ref",
            retrieved_chunks=[], generated_answer="gen",
        )

    assert result["cost_usd"] == 0.0


def test_score_computes_cost_for_known_model():
    judge = LLMJudge(model="gpt-4o-mini", api_key="test-key-not-used")
    fake_resp = _mock_response({}, prompt_tokens=1_000_000, completion_tokens=1_000_000)

    with patch.object(judge._client.beta.chat.completions, "parse", return_value=fake_resp):
        result = judge.score(
            question="q", query_type="factual", expected_answer="ref",
            retrieved_chunks=[], generated_answer="gen",
        )

    # 1M input tokens at $0.15 + 1M output tokens at $0.60 = $0.75
    assert result["cost_usd"] == pytest.approx(0.75, abs=0.001)


# ---------------------------------------------------------------------------
# LLMJudge.score — refusal branch
# ---------------------------------------------------------------------------

def test_score_handles_judge_refusal_gracefully():
    """If the judge model refuses to score, return scores=None + refusal_reason."""
    judge = _stub_judge()

    refusal_message = MagicMock()
    refusal_message.refusal = "I decline to evaluate this content."
    refusal_message.parsed = None
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=refusal_message)]
    fake_response.usage = MagicMock(prompt_tokens=200, completion_tokens=20)

    with patch.object(judge._client.beta.chat.completions, "parse", return_value=fake_response):
        result = judge.score(
            question="q", query_type="factual", expected_answer="ref",
            retrieved_chunks=[_mk_chunk()], generated_answer="gen",
        )

    assert result["scores"] is None
    assert "refusal_reason" in result
    assert result["prompt_tokens"] == 200


# ---------------------------------------------------------------------------
# Timeout + retry wiring
# ---------------------------------------------------------------------------

def test_judge_client_has_timeout_and_retries():
    """Match the SDK resilience posture used by Synthesiser."""
    judge = _stub_judge()
    assert judge._client.timeout == 30.0
    assert judge._client.max_retries == 2


# ---------------------------------------------------------------------------
# _estimate_cost helper
# ---------------------------------------------------------------------------

def test_estimate_cost_known_model():
    # 500 input tokens at $0.15/M + 200 output at $0.60/M
    cost = _estimate_cost("gpt-4o-mini", 500, 200)
    expected = (500 / 1_000_000) * 0.15 + (200 / 1_000_000) * 0.60
    assert cost == pytest.approx(expected, abs=1e-6)


def test_estimate_cost_unknown_model_returns_zero():
    assert _estimate_cost("some-model-i-dont-know", 1000, 500) == 0.0
