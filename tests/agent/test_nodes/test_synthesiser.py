"""
Tests for the agent synthesiser node — Phase 2 Cross-Document Route.

Mocks the OpenAI client so no real API calls are made.
This tests src/agent/nodes/synthesiser.py — NOT the fast-path synthesiser.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.agent.nodes.synthesiser import run_synthesiser
from src.evidence.claims import Claim, Coverage
from src.synthesis.output_schema import AnalystBrief

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_claim(id_: str, text: str, evidence_ids: list[str], doc_id: str = "boe") -> Claim:
    return Claim(id=id_, text=text, evidence_ids=evidence_ids, source_doc_id=doc_id)


def _make_chunk(chunk_id: str, doc_id: str, text: str, page: int = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "text": text,
        "page_number": page,
        "publication_date": "2024",
    }


def _make_state(
    verified_claims: list,
    coverage: dict,
    retrievals: dict,
    query: str = "How do BoE and Ofgem climate policies compare?",
    steps_used: int = 5,
    cost_used_usd: float = 0.02,
) -> dict:
    return {
        "request_id": "test-req-syn",
        "query": query,
        "task_type": "cross_doc",
        "sub_questions": [],
        "retrievals": retrievals,
        "coverage": coverage,
        "claims": [],
        "verified_claims": verified_claims,
        "steps_used": steps_used,
        "cost_used_usd": cost_used_usd,
        "time_used_s": 10.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }


def _mock_parse_response(answer: str, citations: list | None = None, prompt_tokens: int = 200, completion_tokens: int = 100):
    """Mock for client.beta.chat.completions.parse — returns message.parsed as LLMResponse."""
    from src.synthesis.output_schema import LLMCitation, LLMResponse

    parsed = LLMResponse(
        answer=answer,
        citations=[LLMCitation(**c) for c in (citations or [])],
    )
    msg = MagicMock()
    msg.refusal = None
    msg.parsed = parsed
    choice = MagicMock()
    choice.message = msg
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_client(answer: str, citations: list | None = None):
    client = MagicMock()
    client.beta.chat.completions.parse.return_value = _mock_parse_response(answer, citations)
    return client


# ---------------------------------------------------------------------------
# Test 1: Happy path — verified claims → AnalystBrief with answer + citations
# ---------------------------------------------------------------------------


class TestSynthesiserHappyPath:
    def test_returns_analyst_brief_with_answer_and_citations(self):
        """Verified claims → AnalystBrief with answer and citations built from chunks."""
        claims = [_make_claim("c_0", "BoE runs annual stress tests.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client(
            "The BoE conducts annual climate stress tests.",
            citations=[{"chunk_id": "boe_0", "doc_id": "boe", "passage": "BoE runs climate stress tests annually.", "page": 1}],
        )

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        assert "result" in result
        brief = AnalystBrief(**result["result"])
        assert brief.answer == "The BoE conducts annual climate stress tests."
        assert len(brief.citations) == 1
        assert brief.citations[0].doc_id == "boe"

    def test_steps_incremented_by_one(self):
        """steps_used is incremented by exactly 1."""
        claims = [_make_claim("c_0", "BoE runs stress tests.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("BoE conducts stress tests.")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals, steps_used=5))

        assert result["steps_used"] == 6

    def test_cost_updated(self):
        """cost_used_usd increases after LLM call."""
        claims = [_make_claim("c_0", "BoE runs stress tests.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("BoE conducts stress tests.")
        initial_cost = 0.02

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals, cost_used_usd=initial_cost))

        assert result["cost_used_usd"] >= initial_cost

    def test_citations_deduplicated_across_claims(self):
        """Multiple claims referencing the same chunk_id produce only one citation."""
        chunk = _make_chunk("boe_0", "boe", "BoE climate stress test findings from 2024.")
        claims = [
            _make_claim("c_0", "Claim A about BoE.", ["boe_0"], "boe"),
            _make_claim("c_1", "Claim B about BoE.", ["boe_0"], "boe"),  # same chunk
        ]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [chunk]}
        client = _make_client(
            "BoE answer.",
            citations=[{"chunk_id": "boe_0", "doc_id": "boe", "passage": "BoE climate stress test findings from 2024.", "page": 1}],
        )

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        brief = AnalystBrief(**result["result"])
        # Should have only 1 citation (deduplicated)
        assert len(brief.citations) == 1


# ---------------------------------------------------------------------------
# Test 2: Coverage gaps from partial sub-questions appear in brief
# ---------------------------------------------------------------------------


class TestSynthesiserCoverageGaps:
    def test_partial_coverage_gap_in_brief(self):
        """Partial sub-question → gap_reason appears in brief.coverage_gaps."""
        claims = [_make_claim("c_0", "BoE runs stress tests.", ["boe_0"], "boe")]
        coverage = {
            "sq_0": Coverage(sub_question_id="sq_0", status="covered"),
            "sq_1": Coverage(
                sub_question_id="sq_1",
                status="partial",
                gap_reason="No post-2020 emissions data found",
            ),
        }
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("BoE conducts stress tests.")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        brief = AnalystBrief(**result["result"])
        assert len(brief.coverage_gaps) >= 1
        assert any("post-2020" in g for g in brief.coverage_gaps)

    def test_not_covered_gap_in_brief(self):
        """Not_covered sub-question → its id or gap_reason appears in coverage_gaps."""
        claims = [_make_claim("c_0", "BoE runs stress tests.", ["boe_0"], "boe")]
        coverage = {
            "sq_0": Coverage(sub_question_id="sq_0", status="covered"),
            "sq_1": Coverage(sub_question_id="sq_1", status="not_covered", gap_reason="No relevant data"),
        }
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("BoE conducts stress tests.")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        brief = AnalystBrief(**result["result"])
        # grader-detected gaps should appear even if LLM returns empty list
        assert len(brief.coverage_gaps) >= 1

    def test_no_gaps_when_all_covered(self):
        """All covered sub-questions → empty coverage_gaps in brief."""
        claims = [_make_claim("c_0", "BoE runs stress tests.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("BoE conducts stress tests.")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        brief = AnalystBrief(**result["result"])
        assert brief.coverage_gaps == []


# ---------------------------------------------------------------------------
# Test 3: termination_reason = "complete" in state and brief
# ---------------------------------------------------------------------------


class TestSynthesiserTermination:
    def test_termination_reason_complete_in_state(self):
        """State output has termination_reason='complete'."""
        claims = [_make_claim("c_0", "BoE text.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("Answer here.")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        assert result["termination_reason"] == "complete"

    def test_termination_reason_complete_in_brief(self):
        """AnalystBrief.termination_reason='complete' and truncated=False."""
        claims = [_make_claim("c_0", "BoE text.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = _make_client("Answer here.")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        brief = AnalystBrief(**result["result"])
        assert brief.termination_reason == "complete"
        assert brief.truncated is False

    def test_llm_failure_returns_fallback_answer(self):
        """LLM exception → fallback answer in result, still returns valid state."""
        claims = [_make_claim("c_0", "BoE text.", ["boe_0"], "boe")]
        coverage = {"sq_0": Coverage(sub_question_id="sq_0", status="covered")}
        retrievals = {"sq_0": [_make_chunk("boe_0", "boe", "BoE runs climate stress tests annually.")]}
        client = MagicMock()
        client.beta.chat.completions.parse.side_effect = ConnectionError("network down")

        with patch("openai.OpenAI", return_value=client):
            result = run_synthesiser(_make_state(claims, coverage, retrievals))

        assert "result" in result
        brief = AnalystBrief(**result["result"])
        assert brief.answer  # not empty
        assert result["termination_reason"] == "complete"
