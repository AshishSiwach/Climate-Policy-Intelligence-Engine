"""
Unit tests for synthesis — Pydantic schemas, citation verification,
pipeline-derived confidence, LLM refusal branch, and the empty-chunks guard.

The RRF-threshold short-circuit was removed in Week 5 Step 2b (see
docs/week5_failure_analysis.md); tests that exercised that branch are gone.

All OpenAI calls are mocked; no network I/O.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from synthesis.output_schema import (
    AnalystBrief,
    Citation,
    Contradiction,
    LLMCitation,
    LLMResponse,
)
from synthesis.synthesiser import (
    CITATION_SATURATION,
    MARGIN_NORMALISER,
    OUT_OF_CORPUS_ANSWER,
    SCORE_NORMALISER,
    Synthesiser,
    _compute_confidence,
    _verify_citations,
)


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

def test_analyst_brief_confidence_must_be_between_0_and_1():
    with pytest.raises(ValidationError):
        AnalystBrief(answer="x", citations=[], confidence=1.5, contradictions=[])
    with pytest.raises(ValidationError):
        AnalystBrief(answer="x", citations=[], confidence=-0.1, contradictions=[])


def test_analyst_brief_confidence_is_required():
    with pytest.raises(ValidationError):
        AnalystBrief(answer="x", citations=[], contradictions=[])   # no confidence


def test_citation_page_must_be_ge_1():
    with pytest.raises(ValidationError):
        Citation(doc_id="X", passage="p", page=0)


def test_citation_publication_date_defaults_to_none():
    c = Citation(doc_id="X", passage="p", page=1)
    assert c.publication_date is None


def test_llm_citation_has_no_publication_date_field():
    """LLM must not be asked for publication_date. Enforced by schema separation."""
    assert "publication_date" not in LLMCitation.model_fields
    assert "publication_date" in Citation.model_fields


def test_llm_response_uses_llm_citation_not_citation():
    """LLMResponse.citations must be typed as LLMCitation to keep publication_date out of the LLM schema."""
    citations_field = LLMResponse.model_fields["citations"]
    # The annotation is list[LLMCitation]; extract via string check to be Pydantic-version-agnostic
    assert "LLMCitation" in str(citations_field.annotation)


# ---------------------------------------------------------------------------
# _verify_citations — the fact-check layer
# ---------------------------------------------------------------------------

def test_verify_citations_keeps_matching(sample_chunks):
    """A citation whose passage appears in a retrieved chunk survives."""
    llm_citation = LLMCitation(
        doc_id="OFGEM_TEST",
        passage="Ofgem proposes new load control licensing requirements",
        page=1,
    )
    verified = _verify_citations([llm_citation], sample_chunks)
    assert len(verified) == 1
    assert isinstance(verified[0], Citation)


def test_verify_citations_drops_fabricated(sample_chunks):
    """A citation whose passage is nowhere in the chunks gets dropped."""
    fabricated = LLMCitation(
        doc_id="OFGEM_TEST",
        passage="The moon is made of cheese and Ofgem regulates dairy",
        page=1,
    )
    verified = _verify_citations([fabricated], sample_chunks)
    assert verified == []


def test_verify_citations_case_insensitive(sample_chunks):
    citation = LLMCitation(
        doc_id="OFGEM_TEST",
        passage="OFGEM PROPOSES NEW LOAD CONTROL LICENSING",
        page=1,
    )
    verified = _verify_citations([citation], sample_chunks)
    assert len(verified) == 1


def test_verify_citations_injects_publication_date(sample_chunks):
    """The pipeline (not the LLM) fills in publication_date from the matched chunk."""
    llm_citation = LLMCitation(
        doc_id="BOE_TEST",
        passage="UK banks faced aggregate losses of £334 billion under the CBES",
        page=53,
    )
    verified = _verify_citations([llm_citation], sample_chunks)
    assert len(verified) == 1
    assert verified[0].publication_date == "2021", (
        "publication_date should be pulled from the matched BoE chunk"
    )


def test_verify_citations_empty_input_returns_empty():
    assert _verify_citations([], []) == []


# ---------------------------------------------------------------------------
# _compute_confidence — pipeline-derived confidence
# ---------------------------------------------------------------------------

def test_compute_confidence_returns_score_and_all_four_signals(sample_chunks):
    citations = [Citation(doc_id="OFGEM_TEST", passage="x", page=1)]
    confidence, signals = _compute_confidence(sample_chunks, citations)

    assert 0.0 <= confidence <= 1.0
    for key in ("score_signal", "agreement_signal", "margin_signal",
                "citation_signal", "llm_refusal"):
        assert key in signals


def test_compute_confidence_empty_chunks_returns_zero():
    confidence, signals = _compute_confidence([], [])
    assert confidence == 0.0
    for numeric_key in ("score_signal", "agreement_signal", "margin_signal", "citation_signal"):
        assert signals[numeric_key] == 0.0


def test_compute_confidence_uses_equal_weights():
    """Formula: 0.25*(score + agreement + margin + citation). Verify from known signals."""
    # Craft chunks so signals are exactly known
    chunks = [
        {"rrf_score": 0.05, "doc_id": "A", "chunk_index": 0, "text": "x"},   # score=1.0
        {"rrf_score": 0.05, "doc_id": "A", "chunk_index": 1, "text": "x"},   # agreement=1.0, margin=0
        {"rrf_score": 0.05, "doc_id": "A", "chunk_index": 2, "text": "x"},
    ]
    citations = [Citation(doc_id="A", passage="x", page=1)] * 3   # citation=1.0

    confidence, signals = _compute_confidence(chunks, citations)
    expected = 0.25 * (1.0 + 1.0 + 0.0 + 1.0)   # score + agreement + margin + citation
    assert confidence == pytest.approx(expected, abs=0.001)


def test_compute_confidence_single_chunk_margin_maxed():
    """Only one chunk → margin undefined → default to 1.0 per implementation."""
    chunks = [{"rrf_score": 0.05, "doc_id": "A", "chunk_index": 0, "text": "x"}]
    _, signals = _compute_confidence(chunks, [])
    assert signals["margin_signal"] == 1.0


def test_compute_confidence_margin_uses_normaliser():
    """(top - top2) / MARGIN_NORMALISER, capped at 1.0."""
    chunks = [
        {"rrf_score": 0.03, "doc_id": "A", "chunk_index": 0, "text": "x"},
        {"rrf_score": 0.025, "doc_id": "A", "chunk_index": 1, "text": "x"},   # gap 0.005
    ]
    _, signals = _compute_confidence(chunks, [])
    expected_margin = min(0.005 / MARGIN_NORMALISER, 1.0)
    assert signals["margin_signal"] == pytest.approx(expected_margin, abs=0.001)


def test_compute_confidence_citation_saturates_at_configured_cap():
    """Citation signal caps at 1.0 once we hit CITATION_SATURATION citations."""
    chunks = [{"rrf_score": 0.05, "doc_id": "A", "chunk_index": 0, "text": "x"}]
    many_citations = [Citation(doc_id="A", passage="x", page=1)] * (CITATION_SATURATION + 5)
    _, signals = _compute_confidence(chunks, many_citations)
    assert signals["citation_signal"] == 1.0


def test_compute_confidence_score_uses_normaliser():
    """score_signal = min(top_rrf / SCORE_NORMALISER, 1.0)."""
    chunks = [{"rrf_score": SCORE_NORMALISER / 2, "doc_id": "A", "chunk_index": 0, "text": "x"}]
    _, signals = _compute_confidence(chunks, [])
    assert signals["score_signal"] == pytest.approx(0.5, abs=0.001)


# ---------------------------------------------------------------------------
# Synthesiser — empty-chunks guard and LLM-refusal branch, no live API
# ---------------------------------------------------------------------------

def _stub_synthesiser() -> Synthesiser:
    """Instantiate Synthesiser without needing a real API key."""
    return Synthesiser(api_key="test-key-not-used")


def test_synthesise_empty_chunks_short_circuits():
    """Zero retrieved chunks → LLM never called; return canonical refusal."""
    synth = _stub_synthesiser()
    with patch.object(synth._client.beta.chat.completions, "parse") as mock_parse:
        result = synth.synthesise("anything", [])
    mock_parse.assert_not_called()
    assert result["brief"].answer == OUT_OF_CORPUS_ANSWER
    assert result["brief"].confidence == 0.0
    assert result["cost_usd"] == 0.0
    assert result["confidence_signals"]["llm_refusal"] is False


def test_synthesise_handles_llm_refusal(sample_chunks):
    """When LLM emits message.refusal, return canonical brief with llm_refusal signal."""
    synth = _stub_synthesiser()

    refusal_message = MagicMock()
    refusal_message.refusal = "I cannot answer that."
    refusal_message.parsed = None

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=refusal_message)]
    fake_response.usage = MagicMock(prompt_tokens=100, completion_tokens=10)

    with patch.object(synth._client.beta.chat.completions, "parse", return_value=fake_response):
        result = synth.synthesise("some query", sample_chunks)

    assert result["brief"].answer == OUT_OF_CORPUS_ANSWER
    assert result["brief"].confidence == 0.0
    assert result["confidence_signals"]["llm_refusal"] is True
    assert result["refusal_reason"] == "I cannot answer that."


def test_synthesise_normal_path_returns_verified_citations(sample_chunks):
    """Happy path: LLM returns a valid LLMResponse; citations get verified + enriched."""
    synth = _stub_synthesiser()

    llm_response = LLMResponse(
        answer="Ofgem proposes new load control licensing.",
        citations=[
            LLMCitation(
                doc_id="OFGEM_TEST",
                passage="Ofgem proposes new load control licensing requirements",
                page=1,
            )
        ],
        contradictions=[],
    )

    normal_message = MagicMock()
    normal_message.refusal = None
    normal_message.parsed = llm_response

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=normal_message)]
    fake_response.usage = MagicMock(prompt_tokens=500, completion_tokens=50)

    with patch.object(synth._client.beta.chat.completions, "parse", return_value=fake_response):
        result = synth.synthesise("What does Ofgem propose?", sample_chunks)

    brief = result["brief"]
    assert isinstance(brief, AnalystBrief)
    assert brief.answer.startswith("Ofgem")
    assert len(brief.citations) == 1
    assert brief.citations[0].publication_date == "2024", (
        "publication_date should be enriched from the matched Ofgem chunk"
    )
    assert result["cost_usd"] > 0
    assert result["confidence_signals"]["llm_refusal"] is False
