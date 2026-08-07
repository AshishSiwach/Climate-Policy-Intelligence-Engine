"""
Unit tests for synthesis — Pydantic schemas, citation verification,
LLM refusal branch, and the empty-chunks guard.

Removed in Week 5:
  - RRF-threshold short-circuit tests (Step 2b — mechanism deleted)
  - Confidence + confidence_signals tests (Step 2d — pipeline-derived
    confidence removed from v1; AUC 0.668 was too weak for a user promise)

Both removals are documented in docs/week5_failure_analysis.md.

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
    OUT_OF_CORPUS_ANSWER,
    Synthesiser,
    _verify_citations,
)


# ---------------------------------------------------------------------------
# Pydantic schema validation
# ---------------------------------------------------------------------------

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
    assert result["cost_usd"] == 0.0


def test_synthesise_handles_llm_refusal(sample_chunks):
    """When LLM emits message.refusal, return canonical brief."""
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
