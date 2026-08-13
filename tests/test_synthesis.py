"""
Unit tests for synthesis — Pydantic schemas, citation verification,
LLM refusal branch, and the empty-chunks guard.

Removed in Week 5:
  - RRF-threshold short-circuit tests (Step 2b — mechanism deleted)
  - Confidence + confidence_signals tests (Step 2d — pipeline-derived
    confidence removed; AUC 0.668 was too weak for a user promise)

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
    PROMPT_REGISTRY,
    PROMPT_VERSION,
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


# ---------------------------------------------------------------------------
# Prompt versioning (Week 5 Step 3a)
# ---------------------------------------------------------------------------

def test_prompt_registry_contains_v1_and_variants():
    """PROMPT_REGISTRY must include v1 baseline + the 2 v2 variants."""
    assert "v1" in PROMPT_REGISTRY
    assert "v2_crossdoc" in PROMPT_REGISTRY
    assert "v2_numeric" in PROMPT_REGISTRY


def test_default_prompt_version_is_v2_numeric():
    """v2_numeric shipped as default after the Step 3a A/B (best aggregate
    Correctness, zero regressions). Guardrail against silent version drift."""
    assert PROMPT_VERSION == "v2_numeric"


def test_synthesiser_default_uses_shipped_prompt():
    synth = Synthesiser(api_key="test-key-not-used")
    assert synth.prompt_version == PROMPT_VERSION
    assert synth.system_prompt == PROMPT_REGISTRY[PROMPT_VERSION]


def test_synthesiser_accepts_alternate_prompt_version():
    synth = Synthesiser(api_key="test-key-not-used", prompt_version="v2_crossdoc")
    assert synth.prompt_version == "v2_crossdoc"
    assert synth.system_prompt == PROMPT_REGISTRY["v2_crossdoc"]


def test_synthesiser_rejects_unknown_prompt_version():
    with pytest.raises(ValueError, match="Unknown prompt_version"):
        Synthesiser(api_key="test-key-not-used", prompt_version="v99_nonexistent")


def test_synthesise_result_includes_prompt_version_on_empty_chunks():
    """Empty-chunks guard path must still tag prompt_version — needed for log slicing."""
    synth = Synthesiser(api_key="test-key-not-used", prompt_version="v2_numeric")
    with patch.object(synth._client.beta.chat.completions, "parse"):
        result = synth.synthesise("anything", [])
    assert result["prompt_version"] == "v2_numeric"


def test_synthesise_uses_the_configured_prompt_at_call_time(sample_chunks):
    """The LLM call must receive the version-specific system prompt, not v1 unconditionally."""
    synth = Synthesiser(api_key="test-key-not-used", prompt_version="v2_crossdoc")

    llm_response = LLMResponse(answer="ok", citations=[], contradictions=[])
    normal_message = MagicMock()
    normal_message.refusal = None
    normal_message.parsed = llm_response
    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=normal_message)]
    fake_response.usage = MagicMock(prompt_tokens=100, completion_tokens=10)

    with patch.object(synth._client.beta.chat.completions, "parse", return_value=fake_response) as mock_parse:
        result = synth.synthesise("q", sample_chunks)

    # First positional call has kwargs, including messages=[system, user]
    call_kwargs = mock_parse.call_args.kwargs
    system_msg = next(m for m in call_kwargs["messages"] if m["role"] == "system")
    assert system_msg["content"] == PROMPT_REGISTRY["v2_crossdoc"]
    assert result["prompt_version"] == "v2_crossdoc"
