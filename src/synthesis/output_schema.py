# ruff: noqa: E501
# Field descriptions are intentionally verbose — they document the schema for
# the LLM via Structured Outputs and must not be truncated.
"""
Pydantic output schema for the analyst brief.

Two Citation classes on purpose:
  LLMCitation — what the LLM must return via Structured Outputs. Only fields
                the LLM can legitimately produce (doc_id, passage, page).
  Citation   — final citation in the AnalystBrief, enriched by the pipeline
                with publication_date pulled from the retrieved chunk's
                metadata. Prevents the LLM from fabricating dates.

Confidence: removed after calibration (Week 5 Step 2d). On 47 queries
the pipeline-derived signals had at best AUC 0.668 for predicting
correctness (95% CI overlapping random). Shipping a weak signal as a user
promise was worse than shipping nothing. Future work: re-introduce once
signals are strong (semantic_sim + doc_aware_margin candidates, n>=100
ground-truth data, AUC >= 0.75). See docs/week5_failure_analysis.md § 2d.
"""

from pydantic import BaseModel, Field


class LLMCitation(BaseModel):
    """What the LLM returns via Structured Outputs. No enriched metadata."""

    doc_id: str = Field(..., description="Source document identifier, e.g. OFGEM_SMART_SECURE_2025")
    passage: str = Field(..., description="Verbatim quote from the retrieved chunk that supports the claim")
    page: int = Field(..., ge=1, description="Page number in the source document")


class Citation(BaseModel):
    """Final citation in AnalystBrief. Enriched from retrieved chunk metadata."""

    doc_id: str
    passage: str
    page: int = Field(..., ge=1)
    publication_date: str | None = Field(
        default=None,
        description="Source document publication year/date, injected by pipeline from chunk metadata. Analysts use this to gauge freshness.",
    )


class Contradiction(BaseModel):
    """Experimental — LLM self-report of conflicting claims across sources."""

    doc_a: str
    doc_b: str
    summary: str


class LLMResponse(BaseModel):
    """Raw JSON shape returned by the LLM."""

    answer: str
    citations: list[LLMCitation] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


class AnalystBrief(BaseModel):
    """Final structured brief returned to the user."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
