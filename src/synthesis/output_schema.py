"""
Pydantic output schema for the analyst brief.

Two Citation classes on purpose:
  LLMCitation — what the LLM must return via Structured Outputs. Only fields
                the LLM can legitimately produce (doc_id, passage, page).
  Citation   — final citation in the AnalystBrief, enriched by the pipeline
                with publication_date pulled from the retrieved chunk's
                metadata. Prevents the LLM from fabricating dates.

The LLM is asked to return LLMResponse (excluding `confidence` — pipeline-derived).
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
    """Experimental in v1 — LLM self-report of conflicting claims across sources."""
    doc_a: str
    doc_b: str
    summary: str


class LLMResponse(BaseModel):
    """Raw JSON shape returned by the LLM. Confidence is added by the pipeline."""
    answer: str
    citations: list[LLMCitation] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


class AnalystBrief(BaseModel):
    """Final structured brief returned to the user."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # confidence is just Pydantic validation constraint(... means it's required, no default; ge=0.0, le=1.0 enforces it's a valid probability-like score between 0 and 1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    contradictions: list[Contradiction] = Field(default_factory=list)
