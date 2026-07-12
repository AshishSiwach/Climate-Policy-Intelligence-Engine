"""
Pydantic output schema for the analyst brief.

The LLM is asked to return JSON matching AnalystBrief (excluding the
`confidence` field — that's computed by the pipeline, not the model).
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    doc_id: str = Field(..., description="Source document identifier, e.g. OFGEM_SMART_SECURE_2024")
    passage: str = Field(..., description="Verbatim quote from the retrieved chunk that supports the claim")
    page: int = Field(..., ge=1, description="Page number in the source document")


class Contradiction(BaseModel):
    """Experimental in v1 — LLM self-report of conflicting claims across sources."""
    doc_a: str
    doc_b: str
    summary: str


class LLMResponse(BaseModel):
    """Raw JSON shape returned by the LLM. Confidence is added by the pipeline."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)


class AnalystBrief(BaseModel):
    """Final structured brief returned to the user."""
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # confidence is just Pydantic validation constraint(... means it's required, no default; ge=0.0, le=1.0 enforces it's a valid probability-like score between 0 and 1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    contradictions: list[Contradiction] = Field(default_factory=list)
