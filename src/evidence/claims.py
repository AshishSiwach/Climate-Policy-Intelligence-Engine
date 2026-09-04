"""
Pydantic models for agent evidence pipeline — Phase 1 Foundations.

SubQuestion  — a decomposed sub-query produced by the Planner node.
Coverage     — grader decision for a sub-question.
EvidenceRef  — a single chunk reference used as evidence.
Claim        — a claim extracted from evidence (Phase 2; placeholder for Phase 1).
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class SubQuestion(BaseModel):
    id: str  # e.g. "sq_0", "sq_1"
    question: str
    required_source: Optional[str] = None  # institution filter e.g. "BoE", "Ofgem"
    task_type: str = "factual"


class Coverage(BaseModel):
    sub_question_id: str
    status: str  # "covered" | "partial" | "not_covered"
    gap_reason: Optional[str] = None


class EvidenceRef(BaseModel):
    chunk_id: str
    doc_id: str
    page: int
    passage: str


class Claim(BaseModel):
    id: str
    text: str
    evidence_ids: list[str]  # chunk_ids
    source_doc_id: str
