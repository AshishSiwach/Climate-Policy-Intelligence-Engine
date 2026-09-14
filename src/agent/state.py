"""
AgentState — the shared state dict that flows through the LangGraph workflow.

All node functions consume and return plain Python dicts matching this shape.
The TypedDict is used for type-checking only; nodes never import TypedDict at
runtime — they just receive and return dict.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from src.evidence.claims import Claim, Coverage, SubQuestion


class AgentState(TypedDict):
    # Immutable input
    request_id: str
    query: str
    task_type: Literal["cross_doc", "contradiction", "weak_evidence"]

    # Populated during workflow
    sub_questions: list[SubQuestion]
    retrievals: dict[str, list[dict]]  # sub_question_id → chunks
    coverage: dict[str, Coverage]  # sub_question_id → coverage decision

    # Phase 2+ (leave as empty lists/None for Phase 1)
    claims: list[Claim]
    verified_claims: list[Claim]

    # Budget tracking (checked before each node)
    steps_used: int
    cost_used_usd: float
    time_used_s: float
    retries_used: dict[str, int]  # sub_question_id → retry count

    # Terminal state
    result: Optional[dict]  # AnalystBrief dict; None in Phase 1
    termination_reason: Optional[str]  # complete | max_steps | max_cost | max_time | fallback_to_fast

    # Runtime injection — not serialised, not part of eval output
    _retriever: Optional[object]  # HybridRetriever instance injected by caller
