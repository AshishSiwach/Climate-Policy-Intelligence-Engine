"""
Retriever node — Phase 1 Foundations.

Calls HybridRetriever for each sub-question in state["sub_questions"].
No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.

The HybridRetriever instance must be injected via state["_retriever"] — the
workflow.py build_graph() function is responsible for binding it. This keeps
the node testable without the full ML stack.
"""

from __future__ import annotations

import logging

from src.evidence.claims import SubQuestion

logger = logging.getLogger(__name__)

_DEFAULT_TOP_K = 5


def run_retriever(state: dict) -> dict:
    """Retriever node: fetches chunks for each sub-question.

    Expects state["sub_questions"] to contain SubQuestion objects (or dicts).
    Uses the HybridRetriever bound in state["_retriever"] if present;
    falls back to an empty result list if the retriever is unavailable
    (so tests can run without the full ML stack).

    Returns partial dict with "retrievals" and incremented "steps_used".
    """
    sub_questions: list = state.get("sub_questions", [])
    steps_used = state.get("steps_used", 0)
    retriever = state.get("_retriever")  # injected by workflow or test

    retrievals: dict[str, list[dict]] = {}

    for sq in sub_questions:
        # Accept both SubQuestion objects and plain dicts
        if isinstance(sq, SubQuestion):
            sq_id = sq.id
            question = sq.question
            required_source = sq.required_source
        else:
            sq_id = sq.get("id", "sq_unknown")
            question = sq.get("question", "")
            required_source = sq.get("required_source")

        if retriever is None:
            logger.warning("run_retriever: no retriever in state — returning empty results for %s", sq_id)
            retrievals[sq_id] = []
            continue

        try:
            kwargs: dict = {"top_k": _DEFAULT_TOP_K}
            if required_source:
                kwargs["institutions"] = [required_source]

            chunks = retriever.retrieve(question, **kwargs)
            retrievals[sq_id] = chunks
            logger.debug("run_retriever: %s → %d chunks", sq_id, len(chunks))

        except Exception as exc:
            logger.warning("run_retriever: retrieval failed for %s: %s", sq_id, exc)
            retrievals[sq_id] = []

    return {"retrievals": retrievals, "steps_used": steps_used + 1}
