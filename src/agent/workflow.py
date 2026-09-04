"""
LangGraph workflow — Phase 2 Cross-Document Route.

THIS IS THE ONLY FILE IN src/ THAT IMPORTS LANGGRAPH.

Phase 2 graph topology:
    planner → [budget?] → retriever → [budget?] → grader → [check_coverage]
                               ↓                                    |
                        handle_termination          ┌───────────────┼──────────────┐
                                                    ↓               ↓              ↓
                                               retry_retriever   claim_builder  terminate
                                                    ↓               ↓
                                            [budget?] → grader   [budget?] → verifier
                                            (loop while retries        ↓
                                             available)          [budget?] → synthesiser → END
                                                                           ↓
                                                                    handle_termination → END

Budget checks happen between nodes via conditional edges.
The retry loop refines queries for sub-questions with coverage gaps.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.agent.nodes.claim_builder import run_claim_builder
from src.agent.nodes.grader import run_grader
from src.agent.nodes.planner import run_planner
from src.agent.nodes.retriever import run_retriever
from src.agent.nodes.synthesiser import run_synthesiser
from src.agent.nodes.verifier import run_verifier
from src.agent.policies import MAX_COST_USD, MAX_STEPS, MAX_TIME_S, RETRY_LIMIT
from src.agent.state import AgentState
from src.evidence.claims import SubQuestion

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------


def check_budget(state: AgentState) -> str:
    """Returns 'continue' or 'terminate'."""
    if state.get("steps_used", 0) >= MAX_STEPS:
        logger.info("Budget: MAX_STEPS reached (%d)", MAX_STEPS)
        return "terminate"
    if state.get("cost_used_usd", 0.0) >= MAX_COST_USD:
        logger.info("Budget: MAX_COST_USD reached ($%.4f)", MAX_COST_USD)
        return "terminate"
    if state.get("time_used_s", 0.0) >= MAX_TIME_S:
        logger.info("Budget: MAX_TIME_S reached (%.1fs)", MAX_TIME_S)
        return "terminate"
    if state.get("termination_reason") == "fallback_to_fast":
        logger.info("Budget: fallback_to_fast triggered")
        return "terminate"
    return "continue"


# ---------------------------------------------------------------------------
# Coverage router (after grader)
# ---------------------------------------------------------------------------


def check_coverage(state: AgentState) -> str:
    """After grader: decide whether to retry, proceed, or terminate.

    Returns: "retry" | "proceed" | "terminate"
    """
    coverage = state.get("coverage", {})
    retries = state.get("retries_used", {})

    # Budget check takes priority
    if check_budget(state) == "terminate":
        return "terminate"

    # Find sub-questions with coverage gaps
    gaps = [sq_id for sq_id, cov in coverage.items() if cov.status in ("partial", "not_covered")]

    if not gaps:
        return "proceed"  # all covered → claim builder

    # Gaps exist — retry if any sub-question has retries remaining
    can_retry = any(retries.get(sq_id, 0) < RETRY_LIMIT for sq_id in gaps)
    if can_retry:
        return "retry"

    return "proceed"  # gaps but retries exhausted → proceed with partial coverage


# ---------------------------------------------------------------------------
# Retry retriever node
# ---------------------------------------------------------------------------


def run_retry_retriever(state: AgentState) -> dict:
    """Re-run retrieval only for sub-questions with coverage gaps.

    Refines queries by appending the gap_reason from the Coverage object.
    Increments retries_used[sq_id] for each retried sub-question.
    """
    coverage = state.get("coverage", {})
    retries_used = state.get("retries_used", {})
    sub_questions = state.get("sub_questions", [])
    retrievals = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)
    retriever = state.get("_retriever")

    # Determine which sub-questions to retry
    gap_sq_ids = {
        sq_id
        for sq_id, cov in coverage.items()
        if cov.status in ("partial", "not_covered") and retries_used.get(sq_id, 0) < RETRY_LIMIT
    }

    new_retrievals = dict(retrievals)
    new_retries = dict(retries_used)

    for sq in sub_questions:
        sq_id = sq.id if isinstance(sq, SubQuestion) else sq.get("id", "")
        if sq_id not in gap_sq_ids:
            continue

        question = sq.question if isinstance(sq, SubQuestion) else sq.get("question", "")
        gap_reason = coverage[sq_id].gap_reason or ""
        refined_query = f"{question} {gap_reason}".strip()

        if retriever is None:
            logger.warning("run_retry_retriever: no retriever in state for %s", sq_id)
            new_retrievals[sq_id] = []
        else:
            try:
                chunks = retriever.retrieve(refined_query, top_k=5)
                new_retrievals[sq_id] = chunks
                logger.debug("run_retry_retriever: %s → %d chunks (refined)", sq_id, len(chunks))
            except Exception as exc:
                logger.warning("run_retry_retriever: failed for %s: %s", sq_id, exc)
                new_retrievals[sq_id] = []

        new_retries[sq_id] = new_retries.get(sq_id, 0) + 1

    return {
        "retrievals": new_retrievals,
        "retries_used": new_retries,
        "steps_used": steps_used + 1,
    }


# ---------------------------------------------------------------------------
# Termination handler
# ---------------------------------------------------------------------------


def handle_termination(state: AgentState) -> dict:
    """Sets a termination reason and marks any partial result as truncated."""
    existing_reason = state.get("termination_reason")

    # Determine breach reason if not already set
    if existing_reason and existing_reason not in ("complete",):
        reason = existing_reason
    elif state.get("steps_used", 0) >= MAX_STEPS:
        reason = "max_steps"
    elif state.get("cost_used_usd", 0.0) >= MAX_COST_USD:
        reason = "max_cost"
    elif state.get("time_used_s", 0.0) >= MAX_TIME_S:
        reason = "max_time"
    elif existing_reason == "fallback_to_fast":
        reason = "fallback_to_fast"
    else:
        reason = "max_steps"  # safe default

    update: dict = {"termination_reason": reason}

    # Mark any partial result as truncated
    existing_result = state.get("result")
    if existing_result:
        truncated_result = dict(existing_result)
        truncated_result["truncated"] = True
        truncated_result["termination_reason"] = reason
        update["result"] = truncated_result

    return update


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Build and compile the Phase 2 LangGraph workflow.

    Graph topology:
        planner → [budget?] → retriever → [budget?] → grader → [check_coverage]
                      ↓             ↓               retry ↓    proceed ↓   terminate ↓
               terminate      terminate       retry_retriever  claim_builder  terminate
                                                  ↓               ↓
                                           [budget?]→grader  [budget?]→verifier
                                                               [budget?]→synthesiser→END
                                All terminate branches → handle_termination → END
    """
    graph = StateGraph(AgentState)

    # Register all node functions
    graph.add_node("planner", run_planner)
    graph.add_node("retriever", run_retriever)
    graph.add_node("grader", run_grader)
    graph.add_node("retry_retriever", run_retry_retriever)
    graph.add_node("claim_builder", run_claim_builder)
    graph.add_node("verifier", run_verifier)
    graph.add_node("synthesiser", run_synthesiser)
    graph.add_node("handle_termination", handle_termination)

    # Entry point
    graph.set_entry_point("planner")

    # After planner: budget check → retriever or termination
    graph.add_conditional_edges(
        "planner",
        check_budget,
        {
            "continue": "retriever",
            "terminate": "handle_termination",
        },
    )

    # After retriever: budget check → grader or termination
    graph.add_conditional_edges(
        "retriever",
        check_budget,
        {
            "continue": "grader",
            "terminate": "handle_termination",
        },
    )

    # After grader: coverage check → retry | proceed | terminate
    graph.add_conditional_edges(
        "grader",
        check_coverage,
        {
            "retry": "retry_retriever",
            "proceed": "claim_builder",
            "terminate": "handle_termination",
        },
    )

    # After retry_retriever: budget check → grader (loop) or termination
    graph.add_conditional_edges(
        "retry_retriever",
        check_budget,
        {
            "continue": "grader",
            "terminate": "handle_termination",
        },
    )

    # After claim_builder: budget check → verifier or termination
    graph.add_conditional_edges(
        "claim_builder",
        check_budget,
        {
            "continue": "verifier",
            "terminate": "handle_termination",
        },
    )

    # After verifier: budget check → synthesiser or termination
    graph.add_conditional_edges(
        "verifier",
        check_budget,
        {
            "continue": "synthesiser",
            "terminate": "handle_termination",
        },
    )

    # After synthesiser → END (success path)
    graph.add_edge("synthesiser", END)

    # Termination always → END
    graph.add_edge("handle_termination", END)

    return graph.compile()


# Module-level compiled graph — import this in shadow_run.py and other callers
agent_graph = build_graph()
