"""
LangGraph workflow — Cross-Document Route.

THIS IS THE ONLY FILE IN src/ THAT IMPORTS LANGGRAPH.

Graph topology:
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
import time
from typing import Callable

from langgraph.graph import END, StateGraph

import src.agent.policies as _policies
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
# Tracing wrapper
# ---------------------------------------------------------------------------


def _traced(node_fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
    """Wrap a node function to emit a span to agent_traces after each call."""
    node_name = node_fn.__name__.removeprefix("run_")

    def wrapper(state: dict) -> dict:
        t0 = time.time()
        result = node_fn(state)
        latency_ms = int((time.time() - t0) * 1000)

        try:
            from src.observability.tracing import emit_span
            cost = result.get("cost_used_usd", 0.0) - state.get("cost_used_usd", 0.0)
            emit_span(
                trace_id=state.get("request_id", ""),
                step_no=state.get("steps_used", 0) + 1,
                node_name=node_name,
                input_summary={"query": state.get("query", ""), "steps_used": state.get("steps_used", 0)},
                output_summary={"keys": list(result.keys())},
                tool_called=None,
                latency_ms=latency_ms,
                cost_usd=max(cost, 0.0),
                termination_reason=result.get("termination_reason"),
            )
        except Exception:
            pass  # tracing is non-critical — never break the agent

        return result

    wrapper.__name__ = node_fn.__name__
    return wrapper


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
        logger.info("Budget: termination_reason=%s", state.get("termination_reason"))
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

    # Find sub-questions with coverage gaps — partial has evidence, only retry not_covered
    gaps = [sq_id for sq_id, cov in coverage.items() if cov.status == "not_covered"]

    if not gaps:
        logger.info("Coverage: no not_covered sub-questions; proceeding without retry")
        return "proceed"  # all covered → claim builder

    # Gaps exist — retry if any sub-question has retries remaining
    can_retry = any(retries.get(sq_id, 0) < RETRY_LIMIT for sq_id in gaps)
    if can_retry:
        logger.info("Coverage: retrying not_covered sub-questions=%s retries=%s", gaps, retries)
        return "retry"

    logger.info("Coverage: retries exhausted for sub-questions=%s; proceeding", gaps)
    return "proceed"  # gaps but retries exhausted → proceed with partial coverage


# ---------------------------------------------------------------------------
# Retry retriever node
# ---------------------------------------------------------------------------


def run_retry_retriever(state: AgentState) -> dict:
    """Re-run retrieval only for sub-questions with coverage gaps.

    Gives retrieval one contextual second chance using the original user query
    plus the focused factual sub-question. Grader diagnostics are deliberately
    excluded because score/threshold text pollutes the retrieval query.
    Increments retries_used[sq_id] for each retried sub-question.
    """
    coverage = state.get("coverage", {})
    retries_used = state.get("retries_used", {})
    sub_questions = state.get("sub_questions", [])
    retrievals = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)
    retriever = state.get("_retriever")
    parent_query = state.get("query", "")

    # Determine which sub-questions to retry
    gap_sq_ids = {
        sq_id
        for sq_id, cov in coverage.items()
        if cov.status == "not_covered" and retries_used.get(sq_id, 0) < RETRY_LIMIT
    }

    new_retrievals = dict(retrievals)
    new_retries = dict(retries_used)

    for sq in sub_questions:
        sq_id = sq.id if isinstance(sq, SubQuestion) else sq.get("id", "")
        if sq_id not in gap_sq_ids:
            continue

        question = sq.question if isinstance(sq, SubQuestion) else sq.get("question", "")
        refined_query = _build_retry_query(parent_query, question)

        if retriever is None:
            logger.warning("run_retry_retriever: no retriever in state for %s", sq_id)
            new_retrievals[sq_id] = []
        else:
            try:
                chunks = retriever.retrieve(refined_query, top_k=_policies.RETRIEVER_TOP_K)
                new_retrievals[sq_id] = chunks
                logger.info(
                    "run_retry_retriever: %s → %d chunks query=%r",
                    sq_id,
                    len(chunks),
                    refined_query,
                )
            except Exception as exc:
                logger.warning("run_retry_retriever: failed for %s: %s", sq_id, exc)
                new_retrievals[sq_id] = []

        new_retries[sq_id] = new_retries.get(sq_id, 0) + 1

    return {
        "retrievals": new_retrievals,
        "retries_used": new_retries,
        "steps_used": steps_used + 1,
    }


def _build_retry_query(parent_query: str, question: str) -> str:
    """Build a contextual retry query without grader diagnostic text."""
    parent = " ".join(parent_query.split())
    focus = " ".join(question.split())

    if not parent or parent.casefold() == focus.casefold():
        return focus
    if not focus:
        return parent
    return f"{parent} Focus specifically on: {focus}"


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
    """Build and compile the LangGraph workflow for cross-document queries."""
    graph = StateGraph(AgentState)

    # Register all node functions — wrapped with tracing
    graph.add_node("planner", _traced(run_planner))
    graph.add_node("retriever", _traced(run_retriever))
    graph.add_node("grader", _traced(run_grader))
    graph.add_node("retry_retriever", _traced(run_retry_retriever))
    graph.add_node("claim_builder", _traced(run_claim_builder))
    graph.add_node("verifier", _traced(run_verifier))
    graph.add_node("synthesiser", _traced(run_synthesiser))
    graph.add_node("handle_termination", _traced(handle_termination))

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
