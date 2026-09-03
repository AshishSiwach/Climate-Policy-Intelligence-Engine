"""
LangGraph workflow — Phase 1 Foundations.

THIS IS THE ONLY FILE IN src/ THAT IMPORTS LANGGRAPH.

Phase 1 graph: planner → retriever → grader → END
Budget checks happen between nodes via conditional edges.
"""

from __future__ import annotations

import logging

from langgraph.graph import END, StateGraph

from src.agent.nodes.grader import run_grader
from src.agent.nodes.planner import run_planner
from src.agent.nodes.retriever import run_retriever
from src.agent.policies import MAX_COST_USD, MAX_STEPS, MAX_TIME_S
from src.agent.state import AgentState

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
# Termination handler
# ---------------------------------------------------------------------------


def handle_termination(state: AgentState) -> AgentState:
    """Sets a default termination reason if none is already set."""
    reason = state.get("termination_reason") or "max_steps"
    return {**state, "termination_reason": reason}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def build_graph():
    """Build and compile the Phase 1 LangGraph workflow.

    Graph topology:
        planner → [budget?] → retriever → [budget?] → grader → END
                       ↓                      ↓
                handle_termination      handle_termination → END
    """
    graph = StateGraph(AgentState)

    # Register node functions
    graph.add_node("planner", run_planner)
    graph.add_node("retriever", run_retriever)
    graph.add_node("grader", run_grader)
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

    # After grader: Phase 1 stops here (no retry loop — that's Phase 2)
    graph.add_edge("grader", END)

    # Termination always → END
    graph.add_edge("handle_termination", END)

    return graph.compile()


# Module-level compiled graph — import this in shadow_run.py and other callers
agent_graph = build_graph()
