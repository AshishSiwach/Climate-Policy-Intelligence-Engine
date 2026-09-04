"""
Budget policies for the agent workflow — Phase 1 Foundations.

These constants are checked by check_budget() in workflow.py before each node.
Adjust values here; never hard-code them in node functions.
"""

MAX_STEPS: int = 14   # Phase 1 used 8; Phase 2 needs 10 minimum (planner+retriever+2×(retry+grader)+claim_builder+verifier+synthesiser); 14 gives buffer
MAX_COST_USD: float = 0.05
MAX_TIME_S: float = 60.0
RETRY_LIMIT: int = 2
RETRIEVER_TOP_K: int = 6  # chunks per sub-question; determined by k-sweep (scripts/sweep_retriever_k.py): k=6 is lowest k clearing both gates (correct≥3.50, complete≥3.25); k=10 regresses (noise outweighs signal)
