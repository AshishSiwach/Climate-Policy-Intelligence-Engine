"""
Budget policies for the agent workflow — Phase 1 Foundations.

These constants are checked by check_budget() in workflow.py before each node.
Adjust values here; never hard-code them in node functions.
"""

MAX_STEPS: int = 14   # One-retry path needs 8 steps; 14 retains headroom for termination handling and future nodes
MAX_COST_USD: float = 0.05
MAX_TIME_S: float = 60.0
RETRY_LIMIT: int = 1  # one contextual retry; repeated identical retrieval did not improve genuine corpus gaps
# Determined by scripts/sweep_retriever_k.py: k=6 is the lowest value clearing
# correctness >= 3.50 and completeness >= 3.25; k=10 regressed due to noise.
RETRIEVER_TOP_K: int = 6
# For summary sub-questions the retriever fetches globally then filters to a single
# doc_id. With RETRIEVER_TOP_K=6 only 1-3 doc-specific chunks survive the filter.
# Fetching 25 globally before filtering keeps ≥8-10 on-target chunks per sub-question.
SUMMARY_RETRIEVER_TOP_K: int = 25
