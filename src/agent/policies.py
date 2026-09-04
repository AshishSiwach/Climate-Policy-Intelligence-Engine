"""
Budget policies for the agent workflow — Phase 1 Foundations.

These constants are checked by check_budget() in workflow.py before each node.
Adjust values here; never hard-code them in node functions.
"""

MAX_STEPS: int = 8
MAX_COST_USD: float = 0.05
MAX_TIME_S: float = 60.0
RETRY_LIMIT: int = 2
