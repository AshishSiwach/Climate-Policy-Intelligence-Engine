"""
Tests for budget termination checks — Phase 1 Foundations.

Verifies that check_budget() correctly identifies budget breaches
without making any real LLM calls.
"""

from __future__ import annotations

from src.agent.policies import MAX_COST_USD, MAX_STEPS, MAX_TIME_S
from src.agent.workflow import check_budget


def _state(**overrides) -> dict:
    """Build a minimal AgentState dict with sane defaults."""
    base = {
        "request_id": "test-req-001",
        "query": "test query",
        "task_type": "cross_doc",
        "sub_questions": [],
        "retrievals": {},
        "coverage": {},
        "claims": [],
        "verified_claims": [],
        "steps_used": 0,
        "cost_used_usd": 0.0,
        "time_used_s": 0.0,
        "retries_used": {},
        "result": None,
        "termination_reason": None,
    }
    base.update(overrides)
    return base


def test_max_steps_triggers_terminate():
    """steps_used at MAX_STEPS → terminate."""
    state = _state(steps_used=MAX_STEPS)
    assert check_budget(state) == "terminate"


def test_steps_below_max_is_continue():
    """steps_used below MAX_STEPS → continue (all else fine)."""
    state = _state(steps_used=MAX_STEPS - 1)
    assert check_budget(state) == "continue"


def test_max_cost_triggers_terminate():
    """cost_used_usd at MAX_COST_USD → terminate."""
    state = _state(cost_used_usd=MAX_COST_USD)
    assert check_budget(state) == "terminate"


def test_cost_above_max_triggers_terminate():
    """cost_used_usd above MAX_COST_USD → terminate."""
    state = _state(cost_used_usd=MAX_COST_USD + 0.01)
    assert check_budget(state) == "terminate"


def test_max_time_triggers_terminate():
    """time_used_s at MAX_TIME_S → terminate."""
    state = _state(time_used_s=MAX_TIME_S)
    assert check_budget(state) == "terminate"


def test_time_above_max_triggers_terminate():
    """time_used_s above MAX_TIME_S → terminate."""
    state = _state(time_used_s=MAX_TIME_S + 1.0)
    assert check_budget(state) == "terminate"


def test_fallback_to_fast_triggers_terminate():
    """termination_reason='fallback_to_fast' → terminate."""
    state = _state(termination_reason="fallback_to_fast")
    assert check_budget(state) == "terminate"


def test_normal_state_continues():
    """All budgets healthy, no termination_reason → continue."""
    state = _state(
        steps_used=2,
        cost_used_usd=0.01,
        time_used_s=10.0,
    )
    assert check_budget(state) == "continue"


def test_zero_values_continue():
    """Zero budget usage → continue."""
    state = _state()
    assert check_budget(state) == "continue"
