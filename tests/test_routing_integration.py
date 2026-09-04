"""
Integration tests for the feature-flag routing logic in main.run_query.

All external calls (LLM router, agent_graph, Postgres) are mocked per-test
via monkeypatch so no real API calls are made.  The tests verify that the
correct path is taken — fast vs agent — and that shadow threads are spawned
when expected.

Tested scenarios:
1. AGENT_ROUTE_ENABLED=false + cross_doc query → fast path, shadow spawned
2. AGENT_ROUTE_ENABLED=true  + cross_doc query → agent path used
3. AGENT_ROUTE_ENABLED=canary + cross_doc query → agent path for ~10% (mocked random)
4. Fast-eligible query (factual) + any flag → fast path always
5. Agent path raises exception → falls back to fast path
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from main import _should_use_agent, run_query
from monitoring import QueryLogger
from src.agent.router import Path as RoutePath

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_synth_mock(answer: str = "fast answer") -> MagicMock:
    """Synthesiser mock that returns a well-formed result dict."""
    synth = MagicMock()
    synth.model = "gpt-5.4-mini"
    brief = MagicMock()
    brief.answer = answer
    brief.citations = []
    brief.contradictions = []
    brief.model_dump = lambda: {
        "answer": answer,
        "citations": [],
        "contradictions": [],
        "coverage_gaps": [],
        "truncated": False,
        "termination_reason": None,
    }
    synth.synthesise = MagicMock(
        return_value={
            "brief": brief,
            "latency_ms": 100.0,
            "prompt_tokens": 500,
            "completion_tokens": 50,
            "cost_usd": 0.003,
        }
    )
    return synth


def _make_hybrid_mock() -> MagicMock:
    """Retriever mock that returns an empty chunk list."""
    hybrid = MagicMock()
    hybrid.retrieve = MagicMock(return_value=[])
    return hybrid


def _make_agent_state(answer: str = "agent answer") -> dict:
    """Simulate the final state returned by agent_graph.invoke."""
    return {
        "result": {
            "answer": answer,
            "citations": [],
            "contradictions": [],
            "coverage_gaps": [],
            "truncated": False,
            "termination_reason": "complete",
        },
        "cost_used_usd": 0.008,
        "termination_reason": "complete",
        "steps_used": 4,
    }


def _mock_settings(route_enabled: str, canary_pct: float = 0.10) -> MagicMock:
    settings = MagicMock()
    settings.agent.route_enabled = route_enabled
    settings.agent.canary_pct = canary_pct
    return settings


# ---------------------------------------------------------------------------
# Autouse: no real Postgres writes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_pg(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("main.db_insert_query_record", lambda record: None)


# ---------------------------------------------------------------------------
# _should_use_agent unit tests
# ---------------------------------------------------------------------------


def test_should_use_agent_false_always_returns_false() -> None:
    assert _should_use_agent("false", 0.10) is False


def test_should_use_agent_true_always_returns_true() -> None:
    assert _should_use_agent("true", 0.10) is True


def test_should_use_agent_canary_uses_random(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("main.random.random", lambda: 0.05)  # 0.05 < 0.10 → True
    assert _should_use_agent("canary", 0.10) is True

    monkeypatch.setattr("main.random.random", lambda: 0.50)  # 0.50 >= 0.10 → False
    assert _should_use_agent("canary", 0.10) is False


def test_should_use_agent_unknown_mode_returns_false() -> None:
    assert _should_use_agent("unknown_value", 0.10) is False


# ---------------------------------------------------------------------------
# Scenario 1: false flag + cross_doc → fast path + shadow spawned
# ---------------------------------------------------------------------------


def test_flag_false_cross_doc_uses_fast_path_and_spawns_shadow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """AGENT_ROUTE_ENABLED=false + cross_doc query → fast path returned, shadow spawned."""
    tmp_log = tmp_path / "queries.jsonl"

    # Router returns AGENT/cross_doc
    monkeypatch.setattr(
        "main.complexity_router",
        lambda q: (RoutePath.AGENT, "cross_doc"),
    )
    monkeypatch.setattr("main.get_settings", lambda: _mock_settings("false"))

    # Track shadow calls
    shadow_calls: list[tuple] = []
    monkeypatch.setattr(
        "main._run_agent_shadow",
        lambda query, task_type, hybrid, query_id: shadow_calls.append((query, task_type)),
    )

    # Track agent path calls — should NOT be called
    monkeypatch.setattr(
        "main._run_agent_path",
        MagicMock(side_effect=AssertionError("_run_agent_path must not be called")),
    )

    synth = _make_synth_mock()
    hybrid = _make_hybrid_mock()
    qlogger = QueryLogger(log_path=tmp_log)

    result = run_query("cross-doc query text", hybrid, synth, qlogger, log_path=tmp_log)

    # Fast path answer returned
    assert result["source"] == "fast"
    synth.synthesise.assert_called_once()

    # Shadow was spawned
    assert len(shadow_calls) == 1, "Shadow should have been spawned exactly once"
    assert shadow_calls[0][1] == "cross_doc"


# ---------------------------------------------------------------------------
# Scenario 2: true flag + cross_doc → agent path used
# ---------------------------------------------------------------------------


def test_flag_true_cross_doc_uses_agent_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """AGENT_ROUTE_ENABLED=true + cross_doc query → agent path is used."""
    tmp_log = tmp_path / "queries.jsonl"

    monkeypatch.setattr(
        "main.complexity_router",
        lambda q: (RoutePath.AGENT, "cross_doc"),
    )
    monkeypatch.setattr("main.get_settings", lambda: _mock_settings("true"))

    # Agent path returns a valid result
    agent_result = {
        "answer": "agent answer",
        "citations": [],
        "contradictions": [],
        "coverage_gaps": [],
        "truncated": False,
        "termination_reason": "complete",
        "query_id": "test-id",
        "source": "agent",
    }
    monkeypatch.setattr("main._run_agent_path", lambda *a, **kw: agent_result)

    # Shadow should NOT be spawned when agent path is active
    shadow_calls: list = []
    monkeypatch.setattr(
        "main._run_agent_shadow",
        lambda *a, **kw: shadow_calls.append(True),
    )

    synth = _make_synth_mock()
    hybrid = _make_hybrid_mock()
    qlogger = QueryLogger(log_path=tmp_log)

    result = run_query("cross-doc query text", hybrid, synth, qlogger, log_path=tmp_log)

    assert result["source"] == "agent"
    assert result["answer"] == "agent answer"
    synth.synthesise.assert_not_called()
    assert len(shadow_calls) == 0


# ---------------------------------------------------------------------------
# Scenario 3: canary flag — routes ~10% to agent (mocked random)
# ---------------------------------------------------------------------------


def test_flag_canary_routes_to_agent_when_random_below_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canary mode: random < canary_pct → agent path."""
    tmp_log = tmp_path / "queries.jsonl"

    monkeypatch.setattr(
        "main.complexity_router",
        lambda q: (RoutePath.AGENT, "cross_doc"),
    )
    monkeypatch.setattr("main.get_settings", lambda: _mock_settings("canary", 0.10))
    monkeypatch.setattr("main.random.random", lambda: 0.05)  # 0.05 < 0.10 → agent

    agent_result = {
        "answer": "agent answer",
        "citations": [],
        "contradictions": [],
        "coverage_gaps": [],
        "truncated": False,
        "termination_reason": "complete",
        "query_id": "test-id",
        "source": "agent",
    }
    monkeypatch.setattr("main._run_agent_path", lambda *a, **kw: agent_result)
    monkeypatch.setattr("main._run_agent_shadow", lambda *a, **kw: None)

    synth = _make_synth_mock()
    hybrid = _make_hybrid_mock()
    qlogger = QueryLogger(log_path=tmp_log)

    result = run_query("cross-doc query text", hybrid, synth, qlogger, log_path=tmp_log)

    assert result["source"] == "agent"


def test_flag_canary_uses_fast_when_random_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Canary mode: random >= canary_pct → fast path + shadow."""
    tmp_log = tmp_path / "queries.jsonl"

    monkeypatch.setattr(
        "main.complexity_router",
        lambda q: (RoutePath.AGENT, "cross_doc"),
    )
    monkeypatch.setattr("main.get_settings", lambda: _mock_settings("canary", 0.10))
    monkeypatch.setattr("main.random.random", lambda: 0.50)  # 0.50 >= 0.10 → fast

    shadow_calls: list = []
    monkeypatch.setattr("main._run_agent_shadow", lambda *a, **kw: shadow_calls.append(True))
    monkeypatch.setattr(
        "main._run_agent_path",
        MagicMock(side_effect=AssertionError("_run_agent_path must not be called")),
    )

    synth = _make_synth_mock()
    hybrid = _make_hybrid_mock()
    qlogger = QueryLogger(log_path=tmp_log)

    result = run_query("cross-doc query text", hybrid, synth, qlogger, log_path=tmp_log)

    assert result["source"] == "fast"
    assert len(shadow_calls) == 1


# ---------------------------------------------------------------------------
# Scenario 4: fast-eligible query always uses fast path
# ---------------------------------------------------------------------------


def test_factual_query_always_uses_fast_path_regardless_of_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A factual (FAST-path) query never goes to agent, regardless of flag."""
    tmp_log = tmp_path / "queries.jsonl"

    # Router returns FAST/factual
    monkeypatch.setattr(
        "main.complexity_router",
        lambda q: (RoutePath.FAST, "factual"),
    )
    monkeypatch.setattr("main.get_settings", lambda: _mock_settings("true"))  # flag is true

    shadow_calls: list = []
    monkeypatch.setattr("main._run_agent_shadow", lambda *a, **kw: shadow_calls.append(True))
    monkeypatch.setattr(
        "main._run_agent_path",
        MagicMock(side_effect=AssertionError("_run_agent_path must not be called")),
    )

    synth = _make_synth_mock()
    hybrid = _make_hybrid_mock()
    qlogger = QueryLogger(log_path=tmp_log)

    result = run_query("What is Ofgem?", hybrid, synth, qlogger, log_path=tmp_log)

    assert result["source"] == "fast"
    synth.synthesise.assert_called_once()
    assert len(shadow_calls) == 0, "Shadow must not be spawned for fast-eligible queries"


# ---------------------------------------------------------------------------
# Scenario 5: agent path exception → fallback to fast path
# ---------------------------------------------------------------------------


def test_agent_path_exception_falls_back_to_fast_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """When _run_agent_path raises, the fast path must be used as fallback."""
    tmp_log = tmp_path / "queries.jsonl"

    monkeypatch.setattr(
        "main.complexity_router",
        lambda q: (RoutePath.AGENT, "cross_doc"),
    )
    monkeypatch.setattr("main.get_settings", lambda: _mock_settings("true"))

    # Agent path blows up
    def _failing_agent_path(*args, **kwargs):
        raise RuntimeError("agent graph failure")

    monkeypatch.setattr("main._run_agent_path", _failing_agent_path)
    monkeypatch.setattr("main._run_agent_shadow", lambda *a, **kw: None)

    synth = _make_synth_mock()
    hybrid = _make_hybrid_mock()
    qlogger = QueryLogger(log_path=tmp_log)

    result = run_query("cross-doc query text", hybrid, synth, qlogger, log_path=tmp_log)

    # Must fall back to fast path — no exception propagated
    assert result["source"] == "fast"
    synth.synthesise.assert_called_once()
