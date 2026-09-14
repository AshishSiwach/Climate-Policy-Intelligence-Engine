"""
Tests for run_query_with_progress() — the streaming generator entry point.

Verifies the event sequence (routing → node* → result) for each execution path:
  fast query, guardrail refusal, shadow mode, active agent, fallback_to_fast,
  agent exception fallback.

All LLM calls, Postgres writes, and agent graph invocations are mocked so the
suite is fast, deterministic, and offline.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import (
    COST_LIMIT_MSG,
    MAX_QUERY_CHARS,
    QUERY_TOO_LONG_MSG,
    run_query_with_progress,
)
from monitoring import QueryLogger


# ---------------------------------------------------------------------------
# Autouse fixtures — applied to every test in this file
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mock_infra(monkeypatch):
    """Block Postgres writes and provide a default settings stub."""
    monkeypatch.setattr("main.db_insert_query_record", lambda record: None)

    settings = MagicMock()
    settings.agent.route_enabled = "false"
    settings.agent.canary_pct = 0.0
    monkeypatch.setattr("main.get_settings", lambda: settings)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _events(gen) -> list[dict]:
    return list(gen)


def _routing_paths(events: list[dict]) -> list[str]:
    return [e["path"] for e in events if e["type"] == "routing"]


def _node_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e["type"] == "node"]


def _result(events: list[dict]) -> dict:
    return next(e for e in events if e["type"] == "result")


def _mock_fast_rv(monkeypatch, answer: str = "fast answer") -> dict:
    rv = {"answer": answer, "citations": [], "source": "fast", "query_id": "PLACEHOLDER"}
    monkeypatch.setattr("main._run_fast_path", lambda *a, **kw: rv)
    return rv


def _mock_guardrails_pass(monkeypatch) -> None:
    monkeypatch.setattr("main._check_guardrails", lambda *a, **kw: None)


def _mock_route(monkeypatch, path_name: str, task_type: str, use_agent: bool) -> None:
    from src.agent.router import Path as RoutePath

    path = RoutePath.AGENT if path_name == "agent" else RoutePath.FAST
    monkeypatch.setattr(
        "main._evaluate_route",
        lambda *a, **kw: (path, task_type, use_agent),
    )


def _mock_shadow_settings(monkeypatch, route_enabled: str = "false") -> None:
    """Override get_settings to set route_enabled for shadow-check coverage."""
    settings = MagicMock()
    settings.agent.route_enabled = route_enabled
    settings.agent.canary_pct = 0.0
    monkeypatch.setattr("main.get_settings", lambda: settings)


def _make_agent_brief_mock(answer: str = "agent answer") -> MagicMock:
    brief = MagicMock()
    brief.model_dump.return_value = {
        "answer": answer,
        "citations": [],
        "truncated": False,
        "termination_reason": None,
    }
    return brief


def _mock_agent_infra(monkeypatch, stream_chunks: list[dict], brief_answer: str = "agent answer"):
    """Patch the agent graph, brief converter, and logger for an agent-path test."""
    brief = _make_agent_brief_mock(brief_answer)
    monkeypatch.setattr("main._agent_state_to_brief", lambda state: brief)
    monkeypatch.setattr("main._log_agent_run", lambda *a, **kw: None)

    mock_graph = MagicMock()
    mock_graph.stream.return_value = stream_chunks
    monkeypatch.setattr("main.agent_graph", mock_graph)

    return brief


# ---------------------------------------------------------------------------
# 1. Guardrail refusal
# ---------------------------------------------------------------------------


def test_guardrail_length_yields_routing_fast_then_result(tmp_log_path):
    """A query over MAX_QUERY_CHARS → routing:fast + result with refusal text."""
    hybrid = MagicMock()
    synth = MagicMock()
    synth.model = "test-model"
    qlogger = QueryLogger(log_path=tmp_log_path)

    long_query = "x " * MAX_QUERY_CHARS  # well over the limit

    events = _events(
        run_query_with_progress(long_query, hybrid, synth, qlogger, log_path=tmp_log_path)
    )

    types = [e["type"] for e in events]
    assert types == ["routing", "result"], f"unexpected event sequence: {types}"
    assert events[0]["path"] == "fast"
    assert events[1]["brief"]["answer"] == QUERY_TOO_LONG_MSG


def test_guardrail_refusal_emits_no_node_events(tmp_log_path):
    hybrid = MagicMock()
    synth = MagicMock()
    synth.model = "test-model"
    qlogger = QueryLogger(log_path=tmp_log_path)

    long_query = "y " * MAX_QUERY_CHARS
    events = _events(
        run_query_with_progress(long_query, hybrid, synth, qlogger, log_path=tmp_log_path)
    )

    assert _node_events(events) == []


def test_guardrail_refusal_result_carries_query_id(tmp_log_path):
    hybrid = MagicMock()
    synth = MagicMock()
    synth.model = "test-model"
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("z " * MAX_QUERY_CHARS, hybrid, synth, qlogger, log_path=tmp_log_path)
    )

    assert "query_id" in _result(events)["brief"]


# ---------------------------------------------------------------------------
# 2. Fast query path (router returns FAST, no agent)
# ---------------------------------------------------------------------------


def test_fast_query_event_sequence(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "fast", "factual", False)
    _mock_fast_rv(monkeypatch)
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("What is net zero?", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    assert [e["type"] for e in events] == ["routing", "result"]
    assert events[0]["path"] == "fast"


def test_fast_query_result_is_fast_path_output(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "fast", "factual", False)
    fast_rv = _mock_fast_rv(monkeypatch, answer="corpus answer")
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    assert _result(events)["brief"] == fast_rv


# ---------------------------------------------------------------------------
# 3. Shadow mode (agent-eligible query, flag = false/canary)
# ---------------------------------------------------------------------------


def test_shadow_mode_yields_fast_path(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", False)
    _mock_shadow_settings(monkeypatch, route_enabled="false")
    fast_rv = _mock_fast_rv(monkeypatch)
    shadow_calls: list = []
    monkeypatch.setattr("main._run_agent_shadow", lambda *a, **kw: shadow_calls.append(a))
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("multi-doc q", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    assert _routing_paths(events) == ["fast"]
    assert _result(events)["brief"] == fast_rv
    assert len(shadow_calls) == 1, "shadow task must fire exactly once"


def test_shadow_canary_mode_triggers_shadow(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", False)
    _mock_shadow_settings(monkeypatch, route_enabled="canary")
    _mock_fast_rv(monkeypatch)
    shadow_calls: list = []
    monkeypatch.setattr("main._run_agent_shadow", lambda *a, **kw: shadow_calls.append(a))
    qlogger = QueryLogger(log_path=tmp_log_path)

    _events(run_query_with_progress("q", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path))

    assert len(shadow_calls) == 1


# ---------------------------------------------------------------------------
# 4. Active agent path (use_agent=True, graph completes normally)
# ---------------------------------------------------------------------------


def test_active_agent_starts_with_agent_routing(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    _mock_agent_infra(monkeypatch, stream_chunks=[])
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("complex query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    assert events[0] == {"type": "routing", "path": "agent"}


def test_active_agent_emits_node_events_per_stream_chunk(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    stream_chunks = [
        {"planner": {"sub_questions": ["sq1", "sq2"]}},
        {"retriever": {"retrievals": {"sq1": []}}},
        {"grader": {"coverage": {"sq1": "covered"}}},
    ]
    _mock_agent_infra(monkeypatch, stream_chunks=stream_chunks)
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    node_evs = _node_events(events)
    assert len(node_evs) == 3
    assert [e["name"] for e in node_evs] == ["planner", "retriever", "grader"]


def test_active_agent_result_has_agent_source(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    _mock_agent_infra(monkeypatch, stream_chunks=[], brief_answer="synthesised answer")
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    r = _result(events)
    assert r["brief"]["source"] == "agent"
    assert r["brief"]["answer"] == "synthesised answer"
    assert "query_id" in r["brief"]


def test_active_agent_event_sequence_is_routing_nodes_result(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    stream_chunks = [{"planner": {}}, {"retriever": {}}]
    _mock_agent_infra(monkeypatch, stream_chunks=stream_chunks)
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    types = [e["type"] for e in events]
    assert types[0] == "routing"
    assert types[1:-1] == ["node", "node"]
    assert types[-1] == "result"


# ---------------------------------------------------------------------------
# 5. Agent fallback_to_fast (planner sets termination_reason)
# ---------------------------------------------------------------------------


def test_agent_fallback_to_fast_reroutes_to_fast(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    fast_rv = _mock_fast_rv(monkeypatch)
    stream_chunks = [{"planner": {"termination_reason": "fallback_to_fast"}}]
    _mock_agent_infra(monkeypatch, stream_chunks=stream_chunks)
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    assert _routing_paths(events) == ["agent", "fast"]
    assert _result(events)["brief"] == fast_rv


def test_agent_fallback_result_is_not_agent_source(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    _mock_fast_rv(monkeypatch, answer="fast fallback answer")
    stream_chunks = [{"planner": {"termination_reason": "fallback_to_fast"}}]
    _mock_agent_infra(monkeypatch, stream_chunks=stream_chunks)
    qlogger = QueryLogger(log_path=tmp_log_path)

    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    assert _result(events)["brief"].get("source") != "agent"


# ---------------------------------------------------------------------------
# 6. Agent exception fallback
# ---------------------------------------------------------------------------


def test_agent_exception_falls_back_to_fast(monkeypatch, tmp_log_path):
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    fast_rv = _mock_fast_rv(monkeypatch)

    mock_graph = MagicMock()
    mock_graph.stream.side_effect = RuntimeError("graph node exploded")
    monkeypatch.setattr("main.agent_graph", mock_graph)

    qlogger = QueryLogger(log_path=tmp_log_path)
    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    # Must end on fast path
    assert _routing_paths(events)[-1] == "fast"
    assert _result(events)["brief"] == fast_rv


def test_agent_exception_still_emits_result(monkeypatch, tmp_log_path):
    """An agent crash must not leave the generator without a result event."""
    _mock_guardrails_pass(monkeypatch)
    _mock_route(monkeypatch, "agent", "cross_doc", True)
    _mock_fast_rv(monkeypatch)

    mock_graph = MagicMock()
    mock_graph.stream.side_effect = ValueError("unexpected node state")
    monkeypatch.setattr("main.agent_graph", mock_graph)

    qlogger = QueryLogger(log_path=tmp_log_path)
    events = _events(
        run_query_with_progress("query", MagicMock(), MagicMock(), qlogger, log_path=tmp_log_path)
    )

    result_events = [e for e in events if e["type"] == "result"]
    assert len(result_events) == 1, "exactly one result event must be emitted even on crash"
