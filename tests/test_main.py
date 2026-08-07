"""
Unit tests for v1 guardrails in main.run_query — query length limit and
daily cost circuit breaker. Both must refuse the query WITHOUT calling
the LLM and still write a log record with a distinct failure_reason.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from main import (
    COST_LIMIT_MSG,
    DAILY_COST_LIMIT_USD,
    MAX_QUERY_CHARS,
    QUERY_TOO_LONG_MSG,
    _daily_cost_so_far,
    run_query,
)
from monitoring import QueryLogger


# ---------------------------------------------------------------------------
# _daily_cost_so_far — cost accumulator
# ---------------------------------------------------------------------------

def _write_log_entry(path: Path, *, ts: str, cost: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": ts, "cost_usd": cost}) + "\n")


def test_daily_cost_returns_zero_for_missing_log(tmp_path):
    assert _daily_cost_so_far(tmp_path / "nonexistent.jsonl") == 0.0


def test_daily_cost_sums_today_only(tmp_log_path):
    today = datetime.now(timezone.utc).isoformat()
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()

    _write_log_entry(tmp_log_path, ts=yesterday, cost=10.00)   # should NOT count
    _write_log_entry(tmp_log_path, ts=today, cost=0.50)
    _write_log_entry(tmp_log_path, ts=today, cost=0.25)

    assert _daily_cost_so_far(tmp_log_path) == pytest.approx(0.75)


def test_daily_cost_ignores_malformed_lines(tmp_log_path):
    """Corrupt JSON in the log must not crash the guardrail check."""
    today = datetime.now(timezone.utc).isoformat()
    tmp_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_log_path, "w", encoding="utf-8") as f:
        f.write("not-json\n")
        f.write(json.dumps({"timestamp": today, "cost_usd": 0.30}) + "\n")
        f.write("also not json\n")

    assert _daily_cost_so_far(tmp_log_path) == pytest.approx(0.30)


# ---------------------------------------------------------------------------
# Query length guardrail
# ---------------------------------------------------------------------------

def _mock_pipeline():
    """Return (hybrid, synth) mocks that must NOT be called if a guardrail fires."""
    hybrid = MagicMock()
    hybrid.retrieve = MagicMock(side_effect=AssertionError("hybrid must not be called"))
    synth = MagicMock()
    synth.model = "gpt-4o-mini"
    synth.synthesise = MagicMock(side_effect=AssertionError("synth must not be called"))
    return hybrid, synth


def test_query_length_guardrail_refuses_and_skips_pipeline(tmp_log_path):
    hybrid, synth = _mock_pipeline()
    qlogger = QueryLogger(log_path=tmp_log_path)

    long_query = "x " * (MAX_QUERY_CHARS)   # 2 * MAX chars, over the limit
    assert len(long_query) > MAX_QUERY_CHARS

    result = run_query(long_query, hybrid, synth, qlogger, log_path=tmp_log_path)

    assert result["answer"] == QUERY_TOO_LONG_MSG
    assert result["confidence"] == 0.0
    hybrid.retrieve.assert_not_called()
    synth.synthesise.assert_not_called()


def test_query_length_guardrail_writes_log_record(tmp_log_path):
    hybrid, synth = _mock_pipeline()
    qlogger = QueryLogger(log_path=tmp_log_path)

    long_query = "y " * MAX_QUERY_CHARS
    run_query(long_query, hybrid, synth, qlogger, log_path=tmp_log_path)

    lines = tmp_log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert "query_too_long" in rec["failure_reason"]


def test_query_length_within_limit_proceeds(sample_chunks, tmp_log_path):
    """A short query must NOT trigger the length guardrail."""
    hybrid = MagicMock()
    hybrid.retrieve = MagicMock(return_value=sample_chunks)

    synth = MagicMock()
    synth.model = "gpt-4o-mini"
    synth.synthesise = MagicMock(return_value={
        "brief": MagicMock(model_dump=lambda: {"answer": "ok", "confidence": 0.7,
                                               "citations": [], "contradictions": []}),
        "latency_ms": 100.0,
        "prompt_tokens": 500,
        "completion_tokens": 50,
        "cost_usd": 0.0003,
        "confidence_signals": {"score_signal": 0.6, "agreement_signal": 0.8,
                               "margin_signal": 0.5, "citation_signal": 0.3,
                               "llm_refusal": False},
    })

    qlogger = QueryLogger(log_path=tmp_log_path)
    result = run_query("What does Ofgem propose?", hybrid, synth, qlogger, log_path=tmp_log_path)

    hybrid.retrieve.assert_called_once()
    synth.synthesise.assert_called_once()
    assert "answer" in result


# ---------------------------------------------------------------------------
# Daily cost circuit breaker
# ---------------------------------------------------------------------------

def test_cost_circuit_breaker_refuses_when_budget_exhausted(tmp_log_path):
    """Log a spend >= DAILY_COST_LIMIT_USD; next query must be refused."""
    today = datetime.now(timezone.utc).isoformat()
    _write_log_entry(tmp_log_path, ts=today, cost=DAILY_COST_LIMIT_USD + 0.01)

    hybrid, synth = _mock_pipeline()
    qlogger = QueryLogger(log_path=tmp_log_path)

    result = run_query("short query", hybrid, synth, qlogger, log_path=tmp_log_path)

    assert result["answer"] == COST_LIMIT_MSG
    assert result["confidence"] == 0.0
    hybrid.retrieve.assert_not_called()
    synth.synthesise.assert_not_called()


def test_cost_circuit_breaker_writes_failure_reason(tmp_log_path):
    today = datetime.now(timezone.utc).isoformat()
    _write_log_entry(tmp_log_path, ts=today, cost=DAILY_COST_LIMIT_USD + 1.0)

    hybrid, synth = _mock_pipeline()
    qlogger = QueryLogger(log_path=tmp_log_path)
    run_query("short query", hybrid, synth, qlogger, log_path=tmp_log_path)

    lines = tmp_log_path.read_text(encoding="utf-8").splitlines()
    # First line = the seeded spend, second line = the guardrail refusal record
    assert len(lines) == 2
    refusal_rec = json.loads(lines[1])
    assert "daily_cost_limit" in refusal_rec["failure_reason"]


def test_cost_circuit_breaker_below_limit_allows_query(sample_chunks, tmp_log_path):
    """Under budget → pipeline runs normally."""
    today = datetime.now(timezone.utc).isoformat()
    _write_log_entry(tmp_log_path, ts=today, cost=DAILY_COST_LIMIT_USD * 0.5)

    hybrid = MagicMock()
    hybrid.retrieve = MagicMock(return_value=sample_chunks)
    synth = MagicMock()
    synth.model = "gpt-4o-mini"
    synth.synthesise = MagicMock(return_value={
        "brief": MagicMock(model_dump=lambda: {"answer": "ok", "confidence": 0.5,
                                               "citations": [], "contradictions": []}),
        "latency_ms": 50.0, "prompt_tokens": 100, "completion_tokens": 20,
        "cost_usd": 0.0001,
        "confidence_signals": {"score_signal": 0.5, "agreement_signal": 0.5,
                               "margin_signal": 0.5, "citation_signal": 0.0,
                               "llm_refusal": False},
    })

    qlogger = QueryLogger(log_path=tmp_log_path)
    run_query("Ofgem", hybrid, synth, qlogger, log_path=tmp_log_path)
    hybrid.retrieve.assert_called_once()
    synth.synthesise.assert_called_once()


# ---------------------------------------------------------------------------
# Constants — locked values
# ---------------------------------------------------------------------------

def test_guardrail_constants_match_locked_decisions():
    """CLAUDE.md Locked Decisions specify these values."""
    assert MAX_QUERY_CHARS == 500
    assert DAILY_COST_LIMIT_USD == 5.00
