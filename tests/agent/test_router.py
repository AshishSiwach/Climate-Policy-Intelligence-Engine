"""
Tests for the complexity router — Phase 1 Foundations.

Unit tests mock the OpenAI client so no real API calls are made.
The integration test (pytest -m integration) calls the live LLM and measures
true classifier accuracy against the hand-labelled eval set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent.router import _AGENT_TYPES, complexity_router
from src.agent.router import Path as RoutePath

LABELS_PATH = Path("data/eval/router_labels.json")


def _mock_openai_response(task_type: str):
    """Build a mock OpenAI response object returning the given task_type."""
    mock_msg = MagicMock()
    mock_msg.content = json.dumps({"task_type": task_type})
    mock_choice = MagicMock()
    mock_choice.message = mock_msg
    mock_resp = MagicMock()
    mock_resp.choices = [mock_choice]
    return mock_resp


def _make_mock_client(task_type: str):
    """Return a mock OpenAI() instance that returns task_type on create()."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_openai_response(task_type)
    return mock_client


@pytest.fixture
def router_labels():
    """Load the hand-labelled routing eval set."""
    if not LABELS_PATH.exists():
        pytest.skip(f"router_labels.json not found at {LABELS_PATH}")
    with open(LABELS_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_router_labels_loaded(router_labels):
    """Sanity check: labels file has ≥ 30 entries with required keys."""
    assert len(router_labels) >= 30
    for item in router_labels:
        assert "query" in item
        assert "expected" in item


def test_label_to_path_contract(router_labels):
    """Every task_type in the eval set maps to the correct Path enum value.

    This is a contract test for the routing table (_AGENT_TYPES), not a
    classifier accuracy test.  The LLM is mocked to return the expected
    task_type so the test is deterministic and requires no API key.  It
    should always pass 100% — a failure means _AGENT_TYPES is misconfigured.
    """
    for item in router_labels:
        expected_task = item["expected"]
        expected_path = RoutePath.AGENT if expected_task in _AGENT_TYPES else RoutePath.FAST

        mock_client = _make_mock_client(expected_task)
        with patch("openai.OpenAI", return_value=mock_client):
            path, task_type = complexity_router(item["query"])

        assert path == expected_path, (
            f"task_type={expected_task!r} should map to {expected_path!r}, got {path!r}"
        )
        assert task_type == expected_task


@pytest.mark.integration
def test_router_accuracy_live(router_labels):
    """Measure true LLM classifier accuracy against the hand-labelled eval set.

    Calls the real OpenAI API — requires OPENAI_API_KEY.
    Run with:  pytest -m integration

    Fails with a confusion matrix if accuracy < 85%, highlighting cross_doc
    false negatives (queries that should route to AGENT but reach FAST).
    """
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set — skipping live router eval")

    results = []
    for item in router_labels:
        expected_task = item["expected"]
        expected_path = RoutePath.AGENT if expected_task in _AGENT_TYPES else RoutePath.FAST
        actual_path, actual_task = complexity_router(item["query"])
        results.append({
            "query": item["query"],
            "expected_task": expected_task,
            "expected_path": expected_path,
            "actual_task": actual_task,
            "actual_path": actual_path,
            "correct": actual_path == expected_path,
        })

    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total

    if accuracy < 0.85:
        lines = [f"\nRouter accuracy {accuracy:.1%} < 85% ({correct}/{total})"]

        cross_doc_fn = [
            r for r in results
            if r["expected_task"] == "cross_doc" and r["actual_path"] == RoutePath.FAST
        ]
        if cross_doc_fn:
            lines.append(f"\ncross_doc FALSE NEGATIVES — routed to FAST (n={len(cross_doc_fn)}):")
            for r in cross_doc_fn:
                lines.append(f"  [classified as {r['actual_task']!r}] {r['query'][:120]}")

        other_wrong = [
            r for r in results
            if not r["correct"] and r["expected_task"] != "cross_doc"
        ]
        if other_wrong:
            lines.append(f"\nOther misclassifications (n={len(other_wrong)}):")
            for r in other_wrong:
                lines.append(
                    f"  [expected={r['expected_task']!r} actual={r['actual_task']!r}]"
                    f" {r['query'][:120]}"
                )

        pytest.fail("\n".join(lines))


def test_router_agent_types_go_to_agent():
    """cross_doc routes to AGENT (LangGraph workflow)."""
    mock_client = _make_mock_client("cross_doc")
    with patch("openai.OpenAI", return_value=mock_client):
        path, returned_type = complexity_router("any query")
    assert path == RoutePath.AGENT
    assert returned_type == "cross_doc"


def test_router_fast_types_go_to_fast():
    """factual, numeric, summary, unsupported all route to FAST."""
    for task_type in ("factual", "numeric", "summary", "unsupported"):
        mock_client = _make_mock_client(task_type)
        with patch("openai.OpenAI", return_value=mock_client):
            path, returned_type = complexity_router("any query")
        assert path == RoutePath.FAST, f"{task_type} should route to FAST"
        assert returned_type == task_type


def test_router_classifies_regardless_of_agent_route_enabled_env(monkeypatch):
    """complexity_router is a pure classifier — AGENT_ROUTE_ENABLED has no effect on it.
    The caller (main.py) decides whether to act on the returned path."""
    monkeypatch.setenv("AGENT_ROUTE_ENABLED", "false")
    mock_client = _make_mock_client("cross_doc")
    with patch("openai.OpenAI", return_value=mock_client):
        path, task_type = complexity_router("any query")
    assert path == RoutePath.AGENT
    assert task_type == "cross_doc"


def test_router_fail_open_on_exception():
    """Any exception returns (FAST, 'factual') — fail-open."""
    with patch("openai.OpenAI", side_effect=RuntimeError("network down")):
        path, task_type = complexity_router("any query")
    assert path == RoutePath.FAST
    assert task_type == "factual"


def test_router_unknown_task_type_defaults_to_factual():
    """Unknown task_type from LLM is normalised to factual → FAST."""
    mock_client = _make_mock_client("hallucinated_type")
    with patch("openai.OpenAI", return_value=mock_client):
        path, task_type = complexity_router("any query")
    assert path == RoutePath.FAST
    assert task_type == "factual"
