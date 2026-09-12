"""
Tests for the complexity router — Phase 1 Foundations.

Mocks the OpenAI client so no real API calls are made.
Loads router_labels.json and asserts ≥ 85% accuracy on the 30-query set.
"""

from __future__ import annotations

import json
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


def test_router_accuracy_mocked(router_labels):
    """Router achieves ≥ 85% accuracy when the LLM is mocked to return the expected label."""
    correct = 0
    total = len(router_labels)

    for item in router_labels:
        expected_task = item["expected"]
        expected_path = RoutePath.AGENT if expected_task in _AGENT_TYPES else RoutePath.FAST

        mock_client = _make_mock_client(expected_task)

        with patch("openai.OpenAI", return_value=mock_client):
            path, task_type = complexity_router(item["query"])

        if path == expected_path and task_type == expected_task:
            correct += 1

    accuracy = correct / total
    assert accuracy >= 0.85, f"Router accuracy {accuracy:.1%} < 85% ({correct}/{total})"


def test_router_agent_types_go_to_agent():
    """cross_doc and summary route to AGENT (contradiction is Phase 5b, not yet built)."""
    for task_type in ("cross_doc", "summary"):
        mock_client = _make_mock_client(task_type)
        with patch("openai.OpenAI", return_value=mock_client):
            path, returned_type = complexity_router("any query")
        assert path == RoutePath.AGENT, f"{task_type} should route to AGENT"
        assert returned_type == task_type


def test_router_fast_types_go_to_fast():
    """factual, numeric, unsupported all route to FAST."""
    for task_type in ("factual", "numeric", "unsupported"):
        mock_client = _make_mock_client(task_type)
        with patch("openai.OpenAI", return_value=mock_client):
            path, returned_type = complexity_router("any query")
        assert path == RoutePath.FAST, f"{task_type} should route to FAST"
        assert returned_type == task_type


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
