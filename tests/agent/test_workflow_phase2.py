"""
Integration tests for the Phase 2 LangGraph workflow.

All node functions are mocked — no real LLM calls or retrieval.
Tests the graph topology: conditional edges, retry loop, budget termination.

Each test builds a fresh graph using build_graph() with mocked node functions
injected via unittest.mock.patch, ensuring the correct conditional edges fire.
"""

from __future__ import annotations

from unittest.mock import patch

from src.agent.policies import MAX_STEPS, RETRY_LIMIT
from src.agent.workflow import build_graph, check_coverage
from src.evidence.claims import Claim, Coverage, SubQuestion
from src.synthesis.output_schema import AnalystBrief

# ---------------------------------------------------------------------------
# Shared state builders
# ---------------------------------------------------------------------------


def _base_state(**overrides) -> dict:
    base = {
        "request_id": "wf-test-001",
        "query": "How do climate policies compare?",
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


def _make_sq(id_: str, question: str = "Question?") -> SubQuestion:
    return SubQuestion(id=id_, question=question)


def _make_coverage(sq_id: str, status: str) -> Coverage:
    return Coverage(sub_question_id=sq_id, status=status)


def _make_claim(id_: str, text: str, evidence_ids: list[str], doc_id: str = "boe") -> Claim:
    return Claim(id=id_, text=text, evidence_ids=evidence_ids, source_doc_id=doc_id)


def _brief_dict(**kwargs) -> dict:
    """Build a minimal AnalystBrief dict."""
    return AnalystBrief(
        answer=kwargs.get("answer", "Test answer."),
        citations=[],
        coverage_gaps=kwargs.get("coverage_gaps", []),
        truncated=kwargs.get("truncated", False),
        termination_reason=kwargs.get("termination_reason", "complete"),
    ).model_dump()


# ---------------------------------------------------------------------------
# Test 1: Full happy path — all covered, no retry needed
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_full_happy_path_reaches_synthesiser(self):
        """planner→retriever→grader(all covered)→claim_builder→verifier→synthesiser→END."""
        sqs = [_make_sq("sq_0")]
        claim = _make_claim("c_0", "BoE runs stress tests.", ["boe_0"])

        # Mock node return values
        def mock_planner(state):
            return {"sub_questions": sqs, "steps_used": state["steps_used"] + 1}

        def mock_retriever(state):
            return {
                "retrievals": {"sq_0": [{"chunk_id": "boe_0", "doc_id": "boe", "text": "BoE text.", "page_number": 1}]},
                "steps_used": state["steps_used"] + 1,
            }

        def mock_grader(state):
            return {
                "coverage": {"sq_0": _make_coverage("sq_0", "covered")},
                "steps_used": state["steps_used"] + 1,
            }

        def mock_claim_builder(state):
            return {
                "claims": [claim],
                "steps_used": state["steps_used"] + 1,
                "cost_used_usd": state["cost_used_usd"] + 0.001,
            }

        def mock_verifier(state):
            return {
                "verified_claims": [claim],
                "steps_used": state["steps_used"] + 1,
            }

        def mock_synthesiser(state):
            return {
                "result": _brief_dict(termination_reason="complete"),
                "steps_used": state["steps_used"] + 1,
                "cost_used_usd": state["cost_used_usd"] + 0.001,
                "termination_reason": "complete",
            }

        with (
            patch("src.agent.workflow.run_planner", mock_planner),
            patch("src.agent.workflow.run_retriever", mock_retriever),
            patch("src.agent.workflow.run_grader", mock_grader),
            patch("src.agent.workflow.run_claim_builder", mock_claim_builder),
            patch("src.agent.workflow.run_verifier", mock_verifier),
            patch("src.agent.workflow.run_synthesiser", mock_synthesiser),
        ):
            graph = build_graph()
            final_state = graph.invoke(_base_state())

        assert final_state["termination_reason"] == "complete"
        assert final_state["result"] is not None
        brief = AnalystBrief(**final_state["result"])
        assert brief.truncated is False

    def test_happy_path_steps_accumulated(self):
        """All 6 nodes run → steps_used = 6 (planner+retriever+grader+builder+verifier+synth)."""
        sqs = [_make_sq("sq_0")]
        claim = _make_claim("c_0", "BoE text.", ["boe_0"])

        def mock_planner(s):
            return {"sub_questions": sqs, "steps_used": s["steps_used"] + 1}

        def mock_retriever(s):
            return {"retrievals": {"sq_0": []}, "steps_used": s["steps_used"] + 1}

        def mock_grader(s):
            return {"coverage": {"sq_0": _make_coverage("sq_0", "covered")}, "steps_used": s["steps_used"] + 1}

        def mock_claim_builder(s):
            return {"claims": [claim], "steps_used": s["steps_used"] + 1, "cost_used_usd": s["cost_used_usd"]}

        def mock_verifier(s):
            return {"verified_claims": [claim], "steps_used": s["steps_used"] + 1}

        def mock_synthesiser(s):
            return {
                "result": _brief_dict(),
                "steps_used": s["steps_used"] + 1,
                "cost_used_usd": s["cost_used_usd"],
                "termination_reason": "complete",
            }

        with (
            patch("src.agent.workflow.run_planner", mock_planner),
            patch("src.agent.workflow.run_retriever", mock_retriever),
            patch("src.agent.workflow.run_grader", mock_grader),
            patch("src.agent.workflow.run_claim_builder", mock_claim_builder),
            patch("src.agent.workflow.run_verifier", mock_verifier),
            patch("src.agent.workflow.run_synthesiser", mock_synthesiser),
        ):
            graph = build_graph()
            final_state = graph.invoke(_base_state())

        assert final_state["steps_used"] == 6


# ---------------------------------------------------------------------------
# Test 2: One retry — grader finds gap, retry_retriever runs, grader all covered
# ---------------------------------------------------------------------------


class TestOneRetry:
    def test_one_retry_then_proceeds_to_claim_builder(self):
        """grader finds gap → retry_retriever → grader (all covered) → claim builder."""
        sqs = [_make_sq("sq_0")]
        claim = _make_claim("c_0", "BoE text.", ["boe_0"])
        call_count = {"grader": 0}

        def mock_planner(s):
            return {"sub_questions": sqs, "steps_used": s["steps_used"] + 1}

        def mock_retriever(s):
            return {"retrievals": {"sq_0": []}, "steps_used": s["steps_used"] + 1}

        def mock_grader(s):
            call_count["grader"] += 1
            # First call: partial; second call (after retry): covered
            if call_count["grader"] == 1:
                cov = Coverage(sub_question_id="sq_0", status="partial", gap_reason="Need more data")
            else:
                cov = _make_coverage("sq_0", "covered")
            return {"coverage": {"sq_0": cov}, "steps_used": s["steps_used"] + 1}

        def mock_retry_retriever(s):
            chunk = {"chunk_id": "boe_0", "doc_id": "boe", "text": "More BoE data.", "page_number": 1}
            return {
                "retrievals": {"sq_0": [chunk]},
                "retries_used": {"sq_0": 1},
                "steps_used": s["steps_used"] + 1,
            }

        def mock_claim_builder(s):
            return {"claims": [claim], "steps_used": s["steps_used"] + 1, "cost_used_usd": s["cost_used_usd"]}

        def mock_verifier(s):
            return {"verified_claims": [claim], "steps_used": s["steps_used"] + 1}

        def mock_synthesiser(s):
            return {
                "result": _brief_dict(),
                "steps_used": s["steps_used"] + 1,
                "cost_used_usd": s["cost_used_usd"],
                "termination_reason": "complete",
            }

        with (
            patch("src.agent.workflow.run_planner", mock_planner),
            patch("src.agent.workflow.run_retriever", mock_retriever),
            patch("src.agent.workflow.run_grader", mock_grader),
            patch("src.agent.workflow.run_retry_retriever", mock_retry_retriever),
            patch("src.agent.workflow.run_claim_builder", mock_claim_builder),
            patch("src.agent.workflow.run_verifier", mock_verifier),
            patch("src.agent.workflow.run_synthesiser", mock_synthesiser),
        ):
            graph = build_graph()
            final_state = graph.invoke(_base_state())

        # Grader was called twice (once initially, once after retry)
        assert call_count["grader"] == 2
        assert final_state["termination_reason"] == "complete"


# ---------------------------------------------------------------------------
# Test 3: Retries exhausted — proceeds with partial coverage
# ---------------------------------------------------------------------------


class TestRetriesExhausted:
    def test_retries_exhausted_proceeds_to_claim_builder(self):
        """Grader always finds gaps; after RETRY_LIMIT retries, proceeds with partial coverage.

        Note: claim_builder/verifier/synthesiser mocks do NOT increment steps_used so
        the total step count stays within MAX_STEPS=8 for RETRY_LIMIT=2 retries.
        Path: planner(1)+retriever(1)+grader*3(3)+retry*2(2)=7 steps, then proceed.
        """
        sqs = [_make_sq("sq_0")]
        claim = _make_claim("c_0", "BoE text.", ["boe_0"])

        def mock_planner(s):
            return {"sub_questions": sqs, "steps_used": s["steps_used"] + 1}

        def mock_retriever(s):
            return {"retrievals": {"sq_0": []}, "steps_used": s["steps_used"] + 1}

        def mock_grader(s):
            # Always partial — forces retries until exhausted
            cov = Coverage(sub_question_id="sq_0", status="partial", gap_reason="Still partial")
            return {"coverage": {"sq_0": cov}, "steps_used": s["steps_used"] + 1}

        retry_count = {"n": 0}

        def mock_retry_retriever(s):
            retry_count["n"] += 1
            new_retries = dict(s.get("retries_used", {}))
            new_retries["sq_0"] = retry_count["n"]
            return {
                "retrievals": {"sq_0": []},
                "retries_used": new_retries,
                "steps_used": s["steps_used"] + 1,
            }

        # Downstream mocks do NOT increment steps to stay within MAX_STEPS=8
        # (planner+retriever+grader*3+retry*2 = 7 steps already)
        def mock_claim_builder(s):
            return {"claims": [claim], "steps_used": s["steps_used"], "cost_used_usd": s["cost_used_usd"]}

        def mock_verifier(s):
            return {"verified_claims": [claim], "steps_used": s["steps_used"]}

        def mock_synthesiser(s):
            return {
                "result": _brief_dict(),
                "steps_used": s["steps_used"],
                "cost_used_usd": s["cost_used_usd"],
                "termination_reason": "complete",
            }

        with (
            patch("src.agent.workflow.run_planner", mock_planner),
            patch("src.agent.workflow.run_retriever", mock_retriever),
            patch("src.agent.workflow.run_grader", mock_grader),
            patch("src.agent.workflow.run_retry_retriever", mock_retry_retriever),
            patch("src.agent.workflow.run_claim_builder", mock_claim_builder),
            patch("src.agent.workflow.run_verifier", mock_verifier),
            patch("src.agent.workflow.run_synthesiser", mock_synthesiser),
        ):
            graph = build_graph()
            final_state = graph.invoke(_base_state())

        # Retried exactly RETRY_LIMIT times then proceeded with partial coverage
        assert retry_count["n"] == RETRY_LIMIT
        assert final_state["termination_reason"] == "complete"


# ---------------------------------------------------------------------------
# Test 4: Budget breach mid-flow → handle_termination → END, truncated=True
# ---------------------------------------------------------------------------


class TestBudgetBreach:
    def test_max_steps_breach_at_retriever_triggers_termination(self):
        """steps_used hits MAX_STEPS after retriever → handle_termination → END, truncated."""
        sqs = [_make_sq("sq_0")]

        def mock_planner(s):
            # Consume all remaining budget steps
            return {"sub_questions": sqs, "steps_used": MAX_STEPS}

        with (
            patch("src.agent.workflow.run_planner", mock_planner),
        ):
            graph = build_graph()
            final_state = graph.invoke(_base_state())

        # Budget exceeded after planner → handle_termination → END
        assert final_state["termination_reason"] in ("max_steps", "max_cost", "max_time", "fallback_to_fast")
        assert final_state["steps_used"] == MAX_STEPS

    def test_budget_breach_marks_existing_result_truncated(self):
        """If a partial result exists when budget is breached, truncated=True."""
        sqs = [_make_sq("sq_0")]
        claim = _make_claim("c_0", "BoE text.", ["boe_0"])
        # A partial brief that would be in state before truncation
        partial_brief = _brief_dict(truncated=False, termination_reason="complete")

        def mock_planner(s):
            return {"sub_questions": sqs, "steps_used": s["steps_used"] + 1}

        def mock_retriever(s):
            return {"retrievals": {"sq_0": []}, "steps_used": s["steps_used"] + 1}

        def mock_grader(s):
            return {"coverage": {"sq_0": _make_coverage("sq_0", "covered")}, "steps_used": s["steps_used"] + 1}

        def mock_claim_builder(s):
            # Breach the budget: set steps_used to MAX_STEPS
            return {
                "claims": [claim],
                "steps_used": MAX_STEPS,
                "cost_used_usd": s["cost_used_usd"],
                "result": partial_brief,  # set a partial result before breach
            }

        with (
            patch("src.agent.workflow.run_planner", mock_planner),
            patch("src.agent.workflow.run_retriever", mock_retriever),
            patch("src.agent.workflow.run_grader", mock_grader),
            patch("src.agent.workflow.run_claim_builder", mock_claim_builder),
        ):
            graph = build_graph()
            final_state = graph.invoke(_base_state())

        # Should have hit handle_termination (budget breached after claim_builder)
        assert final_state["termination_reason"] in ("max_steps", "max_cost", "max_time")
        if final_state.get("result"):
            assert final_state["result"]["truncated"] is True


# ---------------------------------------------------------------------------
# Unit tests for check_coverage routing function
# ---------------------------------------------------------------------------


class TestCheckCoverageRouting:
    def test_all_covered_returns_proceed(self):
        """All sub-questions covered → 'proceed'."""
        state = _base_state(
            coverage={"sq_0": _make_coverage("sq_0", "covered")},
            retries_used={},
        )
        assert check_coverage(state) == "proceed"

    def test_gap_with_retries_available_returns_retry(self):
        """Gap exists, retries_used < RETRY_LIMIT → 'retry'."""
        state = _base_state(
            coverage={"sq_0": Coverage(sub_question_id="sq_0", status="partial", gap_reason="X")},
            retries_used={"sq_0": 0},
        )
        assert check_coverage(state) == "retry"

    def test_gap_with_retries_exhausted_returns_proceed(self):
        """Gap exists, retries_used == RETRY_LIMIT → 'proceed'."""
        state = _base_state(
            coverage={"sq_0": Coverage(sub_question_id="sq_0", status="partial", gap_reason="X")},
            retries_used={"sq_0": RETRY_LIMIT},
        )
        assert check_coverage(state) == "proceed"

    def test_budget_breach_returns_terminate(self):
        """Budget breach takes priority → 'terminate'."""
        state = _base_state(
            coverage={"sq_0": _make_coverage("sq_0", "covered")},
            steps_used=MAX_STEPS,
        )
        assert check_coverage(state) == "terminate"

    def test_not_covered_with_retries_available_returns_retry(self):
        """not_covered status + retries available → 'retry'."""
        state = _base_state(
            coverage={"sq_0": Coverage(sub_question_id="sq_0", status="not_covered", gap_reason="Nothing found")},
            retries_used={},
        )
        assert check_coverage(state) == "retry"

    def test_empty_coverage_returns_proceed(self):
        """No coverage entries → 'proceed' (no gaps)."""
        state = _base_state(coverage={}, retries_used={})
        assert check_coverage(state) == "proceed"
