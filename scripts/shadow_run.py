"""
Shadow runner — Phase 1 Foundations.

Runs the Phase 1 agent workflow against baseline queries in shadow mode.
Output goes to the agent_traces table. No user-facing output.

Usage:
    uv run python scripts/shadow_run.py --limit 5
    uv run python scripts/shadow_run.py          # all queries

The baseline replay set is created by scripts/freeze_baseline.py.
If it does not exist, this script exits gracefully.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("shadow_run")

_BASELINE_PATH = Path("data/eval/baseline_replay_set.jsonl")

# Mapping from synthesis task type to agent task_type literal
_DEFAULT_TASK_TYPE = "cross_doc"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow runner for Phase 1 agent workflow")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of queries to run (default: all)")
    return parser.parse_args()


def _load_baseline(limit: int | None) -> list[dict]:
    """Load records from baseline_replay_set.jsonl. Returns [] if file missing."""
    if not _BASELINE_PATH.exists():
        logger.warning(
            "Baseline file not found: %s — run scripts/freeze_baseline.py first. Skipping.",
            _BASELINE_PATH,
        )
        return []

    records: list[dict] = []
    with open(_BASELINE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed JSONL line: %s", exc)

    if limit is not None:
        records = records[:limit]

    logger.info("Loaded %d baseline records from %s", len(records), _BASELINE_PATH)
    return records


def _build_initial_state(record: dict, request_id: str) -> dict:
    """Build an AgentState dict from a baseline record."""
    return {
        "request_id": request_id,
        "query": record.get("query", ""),
        "task_type": record.get("task_type", _DEFAULT_TASK_TYPE),
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


def main() -> None:
    args = _parse_args()

    # Deferred imports — only needed at runtime, not during import
    from src.agent.workflow import agent_graph
    from src.observability.tracing import emit_span

    records = _load_baseline(args.limit)

    if not records:
        logger.info("No records to process. Exiting.")
        return

    n_run = 0
    n_clean = 0
    n_budget = 0

    for record in records:
        request_id = str(uuid.uuid4())
        trace_id = request_id
        state = _build_initial_state(record, request_id)

        logger.info("Running query [%s]: %s...", request_id[:8], state["query"][:80])
        t0 = time.monotonic()

        try:
            final_state = agent_graph.invoke(state)
        except Exception as exc:
            logger.error("agent_graph.invoke failed for %s: %s", request_id[:8], exc)
            final_state = {**state, "termination_reason": "error"}

        elapsed_ms = int((time.monotonic() - t0) * 1000)
        termination_reason = final_state.get("termination_reason")

        # Emit a summary span for the completed run
        emit_span(
            trace_id=trace_id,
            step_no=final_state.get("steps_used", 0),
            node_name="shadow_run_summary",
            input_summary={"query": state["query"][:200], "task_type": state["task_type"]},
            output_summary={
                "sub_questions_count": len(final_state.get("sub_questions", [])),
                "coverage_count": len(final_state.get("coverage", {})),
                "termination_reason": termination_reason,
            },
            tool_called=None,
            latency_ms=elapsed_ms,
            cost_usd=final_state.get("cost_used_usd", 0.0),
            termination_reason=termination_reason,
        )

        n_run += 1
        if termination_reason in (None, "complete"):
            n_clean += 1
        else:
            n_budget += 1

    print(f"\nShadow run complete: {n_run} queries run, {n_clean} clean, {n_budget} hit budget/error")


if __name__ == "__main__":
    main()
