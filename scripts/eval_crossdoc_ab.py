"""
Cross-doc A/B evaluation — agent path vs fast path.

Reads data/eval/cross_document_ground_truth.json (30 adjudicated cross-doc
questions) and runs both paths, writing results to
data/eval/ab_results_crossdoc.jsonl.

Usage:
    uv run python scripts/eval_crossdoc_ab.py --limit 5 --dry-run
    uv run python scripts/eval_crossdoc_ab.py --judge

Options:
    --dry-run       Print queries without calling APIs.
    --limit N       Run first N queries only.
    --judge         After running both paths, call the LLM judge to score
                    correctness and completeness for each answer against
                    expected_answer.

Output record (one JSON line per query):
    {
      "id": str,
      "question": str,
      "expected_answer": str,
      "fast_answer": str,
      "fast_cost": float,
      "fast_latency_s": float,
      "fast_correctness": float | null,
      "fast_completeness": float | null,
      "agent_answer": str,
      "agent_cost": float,
      "agent_latency_s": float,
      "agent_correctness": float | null,
      "agent_completeness": float | null,
      "agent_termination_reason": str | null,
      "agent_coverage_gaps": [str],
      "agent_truncated": bool
    }

Do NOT execute this script directly — it makes real API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on path when running as a script
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

CROSS_DOC_GT_PATH = _REPO_ROOT / "data" / "eval" / "cross_document_ground_truth.json"
OUTPUT_PATH = _REPO_ROOT / "data" / "eval" / "ab_results_crossdoc.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_cross_doc_queries(limit: int | None) -> list[dict]:
    """Load the 30-question adjudicated cross-doc ground truth set."""
    with open(CROSS_DOC_GT_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    if limit is not None:
        data = data[:limit]

    return data


# ---------------------------------------------------------------------------
# Fast path runner
# ---------------------------------------------------------------------------


def _run_fast_path(query: str) -> dict:
    """Run the fast-path synthesiser and return timing + result."""
    from retrieval.hybrid_retriever import HybridRetriever  # type: ignore[import]
    from synthesis.synthesiser import Synthesiser

    retriever = HybridRetriever()
    chunks = retriever.retrieve(query, top_k=10)

    synth = Synthesiser()
    t0 = time.time()
    result = synth.synthesise(query, chunks)
    latency_s = time.time() - t0

    brief = result["brief"]
    return {
        "fast_answer": brief.answer,
        "fast_citations": len(brief.citations),
        "fast_cost": result.get("cost_usd", 0.0),
        "fast_latency_s": latency_s,
        "_fast_chunks": chunks,  # kept for judge; stripped from output record
    }


# ---------------------------------------------------------------------------
# Agent path runner
# ---------------------------------------------------------------------------


def _run_agent_path(query: str) -> dict:
    """Run the Phase 2 agent graph and return timing + result."""
    import uuid

    from src.agent.workflow import agent_graph
    from src.synthesis.output_schema import AnalystBrief

    initial_state = {
        "request_id": str(uuid.uuid4()),
        "query": query,
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

    t0 = time.time()
    final_state = agent_graph.invoke(initial_state)
    latency_s = time.time() - t0

    result_dict = final_state.get("result") or {}
    brief = AnalystBrief(**result_dict) if result_dict else None

    return {
        "agent_answer": brief.answer if brief else "",
        "agent_citations": len(brief.citations) if brief else 0,
        "agent_cost": final_state.get("cost_used_usd", 0.0),
        "agent_latency_s": latency_s,
        "agent_termination_reason": final_state.get("termination_reason"),
        "agent_coverage_gaps": (brief.coverage_gaps if brief else []),
        "agent_truncated": (brief.truncated if brief else False),
    }


# ---------------------------------------------------------------------------
# LLM judge scoring
# ---------------------------------------------------------------------------


def _judge_answer(
    *,
    question: str,
    query_type: str,
    expected_answer: str,
    generated_answer: str,
    chunks: list[dict],
) -> dict:
    """Score one answer with the LLM judge. Returns correctness and completeness (1-5)."""
    from src.evaluation.judge import LLMJudge

    judge = LLMJudge()
    try:
        result = judge.score(
            question=question,
            query_type=query_type,
            expected_answer=expected_answer,
            retrieved_chunks=chunks,
            generated_answer=generated_answer,
        )
        scores = result.get("scores")
        if scores is None:
            return {"correctness": None, "completeness": None}
        return {
            "correctness": scores.correctness,
            "completeness": scores.completeness,
        }
    except Exception as exc:
        print(f"    [judge error] {exc}")
        return {"correctness": None, "completeness": None}


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _run_dry(queries: list[dict]) -> None:
    """Print queries without calling any API."""
    print(f"DRY RUN — {len(queries)} cross-doc quer{'y' if len(queries) == 1 else 'ies'}:")
    for i, q in enumerate(queries, 1):
        print(f"  [{i}] {q.get('id', '?')}: {q.get('question', '')[:80]}")


# ---------------------------------------------------------------------------
# A/B runner
# ---------------------------------------------------------------------------


def _run_ab(queries: list[dict], use_judge: bool) -> None:
    """Run the A/B evaluation and write results to JSONL."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, q in enumerate(queries, 1):
            question = q.get("question", "")
            query_id = q.get("id", f"q_{i}")
            expected_answer = q.get("expected_answer", "")
            query_type = q.get("query_type", "cross_document")

            print(f"[{i}/{len(queries)}] Running: {query_id!r} ...")

            try:
                fast = _run_fast_path(question)
            except Exception as exc:
                fast = {
                    "fast_answer": f"ERROR: {exc}",
                    "fast_citations": 0,
                    "fast_cost": 0.0,
                    "fast_latency_s": 0.0,
                    "_fast_chunks": [],
                }

            try:
                agent = _run_agent_path(question)
            except Exception as exc:
                agent = {
                    "agent_answer": f"ERROR: {exc}",
                    "agent_citations": 0,
                    "agent_cost": 0.0,
                    "agent_latency_s": 0.0,
                    "agent_termination_reason": "error",
                    "agent_coverage_gaps": [],
                    "agent_truncated": False,
                }

            # Judge scoring (optional)
            fast_correctness = fast_completeness = None
            agent_correctness = agent_completeness = None

            if use_judge:
                print("    [judge] scoring fast path ...")
                fast_scores = _judge_answer(
                    question=question,
                    query_type=query_type,
                    expected_answer=expected_answer,
                    generated_answer=fast["fast_answer"],
                    chunks=fast.get("_fast_chunks", []),
                )
                fast_correctness = fast_scores["correctness"]
                fast_completeness = fast_scores["completeness"]

                print("    [judge] scoring agent path ...")
                agent_scores = _judge_answer(
                    question=question,
                    query_type=query_type,
                    expected_answer=expected_answer,
                    generated_answer=agent["agent_answer"],
                    chunks=[],  # agent retrieves per sub-question; no single chunk list
                )
                agent_correctness = agent_scores["correctness"]
                agent_completeness = agent_scores["completeness"]

            record = {
                "id": query_id,
                "question": question,
                "expected_answer": expected_answer,
                "fast_answer": fast["fast_answer"],
                "fast_cost": fast["fast_cost"],
                "fast_latency_s": fast["fast_latency_s"],
                "fast_correctness": fast_correctness,
                "fast_completeness": fast_completeness,
                "agent_answer": agent["agent_answer"],
                "agent_cost": agent["agent_cost"],
                "agent_latency_s": agent["agent_latency_s"],
                "agent_correctness": agent_correctness,
                "agent_completeness": agent_completeness,
                "agent_termination_reason": agent["agent_termination_reason"],
                "agent_coverage_gaps": agent["agent_coverage_gaps"],
                "agent_truncated": agent["agent_truncated"],
            }

            records.append(record)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()

            print(
                f"  fast={fast['fast_latency_s']:.2f}s  "
                f"agent={agent['agent_latency_s']:.2f}s  "
                f"reason={agent['agent_termination_reason']}"
            )

    print(f"\nResults written to {OUTPUT_PATH}")

    if use_judge:
        _print_summary(records)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _mean(values: list) -> float | None:
    """Mean of a list, ignoring None. Returns None if all values are None."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _fmt(v: float | None, precision: int = 2) -> str:
    return f"{v:.{precision}f}" if v is not None else "N/A"


def _delta(fast_v: float | None, agent_v: float | None) -> str:
    if fast_v is None or agent_v is None:
        return "N/A"
    d = agent_v - fast_v
    return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"


def _print_summary(records: list[dict]) -> None:
    n = len(records)

    fast_correctness = _mean([r["fast_correctness"] for r in records])
    fast_completeness = _mean([r["fast_completeness"] for r in records])
    fast_cost = _mean([r["fast_cost"] for r in records])
    fast_latency = _mean([r["fast_latency_s"] for r in records])

    agent_correctness = _mean([r["agent_correctness"] for r in records])
    agent_completeness = _mean([r["agent_completeness"] for r in records])
    agent_cost = _mean([r["agent_cost"] for r in records])
    agent_latency = _mean([r["agent_latency_s"] for r in records])

    # Gates from AGENT_ROUTE_REASONING.md
    CORRECTNESS_GATE = 3.50
    COMPLETENESS_GATE = 3.25

    def _gate(v: float | None, threshold: float) -> str:
        if v is None:
            return "N/A"
        return "PASS" if v >= threshold else "FAIL"

    ruler = "─" * 43
    print(f"\nCross-doc A/B Results (N={n})")
    print(ruler)
    print(f"{'Metric':<20} {'Fast':>6}  {'Agent':>6}  {'Δ':>6}")
    print(ruler)
    print(f"{'Correctness mean':<20} {_fmt(fast_correctness):>6}  {_fmt(agent_correctness):>6}  {_delta(fast_correctness, agent_correctness):>6}")
    print(f"{'Completeness mean':<20} {_fmt(fast_completeness):>6}  {_fmt(agent_completeness):>6}  {_delta(fast_completeness, agent_completeness):>6}")
    print(f"{'Cost mean ($)':<20} {_fmt(fast_cost, 3):>6}  {_fmt(agent_cost, 3):>6}  {_delta(fast_cost, agent_cost):>6}")
    print(f"{'Latency mean (s)':<20} {_fmt(fast_latency, 1):>6}  {_fmt(agent_latency, 1):>6}  {_delta(fast_latency, agent_latency):>6}")
    print(ruler)
    print(f"Gate: Correctness  ≥{CORRECTNESS_GATE}  → {_gate(agent_correctness, CORRECTNESS_GATE)}")
    print(f"Gate: Completeness ≥{COMPLETENESS_GATE}  → {_gate(agent_completeness, COMPLETENESS_GATE)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-doc A/B evaluation: agent vs fast path.")
    parser.add_argument("--limit", type=int, default=None, help="Run first N queries only.")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without calling APIs.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help=(
            "After running both paths, call the LLM judge to score correctness "
            "and completeness for each answer. Adds ~$0.002 per query."
        ),
    )
    args = parser.parse_args()

    queries = _load_cross_doc_queries(limit=args.limit)

    if not queries:
        print(f"No queries found in {CROSS_DOC_GT_PATH}.")
        sys.exit(0)

    if args.dry_run:
        _run_dry(queries)
    else:
        _run_ab(queries, use_judge=args.judge)


if __name__ == "__main__":
    main()
