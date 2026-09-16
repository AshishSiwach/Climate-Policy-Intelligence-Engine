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

from dotenv import load_dotenv

load_dotenv()

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


def _run_fast_path(query: str, hybrid, synth) -> dict:
    """Run the fast-path synthesiser and return timing + result."""
    t0 = time.time()  # start before retrieval — apples-to-apples with agent timing
    chunks = hybrid.retrieve(query, top_k=5)  # production default
    result = synth.synthesise(query, chunks)
    latency_s = time.time() - t0

    brief = result["brief"]
    return {
        "fast_answer": brief.answer,
        "fast_citation_count": len(brief.citations),
        "fast_doc_ids": [c.doc_id for c in brief.citations],
        "fast_cost": result.get("cost_usd", 0.0),
        "fast_latency_s": latency_s,
        "_fast_chunks": chunks,  # kept for judge; stripped from output record
    }


# ---------------------------------------------------------------------------
# Agent path runner
# ---------------------------------------------------------------------------


def _run_agent_path(query: str, hybrid) -> dict:
    """Run the Phase 2 agent graph and return timing + result."""
    import uuid

    from src.agent.workflow import agent_graph
    from src.synthesis.output_schema import AnalystBrief

    t0_mono = time.monotonic()
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
        "_retriever": hybrid,
        "_start_time": t0_mono,  # enables wall-clock budget enforcement
    }

    t0 = time.time()
    final_state = agent_graph.invoke(initial_state)
    latency_s = time.time() - t0

    result_dict = final_state.get("result") or {}
    brief = AnalystBrief(**result_dict) if result_dict else None

    # Flatten per-sub-question retrievals so the judge can score faithfulness
    agent_retrievals: dict = final_state.get("retrievals", {})
    agent_chunks: list[dict] = []
    seen_cids: set[str] = set()
    for chunk_list in agent_retrievals.values():
        for chunk in chunk_list:
            cid = chunk.get("chunk_id")
            if cid and cid not in seen_cids:
                seen_cids.add(cid)
                agent_chunks.append(chunk)

    # Retrieval-level doc_ids (before synthesis — what the retriever actually fetched)
    agent_retrieved_doc_ids: list[str] = sorted({
        chunk.get("doc_id") for chunk in agent_chunks if chunk.get("doc_id")
    })

    # Coverage status distribution from grader
    raw_coverage: dict = final_state.get("coverage", {})
    coverage_statuses: dict[str, int] = {"covered": 0, "partial": 0, "not_covered": 0}
    for cov in raw_coverage.values():
        status = cov.status if hasattr(cov, "status") else cov.get("status", "")
        if status in coverage_statuses:
            coverage_statuses[status] += 1

    # Sub-question count from planner
    sub_question_count = len(final_state.get("sub_questions", []))

    return {
        "agent_answer": brief.answer if brief else "",
        "agent_citation_count": len(brief.citations) if brief else 0,
        "agent_doc_ids": [c.doc_id for c in brief.citations] if brief else [],
        "agent_cost": final_state.get("cost_used_usd", 0.0),
        "agent_latency_s": latency_s,
        "agent_termination_reason": final_state.get("termination_reason"),
        "agent_coverage_gaps": (brief.coverage_gaps if brief else []),
        "agent_truncated": (brief.truncated if brief else False),
        "agent_sub_question_count": sub_question_count,
        "agent_coverage_statuses": coverage_statuses,
        "_agent_retrieved_doc_ids": agent_retrieved_doc_ids,
        "_agent_chunks": agent_chunks,  # kept for judge; stripped from output record
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
            return _null_scores()
        return {
            "correctness": scores.correctness,
            "faithfulness": scores.faithfulness,
            "completeness": scores.completeness,
            "refusal_appropriateness": scores.refusal_appropriateness,
            "correctness_rationale": scores.correctness_rationale,
            "faithfulness_rationale": scores.faithfulness_rationale,
            "completeness_rationale": scores.completeness_rationale,
            "refusal_rationale": scores.refusal_appropriateness_rationale,
        }
    except Exception as exc:
        print(f"    [judge error] {exc}")
        return _null_scores()


def _null_scores() -> dict:
    return {
        "correctness": None, "faithfulness": None,
        "completeness": None, "refusal_appropriateness": None,
        "correctness_rationale": None, "faithfulness_rationale": None,
        "completeness_rationale": None, "refusal_rationale": None,
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def _run_dry(queries: list[dict]) -> None:
    """Print queries without calling any API."""
    print(f"DRY RUN — {len(queries)} cross-doc quer{'y' if len(queries) == 1 else 'ies'}:")
    for i, q in enumerate(queries, 1):
        print(f"  [{i}] {q.get('id', '?')}: {q.get('question', '')[:80]}")


# ---------------------------------------------------------------------------
# Source recall helper
# ---------------------------------------------------------------------------


def _source_recall(cited_doc_ids: set[str], expected_sources: set[str]) -> float | None:
    """Fraction of expected source documents that appear in cited doc IDs."""
    if not expected_sources:
        return None
    return len(cited_doc_ids & expected_sources) / len(expected_sources)


# ---------------------------------------------------------------------------
# A/B runner
# ---------------------------------------------------------------------------


def _run_ab(queries: list[dict], use_judge: bool) -> None:
    """Run the A/B evaluation and write results to JSONL."""
    from main import build_pipeline  # type: ignore[import]

    print("Building pipeline (loading indices + embedding model)...")
    hybrid, synth = build_pipeline()
    print("Pipeline ready.\n")

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
                fast = _run_fast_path(question, hybrid, synth)
            except Exception as exc:
                fast = {
                    "fast_answer": f"ERROR: {exc}",
                    "fast_citations": 0,
                    "fast_cost": 0.0,
                    "fast_latency_s": 0.0,
                    "_fast_chunks": [],
                }

            try:
                agent = _run_agent_path(question, hybrid)
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

            # Source recall (independent of judge)
            # expected_sources is a list of {doc_id, page_range} dicts — extract doc_ids
            raw_sources = q.get("expected_sources", [])
            expected_sources: set[str] = {
                (s["doc_id"] if isinstance(s, dict) else s) for s in raw_sources
            }
            fast_source_recall = _source_recall(set(fast.get("fast_doc_ids", [])), expected_sources)
            agent_source_recall = _source_recall(set(agent.get("agent_doc_ids", [])), expected_sources)
            # Retrieval-level recall — what the retriever fetched before synthesis
            agent_retrieval_recall = _source_recall(
                set(agent.get("_agent_retrieved_doc_ids", [])), expected_sources
            )

            # Judge scoring (optional)
            fast_scores = _null_scores()
            agent_scores = _null_scores()

            if use_judge:
                print("    [judge] scoring fast path ...")
                fast_scores = _judge_answer(
                    question=question,
                    query_type=query_type,
                    expected_answer=expected_answer,
                    generated_answer=fast["fast_answer"],
                    chunks=fast.get("_fast_chunks", []),
                )

                print("    [judge] scoring agent path ...")
                agent_scores = _judge_answer(
                    question=question,
                    query_type=query_type,
                    expected_answer=expected_answer,
                    generated_answer=agent["agent_answer"],
                    chunks=agent.get("_agent_chunks", []),
                )

            record = {
                "id": query_id,
                "question": question,
                "expected_answer": expected_answer,
                "expected_sources": sorted(expected_sources),
                # fast path
                "fast_answer": fast["fast_answer"],
                "fast_citation_count": fast.get("fast_citation_count", 0),
                "fast_doc_ids": fast.get("fast_doc_ids", []),
                "fast_source_recall": fast_source_recall,
                "fast_cost": fast["fast_cost"],
                "fast_latency_s": fast["fast_latency_s"],
                "fast_correctness": fast_scores["correctness"],
                "fast_faithfulness": fast_scores["faithfulness"],
                "fast_completeness": fast_scores["completeness"],
                "fast_refusal_appropriateness": fast_scores["refusal_appropriateness"],
                "fast_correctness_rationale": fast_scores["correctness_rationale"],
                "fast_faithfulness_rationale": fast_scores["faithfulness_rationale"],
                "fast_completeness_rationale": fast_scores["completeness_rationale"],
                "fast_refusal_rationale": fast_scores["refusal_rationale"],
                # agent path
                "agent_answer": agent["agent_answer"],
                "agent_citation_count": agent.get("agent_citation_count", 0),
                "agent_doc_ids": agent.get("agent_doc_ids", []),
                "agent_source_recall": agent_source_recall,
                "agent_cost": agent["agent_cost"],
                "agent_latency_s": agent["agent_latency_s"],
                "agent_correctness": agent_scores["correctness"],
                "agent_faithfulness": agent_scores["faithfulness"],
                "agent_completeness": agent_scores["completeness"],
                "agent_refusal_appropriateness": agent_scores["refusal_appropriateness"],
                "agent_correctness_rationale": agent_scores["correctness_rationale"],
                "agent_faithfulness_rationale": agent_scores["faithfulness_rationale"],
                "agent_completeness_rationale": agent_scores["completeness_rationale"],
                "agent_refusal_rationale": agent_scores["refusal_rationale"],
                "agent_termination_reason": agent["agent_termination_reason"],
                "agent_coverage_gaps": agent["agent_coverage_gaps"],
                "agent_truncated": agent["agent_truncated"],
                "agent_sub_question_count": agent.get("agent_sub_question_count", 0),
                "agent_coverage_statuses": agent.get("agent_coverage_statuses", {}),
                "agent_retrieved_doc_ids": agent.get("_agent_retrieved_doc_ids", []),
                "agent_retrieval_recall": agent_retrieval_recall,
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
    _print_summary(records)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def _mean(values: list) -> float | None:
    """Mean of a list, ignoring None. Returns None if all values are None."""
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def _p95(values: list) -> float | None:
    """95th-percentile of a list, ignoring None."""
    clean = sorted(v for v in values if v is not None)
    if not clean:
        return None
    idx = int(len(clean) * 0.95)
    return clean[min(idx, len(clean) - 1)]


def _fmt(v: float | None, precision: int = 2) -> str:
    return f"{v:.{precision}f}" if v is not None else "N/A"


def _delta(fast_v: float | None, agent_v: float | None) -> str:
    if fast_v is None or agent_v is None:
        return "N/A"
    d = agent_v - fast_v
    return f"+{d:.2f}" if d >= 0 else f"{d:.2f}"


def _print_summary(records: list[dict]) -> None:
    n = len(records)

    fast_correctness = _mean([r.get("fast_correctness") for r in records])
    fast_faithfulness = _mean([r.get("fast_faithfulness") for r in records])
    fast_completeness = _mean([r.get("fast_completeness") for r in records])
    fast_refusal = _mean([r.get("fast_refusal_appropriateness") for r in records])
    fast_source_recall = _mean([r.get("fast_source_recall") for r in records])
    fast_cost = _mean([r["fast_cost"] for r in records])
    fast_latency = _mean([r["fast_latency_s"] for r in records])
    fast_p95 = _p95([r["fast_latency_s"] for r in records])

    agent_correctness = _mean([r.get("agent_correctness") for r in records])
    agent_faithfulness = _mean([r.get("agent_faithfulness") for r in records])
    agent_completeness = _mean([r.get("agent_completeness") for r in records])
    agent_refusal = _mean([r.get("agent_refusal_appropriateness") for r in records])
    agent_source_recall = _mean([r.get("agent_source_recall") for r in records])
    agent_cost = _mean([r["agent_cost"] for r in records])
    agent_latency = _mean([r["agent_latency_s"] for r in records])
    agent_p95 = _p95([r["agent_latency_s"] for r in records])

    # Gates from AGENT_ROUTE_REASONING.md / AGENT_ROUTE_PLAN.md
    CORRECTNESS_GATE = 3.50
    COMPLETENESS_GATE = 3.25
    LATENCY_P95_KILL = 15.0

    def _gate(v: float | None, threshold: float, fail_above: bool = False) -> str:
        if v is None:
            return "N/A"
        passing = (v <= threshold) if fail_above else (v >= threshold)
        return "PASS" if passing else "FAIL"

    ruler = "-" * 52
    print(f"\nCross-doc A/B Results (N={n})")
    print(ruler)
    print(f"{'Metric':<24} {'Fast':>7}  {'Agent':>7}  {'Delta':>7}")
    print(ruler)
    print(f"{'Correctness mean':<24} {_fmt(fast_correctness):>7}  {_fmt(agent_correctness):>7}  {_delta(fast_correctness, agent_correctness):>7}")
    print(f"{'Faithfulness mean':<24} {_fmt(fast_faithfulness):>7}  {_fmt(agent_faithfulness):>7}  {_delta(fast_faithfulness, agent_faithfulness):>7}")
    print(f"{'Completeness mean':<24} {_fmt(fast_completeness):>7}  {_fmt(agent_completeness):>7}  {_delta(fast_completeness, agent_completeness):>7}")
    print(f"{'Refusal approp. mean':<24} {_fmt(fast_refusal):>7}  {_fmt(agent_refusal):>7}  {_delta(fast_refusal, agent_refusal):>7}")
    print(f"{'Source recall mean':<24} {_fmt(fast_source_recall):>7}  {_fmt(agent_source_recall):>7}  {_delta(fast_source_recall, agent_source_recall):>7}")
    print(f"{'Cost mean ($)':<24} {_fmt(fast_cost, 3):>7}  {_fmt(agent_cost, 3):>7}  {_delta(fast_cost, agent_cost):>7}")
    print(f"{'Latency mean (s)':<24} {_fmt(fast_latency, 1):>7}  {_fmt(agent_latency, 1):>7}  {_delta(fast_latency, agent_latency):>7}")
    print(f"{'Latency P95 (s)':<24} {_fmt(fast_p95, 1):>7}  {_fmt(agent_p95, 1):>7}  {_delta(fast_p95, agent_p95):>7}")
    print(ruler)
    print(f"Gate: Agent Correctness  >=3.50 -> {_gate(agent_correctness, CORRECTNESS_GATE)}")
    print(f"Gate: Agent Completeness >=3.25 -> {_gate(agent_completeness, COMPLETENESS_GATE)}")
    print(f"Gate: Agent Latency P95  <=15s  -> {_gate(agent_p95, LATENCY_P95_KILL, fail_above=True)}")

    # Retrieval section (always shown — no judge required)
    agent_retrieval_recall = _mean([r.get("agent_retrieval_recall") for r in records])
    avg_sq = _mean([r.get("agent_sub_question_count") for r in records])

    cov_counts: dict[str, list[int]] = {"covered": [], "partial": [], "not_covered": []}
    for r in records:
        cs = r.get("agent_coverage_statuses", {})
        total_sq = sum(cs.values()) or 1
        for k in cov_counts:
            cov_counts[k].append(cs.get(k, 0) / total_sq * 100)

    print(f"\nRetrieval (agent path, N={n})")
    print(ruler)
    print(f"  Retrieval recall (pre-synthesis) : {_fmt(agent_retrieval_recall)}")
    print(f"  Citation recall  (post-synthesis): {_fmt(_mean([r.get('agent_source_recall') for r in records]))}")
    recall_gap = (
        (agent_retrieval_recall or 0) - (_mean([r.get("agent_source_recall") for r in records]) or 0)
    )
    print(f"  Retrieval -> citation gap        : {recall_gap:+.2f}  {'(synthesis drops docs)' if recall_gap > 0.05 else '(ok)'}")
    print(f"  Avg sub-questions per query      : {_fmt(avg_sq, 1)}")
    print(f"  Coverage: covered {_fmt(_mean(cov_counts['covered']), 0)}%  "
          f"partial {_fmt(_mean(cov_counts['partial']), 0)}%  "
          f"not_covered {_fmt(_mean(cov_counts['not_covered']), 0)}%")


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
