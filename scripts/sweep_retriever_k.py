"""
Retriever k-sweep — agent path only.

Determines the best RETRIEVER_TOP_K by running the agent path over a
stratified 15-query sample at k ∈ {5, 6, 8, 10, 12} and scoring with
the LLM judge.

Sample design (stratified):
  - All 6 previously-failing queries (completeness=2 at k=5)
  - 9 randomly sampled from the 24 passing queries (seed=42)

Decision rule: lowest k that clears both gates
  - Correctness ≥ 3.50
  - Completeness ≥ 3.25

Output:
  data/eval/sweep_k_results.jsonl  — one record per (k, query_id)
  Printed summary table

Usage:
    uv run python scripts/sweep_retriever_k.py
    uv run python scripts/sweep_retriever_k.py --dry-run   # print sample only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

K_VALUES = [5, 6, 8, 10, 12]

GATES = {"correctness": 3.50, "completeness": 3.25}

# Queries that scored completeness=2 at k=5 — always included
LOW_COMPLETENESS_IDS = {
    "xdoc_eval_005_ccc_progress_change",
    "xdoc_eval_010_uk_global_ev_trajectory",
    "xdoc_eval_024_transport_assumption_progress",
    "xdoc_eval_026_cbes_design_results",
    "xdoc_eval_029_gdp_macro_channels",
    "xdoc_eval_030_boe_iea_fossil",
}

N_PASSING_SAMPLE = 9   # from the 24 passing queries
RANDOM_SEED = 42       # fixed for reproducibility

GT_PATH = _REPO_ROOT / "data" / "eval" / "cross_document_ground_truth.json"
OUTPUT_PATH = _REPO_ROOT / "data" / "eval" / "sweep_k_results.jsonl"


# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------

def _build_sample(all_queries: list[dict]) -> list[dict]:
    failing = [q for q in all_queries if q["id"] in LOW_COMPLETENESS_IDS]
    passing = [q for q in all_queries if q["id"] not in LOW_COMPLETENESS_IDS]

    rng = random.Random(RANDOM_SEED)
    sampled_passing = rng.sample(passing, min(N_PASSING_SAMPLE, len(passing)))

    sample = failing + sampled_passing
    print(f"Stratified sample: {len(failing)} failing + {len(sampled_passing)} passing = {len(sample)} queries")
    print(f"  Failing: {[q['id'] for q in failing]}")
    print(f"  Passing sample (seed={RANDOM_SEED}): {[q['id'] for q in sampled_passing]}")
    return sample


# ---------------------------------------------------------------------------
# Single agent run
# ---------------------------------------------------------------------------

def _run_agent(query: str, hybrid) -> dict:
    from src.agent.workflow import agent_graph

    state = {
        "request_id": str(uuid.uuid4()),
        "query": query,
        "task_type": "cross_doc",
        "sub_questions": [], "retrievals": {}, "coverage": {},
        "claims": [], "verified_claims": [],
        "steps_used": 0, "cost_used_usd": 0.0, "time_used_s": 0.0,
        "retries_used": {}, "result": None, "termination_reason": None,
        "_retriever": hybrid,
    }
    t0 = time.time()
    final = agent_graph.invoke(state)
    latency = time.time() - t0
    result = final.get("result") or {}
    return {
        "answer": result.get("answer", ""),
        "latency_s": latency,
        "cost_usd": final.get("cost_used_usd", 0.0),
        "termination_reason": final.get("termination_reason"),
    }


# ---------------------------------------------------------------------------
# Judge scoring
# ---------------------------------------------------------------------------

def _judge(question: str, query_type: str, expected: str, answer: str) -> dict:
    from src.evaluation.judge import LLMJudge
    judge = LLMJudge()
    try:
        result = judge.score(
            question=question,
            query_type=query_type,
            expected_answer=expected,
            retrieved_chunks=[],
            generated_answer=answer,
        )
        scores = result.get("scores")
        if scores:
            return {"correctness": scores.correctness, "completeness": scores.completeness}
    except Exception as exc:
        print(f"    [judge error] {exc}")
    return {"correctness": None, "completeness": None}


# ---------------------------------------------------------------------------
# Sweep runner
# ---------------------------------------------------------------------------

def _run_sweep(sample: list[dict]) -> None:
    import src.agent.policies as _policies
    from main import build_pipeline  # type: ignore[import]

    print("\nBuilding pipeline...")
    hybrid, _ = build_pipeline()
    print("Pipeline ready.\n")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for k in K_VALUES:
            _policies.RETRIEVER_TOP_K = k
            print(f"\n{'='*50}")
            print(f"k = {k}")
            print(f"{'='*50}")

            k_correct: list[float] = []
            k_complete: list[float] = []
            k_latency: list[float] = []
            k_cost: list[float] = []

            for i, q in enumerate(sample, 1):
                qid = q["id"]
                question = q["question"]
                expected = q.get("expected_answer", "")
                qtype = q.get("query_type", "cross_document")

                print(f"  [{i}/{len(sample)}] {qid}")

                try:
                    agent = _run_agent(question, hybrid)
                except Exception as exc:
                    print(f"    ERROR: {exc}")
                    agent = {"answer": "", "latency_s": 0.0, "cost_usd": 0.0, "termination_reason": "error"}

                scores = _judge(question, qtype, expected, agent["answer"])

                record = {
                    "k": k,
                    "id": qid,
                    "termination_reason": agent["termination_reason"],
                    "latency_s": agent["latency_s"],
                    "cost_usd": agent["cost_usd"],
                    "correctness": scores["correctness"],
                    "completeness": scores["completeness"],
                }
                records.append(record)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()

                c, cm = scores["correctness"], scores["completeness"]
                if c is not None:
                    k_correct.append(c)
                if cm is not None:
                    k_complete.append(cm)
                k_latency.append(agent["latency_s"])
                k_cost.append(agent["cost_usd"])

                print(f"    correct={c}  complete={cm}  latency={agent['latency_s']:.1f}s  reason={agent['termination_reason']}")

            mean_c = sum(k_correct) / len(k_correct) if k_correct else None
            mean_cm = sum(k_complete) / len(k_complete) if k_complete else None
            mean_lat = sum(k_latency) / len(k_latency) if k_latency else None
            mean_cost = sum(k_cost) / len(k_cost) if k_cost else None
            print(f"\n  k={k} summary: correctness={mean_c:.2f}  completeness={mean_cm:.2f}  latency={mean_lat:.1f}s  cost=${mean_cost:.4f}")

    _print_summary(records)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _mean(vals: list) -> float | None:
    clean = [v for v in vals if v is not None]
    return sum(clean) / len(clean) if clean else None


def _gate(v: float | None, threshold: float) -> str:
    if v is None:
        return "N/A "
    return "PASS" if v >= threshold else "FAIL"


def _print_summary(records: list[dict]) -> None:
    print(f"\n\nK-SWEEP SUMMARY (N per k = {len(records) // len(K_VALUES)})")
    ruler = "─" * 72
    print(ruler)
    print(f"{'k':>4}  {'Correct':>8}  {'Gate':>4}  {'Complete':>9}  {'Gate':>4}  {'Latency(s)':>10}  {'Cost($)':>8}")
    print(ruler)

    first_pass_k = None
    for k in K_VALUES:
        k_recs = [r for r in records if r["k"] == k]
        mc = _mean([r["correctness"] for r in k_recs])
        mcm = _mean([r["completeness"] for r in k_recs])
        ml = _mean([r["latency_s"] for r in k_recs])
        mco = _mean([r["cost_usd"] for r in k_recs])
        cg = _gate(mc, GATES["correctness"])
        cmg = _gate(mcm, GATES["completeness"])

        passes = (mc is not None and mc >= GATES["correctness"] and
                  mcm is not None and mcm >= GATES["completeness"])
        marker = " ← recommended" if (passes and first_pass_k is None) else ""
        if passes and first_pass_k is None:
            first_pass_k = k

        print(f"{k:>4}  {mc or 0:>8.2f}  {cg}  {mcm or 0:>9.2f}  {cmg}  {ml or 0:>10.1f}  {mco or 0:>8.4f}{marker}")

    print(ruler)
    if first_pass_k:
        print(f"\nDecision: set RETRIEVER_TOP_K = {first_pass_k} (lowest k clearing both gates)")
    else:
        print("\nDecision: no k value cleared both gates — review corpus coverage")

    print(f"\nRaw records written to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Retriever k-sweep for agent path.")
    parser.add_argument("--dry-run", action="store_true", help="Print sample selection and exit.")
    args = parser.parse_args()

    with open(GT_PATH, encoding="utf-8") as f:
        all_queries: list[dict] = json.load(f)

    sample = _build_sample(all_queries)

    if args.dry_run:
        print("\nDRY RUN — sample selected, no API calls made.")
        return

    _run_sweep(sample)


if __name__ == "__main__":
    main()
