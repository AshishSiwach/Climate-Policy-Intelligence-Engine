"""
End-to-end evaluation runner:
  ground truth → retrieve → synthesise → judge → aggregate → write.

Runs the full pipeline for every ground truth pair, then scores each generated
answer with LLMJudge. Complements retrieval_eval_runner.py by measuring the
whole RAG output (not just retrieval).

Cost per full run scales with judge model. Log per-query costs for both the
synthesiser and the judge separately so total budget is transparent.

Run:
  uv run python -u -m evaluation.judge_runner
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evaluation.retrieval_metrics import (
    aggregate_metrics,
    evaluate_query,
    retrieved_doc_ids,
)
from retrieval import BM25Retriever, HybridRetriever
from retrieval.institution_detector import detect_institutions

# NOTE: evaluation.judge is deliberately NOT imported here. It pulls in openai,
# which on Windows breaks a later `from retrieval import DenseRetriever` (torch
# import hangs). LLMJudge is instantiated inside run() AFTER _build_pipeline().

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("data/eval/ground_truth.json")
BM25_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_DIR = Path("data/processed/chroma_db")
RESULTS_DIR = Path("data/eval/results")

RETRIEVAL_TOP_K = 5   # what the synthesiser sees; matches main.py default
KS = (5,)             # single k for judge runs (retrieval metrics already covered @5 and @10)

# Metadata filter mode — flip to False to reproduce the pre-filter baseline
# (used for A/B measurement of the filter's effect).
METADATA_FILTER_ENABLED = True

# Prompt version — flip to run A/B across prompt variants defined in
# synthesis.synthesiser.PROMPT_REGISTRY. None = use the module default.
PROMPT_VERSION_OVERRIDE: str | None = None

_PROBE_PATTERN = re.compile(r"\[PROBE:\s*(\w+)\]")


def _load_ground_truth() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit(
            f"Ground truth not found at {GROUND_TRUTH_PATH}. "
            "Run scripts/migrate_ground_truth.py first."
        )
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_pipeline():
    """Load indices + build HybridRetriever + Synthesiser. Deferred DenseRetriever
    import to avoid Windows torch-import hang (same fix as retrieval_eval_runner)."""
    from retrieval import DenseRetriever
    if not BM25_PATH.exists() or not CHROMA_DIR.exists():
        raise SystemExit("Indices missing. Run: uv run python scripts/build_indices.py")
    bm25 = BM25Retriever.load(BM25_PATH)
    dense = DenseRetriever(persist_dir=CHROMA_DIR)
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)

    # Synthesiser import must happen AFTER DenseRetriever, same Windows quirk.
    from synthesis import Synthesiser
    from synthesis.synthesiser import PROMPT_VERSION as _DEFAULT_PROMPT_VERSION
    version = PROMPT_VERSION_OVERRIDE or _DEFAULT_PROMPT_VERSION
    synth = Synthesiser(prompt_version=version)
    return hybrid, synth


def _extract_probe(notes: str | None) -> str | None:
    if not notes:
        return None
    m = _PROBE_PATTERN.search(notes)
    return m.group(1) if m else None


def _score_pair(pair: dict, hybrid, synth, judge) -> dict:
    """Run pipeline + judge for one pair. Returns a row for per_query results."""
    row: dict = {
        "id": pair["id"],
        "query_type": pair["query_type"],
        "probe": _extract_probe(pair.get("notes")),
    }

    # 1. Retrieval — institution filter (Step 2f)
    institutions = detect_institutions(pair["question"]) if METADATA_FILTER_ENABLED else []
    row["detected_institutions"] = institutions

    t0 = time.time()
    try:
        chunks = hybrid.retrieve(
            pair["question"], top_k=RETRIEVAL_TOP_K, institutions=institutions,
        )
    except Exception as e:
        logger.exception("Retrieval failed for %s", pair["id"])
        row["error"] = f"retrieval: {type(e).__name__}: {e}"
        return row
    row["retrieval_latency_ms"] = round((time.time() - t0) * 1000, 1)
    row["top_rrf"] = round(chunks[0]["rrf_score"], 6) if chunks else 0.0
    row["retrieved_doc_ids"] = list(dict.fromkeys(c["doc_id"] for c in chunks))

    # 2. Retrieval metrics (positives only). Negatives are graded downstream
    # by the judge — see _negatives_summary. The prior RRF-threshold based
    # `correctly_rejected` field was removed in Week 5 2b after calibration
    # showed the threshold was net-harmful (see docs/week5_failure_analysis.md).
    if pair["query_type"] != "negative":
        try:
            row.update(evaluate_query(chunks, pair["expected_sources"], ks=KS))
        except Exception as e:
            logger.exception("Retrieval metrics failed for %s", pair["id"])
            row["retrieval_metrics_error"] = f"{type(e).__name__}: {e}"

    # 3. Synthesis
    try:
        synth_result = synth.synthesise(pair["question"], chunks)
    except Exception as e:
        logger.exception("Synthesis failed for %s", pair["id"])
        row["error"] = f"synthesis: {type(e).__name__}: {e}"
        return row

    brief = synth_result["brief"]
    row["synthesis_latency_ms"] = round(synth_result["latency_ms"], 1)
    row["synthesis_cost_usd"] = synth_result["cost_usd"]
    row["synthesis_tokens"] = {
        "prompt": synth_result["prompt_tokens"],
        "completion": synth_result["completion_tokens"],
    }
    row["generated_answer"] = brief.answer
    row["generated_citation_count"] = len(brief.citations)
    row["prompt_version"] = synth_result.get("prompt_version")

    # 4. Judge
    try:
        judge_result = judge.score(
            question=pair["question"],
            query_type=pair["query_type"],
            expected_answer=pair["expected_answer"],
            retrieved_chunks=chunks,
            generated_answer=brief.answer,
        )
    except Exception as e:
        logger.exception("Judge failed for %s", pair["id"])
        row["judge_error"] = f"{type(e).__name__}: {e}"
        return row

    row["judge_latency_ms"] = round(judge_result["latency_ms"], 1)
    row["judge_cost_usd"] = judge_result["cost_usd"]
    row["judge_tokens"] = {
        "prompt": judge_result["prompt_tokens"],
        "completion": judge_result["completion_tokens"],
    }

    scores = judge_result.get("scores")
    if scores is None:
        row["judge_refusal_reason"] = judge_result.get("refusal_reason")
    else:
        row["judge_scores"] = scores.model_dump()
        # Flatten numeric scores for easy aggregation
        row["correctness"] = scores.correctness
        row["faithfulness"] = scores.faithfulness
        row["completeness"] = scores.completeness
        row["refusal_appropriateness"] = scores.refusal_appropriateness

    return row


def _aggregate_by_slice(per_query: list[dict], slice_key: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict]] = {}
    for r in per_query:
        val = r.get(slice_key)
        if val is None:
            continue
        groups.setdefault(str(val), []).append(r)
    return {
        group: {**aggregate_metrics(rows), "n": len(rows)}
        for group, rows in groups.items()
    }


def _negatives_summary(per_query: list[dict]) -> dict:
    """
    Judge-based negatives summary. `handled_well` = judge
    `refusal_appropriateness` >= 4 (the system correctly refused rather than
    fabricating). Replaces the retired RRF-threshold-based `correctly_rejected`.
    """
    negs = [r for r in per_query if r.get("query_type") == "negative"]
    if not negs:
        return {"n": 0}
    n_handled_well = sum(1 for r in negs if (r.get("refusal_appropriateness") or 0) >= 4)
    return {
        "n": len(negs),
        "n_handled_well": n_handled_well,
        "handled_well_rate": round(n_handled_well / len(negs), 4),
        "mean_refusal_appropriateness": round(
            sum((r.get("refusal_appropriateness") or 0) for r in negs) / len(negs), 3
        ),
    }


def run(output_path: Path | None = None) -> Path:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ground_truth = _load_ground_truth()
    logger.info("Loaded %d ground truth pairs", len(ground_truth))

    hybrid, synth = _build_pipeline()

    # Deferred import — pulling openai in earlier breaks DenseRetriever load on Windows.
    from evaluation.judge import LLMJudge

    judge = LLMJudge()
    logger.info(
        "Pipeline + judge ready. Judge=%s  metadata_filter=%s",
        judge.model, METADATA_FILTER_ENABLED,
    )

    t_run = time.time()
    per_query: list[dict] = []
    for i, pair in enumerate(ground_truth, 1):
        per_query.append(_score_pair(pair, hybrid, synth, judge))
        if i % 5 == 0:
            elapsed = time.time() - t_run
            logger.info("Scored %d/%d (%.0fs elapsed)", i, len(ground_truth), elapsed)

    total_wall_time_sec = time.time() - t_run

    # Aggregate — separate positives (have retrieval + judge scores) from negatives.
    scored = [r for r in per_query if "correctness" in r]

    total_synth_cost = sum(r.get("synthesis_cost_usd", 0.0) or 0.0 for r in per_query)
    total_judge_cost = sum(r.get("judge_cost_usd", 0.0) or 0.0 for r in per_query)

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(ground_truth),
        "n_scored": len(scored),
        "wall_time_sec": round(total_wall_time_sec, 1),
        "config": {
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "synthesis_model": synth.model,
            "judge_model": judge.model,
            "metadata_filter_enabled": METADATA_FILTER_ENABLED,
            "prompt_version": synth.prompt_version,
        },
        "cost_summary": {
            "synthesis_total_usd": round(total_synth_cost, 6),
            "judge_total_usd": round(total_judge_cost, 6),
            "total_usd": round(total_synth_cost + total_judge_cost, 6),
        },
        "overall_judge": aggregate_metrics(scored),
        "by_query_type": _aggregate_by_slice(scored, "query_type"),
        "by_probe": _aggregate_by_slice(per_query, "probe"),
        "negatives": _negatives_summary(per_query),
        "per_query": per_query,
    }

    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"judge_scores_{ts}_prompt-{synth.prompt_version}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Wrote results to %s", output_path)

    # ── Console summary ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"OVERALL JUDGE SCORES ({len(scored)} scored pairs)")
    print("=" * 80)
    for m in ("correctness", "faithfulness", "completeness", "refusal_appropriateness"):
        v = results["overall_judge"].get(m)
        if v is not None:
            print(f"  {m:<28} {v:.2f} / 5.0")

    print()
    print("=" * 80)
    print("BY QUERY TYPE")
    print("=" * 80)
    for qtype, agg in results["by_query_type"].items():
        print(f"\n  {qtype} (n={agg['n']})")
        for m in ("correctness", "faithfulness", "completeness", "refusal_appropriateness"):
            if m in agg:
                print(f"    {m:<28} {agg[m]:.2f}")

    print()
    print("=" * 80)
    print("BY PROBE")
    print("=" * 80)
    for probe, agg in results["by_probe"].items():
        print(f"\n  {probe} (n={agg['n']})")
        for m in ("correctness", "faithfulness", "completeness", "refusal_appropriateness"):
            if m in agg:
                print(f"    {m:<28} {agg[m]:.2f}")

    print()
    print("=" * 80)
    print(f"NEGATIVES (n={results['negatives']['n']}) — judge-scored handling")
    print("=" * 80)
    if results["negatives"]["n"]:
        n = results["negatives"]
        print(f"  Handled well:          {n['n_handled_well']}/{n['n']}"
              f" ({n['handled_well_rate']:.0%})   (refusal_appropriateness >= 4)")
        print(f"  Mean refusal_appr:     {n['mean_refusal_appropriateness']:.2f}")

    print()
    print("=" * 80)
    print("COST + LATENCY")
    print("=" * 80)
    print(f"  Wall time:            {total_wall_time_sec:.0f}s")
    print(f"  Synthesis cost:       ${results['cost_summary']['synthesis_total_usd']:.4f}")
    print(f"  Judge cost:           ${results['cost_summary']['judge_total_usd']:.4f}"
          f"  (0 if judge model pricing not yet in _JUDGE_PRICING)")
    print(f"  Total cost:           ${results['cost_summary']['total_usd']:.4f}")

    return output_path


if __name__ == "__main__":
    run()
