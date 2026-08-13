"""
Four-config retrieval ablation.

Runs the full pipeline (retrieve → synthesise → judge) across the 47
ground-truth pairs under four retrieval configurations, then aggregates for
per-config comparison.

Configurations:
    bm25            — BM25 only, top_k=5
    dense           — Dense (Chroma / BAAI/bge-base-en-v1.5) only, top_k=5
    hybrid          — BM25 + Dense → RRF (k=60), top_k=5    [default]
    hybrid_rerank   — hybrid top-20 → cross-encoder rerank → top-5

Everything downstream (synthesis, judge) is held constant across configs so
the delta is attributable to the retrieval choice alone.

Output:
    data/eval/results/ablation_<ts>.json
      keys:
        timestamp, n_queries, wall_time_sec, cost_summary, per_config
        per_config[<name>]:
          overall_judge, by_query_type, by_probe, negatives,
          retrieval_latency_ms_mean, per_query

Run:
    uv run python -u -m evaluation.ablation_runner
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from evaluation.retrieval_metrics import aggregate_metrics, evaluate_query
from retrieval import BM25Retriever, HybridRetriever

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("data/eval/ground_truth.json")
BM25_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_DIR = Path("data/processed/chroma_db")
RESULTS_DIR = Path("data/eval/results")

RETRIEVAL_TOP_K = 5      # what the synthesiser sees; matches main.py default
RERANK_CANDIDATE_K = 20  # hybrid produces 20 candidates for reranker
KS = (5,)

_PROBE_PATTERN = re.compile(r"\[PROBE:\s*(\w+)\]")


# ---------------------------------------------------------------------------
# Config definitions — each config carries its retrieve() closure + the field
# name that holds the "top-1 score" for that retriever family.
# ---------------------------------------------------------------------------

def _build_configs():
    """Load indices once, wrap each config as a callable + score-field spec."""
    # Deferred imports for Windows torch DLL-load order (see judge_runner).
    from retrieval import DenseRetriever
    if not BM25_PATH.exists() or not CHROMA_DIR.exists():
        raise SystemExit("Indices missing. Run: uv run python scripts/build_indices.py")

    bm25 = BM25Retriever.load(BM25_PATH)
    dense = DenseRetriever(persist_dir=CHROMA_DIR)
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)

    from retrieval.reranker import Reranker
    reranker = Reranker(top_k=RETRIEVAL_TOP_K)

    def bm25_retrieve(query: str) -> list[dict]:
        return bm25.query(query, top_k=RETRIEVAL_TOP_K)

    def dense_retrieve(query: str) -> list[dict]:
        return dense.query(query, top_k=RETRIEVAL_TOP_K)

    def hybrid_retrieve(query: str) -> list[dict]:
        return hybrid.retrieve(query, top_k=RETRIEVAL_TOP_K)

    def hybrid_rerank_retrieve(query: str) -> list[dict]:
        candidates = hybrid.retrieve(query, top_k=RERANK_CANDIDATE_K)
        return reranker.rerank(query, candidates)

    return {
        "bm25":          {"retrieve": bm25_retrieve,          "top_score_field": "bm25_score"},
        "dense":         {"retrieve": dense_retrieve,         "top_score_field": "dense_score"},
        "hybrid":        {"retrieve": hybrid_retrieve,        "top_score_field": "rrf_score"},
        "hybrid_rerank": {"retrieve": hybrid_rerank_retrieve, "top_score_field": "rerank_score"},
    }


# ---------------------------------------------------------------------------
# Per-query scoring — mirrors judge_runner._score_pair, config-aware.
# ---------------------------------------------------------------------------

def _extract_probe(notes: str | None) -> str | None:
    if not notes:
        return None
    m = _PROBE_PATTERN.search(notes)
    return m.group(1) if m else None


def _score_pair(pair: dict, retrieve, top_score_field: str, synth, judge) -> dict:
    row: dict = {
        "id": pair["id"],
        "query_type": pair["query_type"],
        "probe": _extract_probe(pair.get("notes")),
    }

    # 1. Retrieval
    t0 = time.time()
    try:
        chunks = retrieve(pair["question"])
    except Exception as e:
        logger.exception("Retrieval failed for %s", pair["id"])
        row["error"] = f"retrieval: {type(e).__name__}: {e}"
        return row
    row["retrieval_latency_ms"] = round((time.time() - t0) * 1000, 1)
    row["top_score"] = (
        round(float(chunks[0][top_score_field]), 6)
        if chunks and top_score_field in chunks[0]
        else 0.0
    )
    row["retrieved_doc_ids"] = list(dict.fromkeys(c["doc_id"] for c in chunks))
    row["retrieved_chunk_types"] = [c.get("chunk_type", "prose") for c in chunks]

    # 2. Retrieval metrics (positives) / rejection flag (negatives)
    if pair["query_type"] == "negative":
        # No standard threshold across all four score scales — record top_score
        # for the calibration step (2b). Do NOT auto-flag correctly_rejected here.
        row["correctly_rejected"] = None
    else:
        try:
            row.update(evaluate_query(chunks, pair["expected_sources"], ks=KS))
        except Exception as e:
            logger.exception("Retrieval metrics failed for %s", pair["id"])
            row["retrieval_metrics_error"] = f"{type(e).__name__}: {e}"

    # 3. Synthesis (hold constant across configs)
    try:
        synth_result = synth.synthesise(pair["question"], chunks)
    except Exception as e:
        logger.exception("Synthesis failed for %s", pair["id"])
        row["error"] = f"synthesis: {type(e).__name__}: {e}"
        return row

    brief = synth_result["brief"]
    row["synthesis_latency_ms"] = round(synth_result["latency_ms"], 1)
    row["synthesis_cost_usd"] = synth_result["cost_usd"]
    row["generated_answer"] = brief.answer
    row["generated_citation_count"] = len(brief.citations)

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

    scores = judge_result.get("scores")
    if scores is None:
        row["judge_refusal_reason"] = judge_result.get("refusal_reason")
    else:
        row["correctness"] = scores.correctness
        row["faithfulness"] = scores.faithfulness
        row["completeness"] = scores.completeness
        row["refusal_appropriateness"] = scores.refusal_appropriateness

    return row


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _load_ground_truth() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit(
            f"Ground truth not found at {GROUND_TRUTH_PATH}. "
            "Run scripts/migrate_ground_truth.py first."
        )
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


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
    Per-config negatives summary. Uses judge refusal_appropriateness as the
    ground-truth for whether the system handled the negative correctly.
    For negatives, refusal_appropriateness=5 means "correctly refused";
    <3 means "fabricated / did not refuse when it should have".
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


def _aggregate_config(per_query: list[dict]) -> dict:
    scored = [r for r in per_query if "correctness" in r]
    return {
        "n_scored": len(scored),
        "overall_judge": aggregate_metrics(scored),
        "by_query_type": _aggregate_by_slice(scored, "query_type"),
        "by_probe": _aggregate_by_slice(per_query, "probe"),
        "negatives": _negatives_summary(per_query),
        "retrieval_latency_ms_mean": round(
            sum(r.get("retrieval_latency_ms", 0) for r in per_query) / max(len(per_query), 1), 1
        ),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(output_path: Path | None = None) -> Path:
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ground_truth = _load_ground_truth()
    logger.info("Loaded %d ground truth pairs", len(ground_truth))

    configs = _build_configs()

    # Synthesiser + Judge — held constant across configs
    from synthesis import Synthesiser
    synth = Synthesiser()
    from evaluation.judge import LLMJudge
    judge = LLMJudge()
    logger.info("Synthesiser=%s  Judge=%s", synth.model, judge.model)

    t_run_all = time.time()
    per_config: dict[str, dict] = {}

    for name, cfg in configs.items():
        logger.info("=" * 78)
        logger.info("CONFIG: %s", name)
        logger.info("=" * 78)
        t_cfg = time.time()
        per_query: list[dict] = []
        for i, pair in enumerate(ground_truth, 1):
            per_query.append(
                _score_pair(pair, cfg["retrieve"], cfg["top_score_field"], synth, judge)
            )
            if i % 5 == 0:
                logger.info("  %s: %d/%d (%.0fs elapsed)",
                            name, i, len(ground_truth), time.time() - t_cfg)

        per_config[name] = _aggregate_config(per_query)
        per_config[name]["config_wall_time_sec"] = round(time.time() - t_cfg, 1)
        per_config[name]["top_score_field"] = cfg["top_score_field"]

    total_wall_time_sec = time.time() - t_run_all

    total_synth_cost = sum(
        sum(r.get("synthesis_cost_usd", 0.0) or 0.0 for r in cfg["per_query"])
        for cfg in per_config.values()
    )
    total_judge_cost = sum(
        sum(r.get("judge_cost_usd", 0.0) or 0.0 for r in cfg["per_query"])
        for cfg in per_config.values()
    )

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(ground_truth),
        "n_configs": len(per_config),
        "wall_time_sec": round(total_wall_time_sec, 1),
        "config_pipeline": {
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "rerank_candidate_k": RERANK_CANDIDATE_K,
            "synthesis_model": synth.model,
            "judge_model": judge.model,
        },
        "cost_summary": {
            "synthesis_total_usd": round(total_synth_cost, 6),
            "judge_total_usd": round(total_judge_cost, 6),
            "total_usd": round(total_synth_cost + total_judge_cost, 6),
        },
        "per_config": per_config,
    }

    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"ablation_{ts}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Wrote ablation results to %s", output_path)

    # ── Console summary ──────────────────────────────────────────────────
    print()
    print("=" * 90)
    print(f"4-CONFIG ABLATION — {len(ground_truth)} queries × {len(per_config)} configs")
    print("=" * 90)
    print(f"  Wall time: {total_wall_time_sec:.0f}s")
    print(f"  Total cost: ${results['cost_summary']['total_usd']:.4f}")
    print()

    # Comparison table
    metrics = ("correctness", "faithfulness", "completeness", "refusal_appropriateness")
    print(f"{'config':16s} " + " ".join(f"{m[:9]:>10s}" for m in metrics) +
          f" {'ret_ms':>8s} {'neg_ok':>8s}")
    for name, cfg in per_config.items():
        row = f"{name:16s} "
        for m in metrics:
            v = cfg["overall_judge"].get(m)
            row += f" {(v if v is not None else 0):>9.2f} "
        row += f" {cfg['retrieval_latency_ms_mean']:>7.0f} "
        row += f" {cfg['negatives'].get('handled_well_rate', 0):>7.2%}"
        print(row)

    return output_path


if __name__ == "__main__":
    run()
