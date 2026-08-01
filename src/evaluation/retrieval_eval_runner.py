"""
Retrieval evaluation runner.

Loads data/eval/ground_truth.json, runs each query through the HybridRetriever,
computes per-query metrics + slices, and writes results to
data/eval/results/retrieval_metrics_<timestamp>.json.

Handles negatives via the pipeline's out-of-corpus RRF threshold — a
"correctly_rejected" flag rather than the doc-level IR metrics.

Run:
  uv run python -m evaluation.retrieval_eval_runner
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.retrieval_metrics import aggregate_metrics, evaluate_query
from retrieval import BM25Retriever, HybridRetriever

# NOTE: synthesis.synthesiser is deliberately NOT imported at module load.
# On Windows, importing it before DenseRetriever causes a downstream torch
# import to hang. Imported inside functions that need it instead.

logger = logging.getLogger(__name__)

GROUND_TRUTH_PATH = Path("data/eval/ground_truth.json")
BM25_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_DIR = Path("data/processed/chroma_db")
RESULTS_DIR = Path("data/eval/results")

RETRIEVAL_TOP_K = 10   # max k we compute metrics at; pipeline default is 5
KS = (5, 10)           # cutoffs for metric computation

_PROBE_PATTERN = re.compile(r"\[PROBE:\s*(\w+)\]")


def _load_ground_truth() -> list[dict]:
    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit(
            f"Ground truth not found at {GROUND_TRUTH_PATH}. "
            "Run scripts/migrate_ground_truth.py first."
        )
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        return json.load(f)


def _build_hybrid() -> HybridRetriever:
    from retrieval import DenseRetriever   # lazy — heavy import
    if not BM25_PATH.exists() or not CHROMA_DIR.exists():
        raise SystemExit(
            "Indices missing. Run: uv run python scripts/build_indices.py"
        )
    bm25 = BM25Retriever.load(BM25_PATH)
    dense = DenseRetriever(persist_dir=CHROMA_DIR)
    return HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)


def _extract_probe(notes: str | None) -> str | None:
    if not notes:
        return None
    m = _PROBE_PATTERN.search(notes)
    return m.group(1) if m else None


def _score_positive(
    pair: dict,
    hybrid: HybridRetriever,
) -> dict:
    """Run pipeline + compute IR metrics for a non-negative query."""
    t0 = time.time()
    chunks = hybrid.retrieve(pair["question"], top_k=RETRIEVAL_TOP_K)
    latency_ms = (time.time() - t0) * 1000

    metrics = evaluate_query(chunks, pair["expected_sources"], ks=KS)

    return {
        "id": pair["id"],
        "query_type": pair["query_type"],
        "probe": _extract_probe(pair.get("notes")),
        "retrieval_latency_ms": round(latency_ms, 1),
        "top_rrf": round(chunks[0]["rrf_score"], 6) if chunks else 0.0,
        "retrieved_doc_ids": [c["doc_id"] for c in chunks[:5]],
        **metrics,
    }


def _score_negative(
    pair: dict,
    hybrid: HybridRetriever,
    oor_threshold: float,
) -> dict:
    """Negatives — correct behaviour is out-of-corpus short-circuit or no retrieval."""
    t0 = time.time()
    chunks = hybrid.retrieve(pair["question"], top_k=RETRIEVAL_TOP_K)
    latency_ms = (time.time() - t0) * 1000

    top_rrf = chunks[0]["rrf_score"] if chunks else 0.0
    correctly_rejected = top_rrf < oor_threshold

    return {
        "id": pair["id"],
        "query_type": pair["query_type"],
        "probe": _extract_probe(pair.get("notes")),
        "retrieval_latency_ms": round(latency_ms, 1),
        "top_rrf": round(top_rrf, 6),
        "retrieved_doc_ids": [c["doc_id"] for c in chunks[:5]],
        "correctly_rejected": correctly_rejected,
    }


def _aggregate_by_slice(
    per_query: list[dict],
    slice_key: str,
) -> dict[str, dict[str, float]]:
    """Group per_query results by a key (e.g. query_type, probe) and aggregate each group."""
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


def _negatives_rejection_rate(per_query: list[dict]) -> dict:
    negs = [r for r in per_query if r.get("query_type") == "negative"]
    if not negs:
        return {"n": 0}
    n_rejected = sum(1 for r in negs if r.get("correctly_rejected"))
    return {
        "n": len(negs),
        "rejection_rate": n_rejected / len(negs),
        "n_correctly_rejected": n_rejected,
        "n_incorrectly_answered": len(negs) - n_rejected,
    }


def run(output_path: Path | None = None) -> Path:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ground_truth = _load_ground_truth()
    logger.info("Loaded %d ground truth pairs", len(ground_truth))

    hybrid = _build_hybrid()
    logger.info("Pipeline loaded. Loading OOR threshold from synthesis config...")

    # Deferred import — see module-level note.
    from synthesis.synthesiser import OUT_OF_CORPUS_RRF_THRESHOLD

    logger.info("Scoring...")
    per_query: list[dict] = []
    for i, pair in enumerate(ground_truth, 1):
        try:
            if pair["query_type"] == "negative":
                result = _score_negative(pair, hybrid, OUT_OF_CORPUS_RRF_THRESHOLD)
            else:
                result = _score_positive(pair, hybrid)
            per_query.append(result)
        except Exception as e:
            logger.exception("Failed to score pair %s", pair.get("id"))
            per_query.append({
                "id": pair.get("id"),
                "query_type": pair.get("query_type"),
                "error": f"{type(e).__name__}: {e}",
            })
        if i % 10 == 0:
            logger.info("Scored %d/%d", i, len(ground_truth))

    positives = [r for r in per_query if r.get("query_type") != "negative" and "error" not in r]

    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_queries": len(ground_truth),
        "n_positives_scored": len(positives),
        "config": {
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "ks": list(KS),
            "rrf_k": 60,
            "out_of_corpus_threshold": OUT_OF_CORPUS_RRF_THRESHOLD,
        },   # OUT_OF_CORPUS_RRF_THRESHOLD imported above via deferred import
        "overall": aggregate_metrics(positives),
        "by_query_type": _aggregate_by_slice(positives, "query_type"),
        "by_probe": _aggregate_by_slice(per_query, "probe"),
        "negatives": _negatives_rejection_rate(per_query),
        "per_query": per_query,
    }

    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = RESULTS_DIR / f"retrieval_metrics_{ts}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info("Wrote results to %s", output_path)

    # ── Console summary ──────────────────────────────────────────────────
    print()
    print("=" * 80)
    print(f"OVERALL ({len(positives)} positive queries)")
    print("=" * 80)
    for m, v in sorted(results["overall"].items()):
        print(f"  {m:<25} {v:.3f}")

    print()
    print("=" * 80)
    print("BY QUERY TYPE")
    print("=" * 80)
    for qtype, agg in results["by_query_type"].items():
        print(f"\n  {qtype} (n={agg['n']})")
        for m in ("recall@5", "mrr@5", "ndcg@5", "hit@5"):
            if m in agg:
                print(f"    {m:<25} {agg[m]:.3f}")

    print()
    print("=" * 80)
    print("BY PROBE (failure-pattern queries from Week 3)")
    print("=" * 80)
    for probe, agg in results["by_probe"].items():
        print(f"\n  {probe} (n={agg['n']})")
        for m in ("recall@5", "mrr@5"):
            if m in agg:
                print(f"    {m:<25} {agg[m]:.3f}")

    print()
    print("=" * 80)
    print(f"NEGATIVES (n={results['negatives']['n']})")
    print("=" * 80)
    if results["negatives"]["n"]:
        print(f"  Correctly rejected:    {results['negatives']['n_correctly_rejected']}/"
              f"{results['negatives']['n']}"
              f" ({results['negatives']['rejection_rate']:.0%})")
        print(f"  Incorrectly answered:  {results['negatives']['n_incorrectly_answered']}")

    return output_path


if __name__ == "__main__":
    run()
