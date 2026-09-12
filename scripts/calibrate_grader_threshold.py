"""
Calibrate the cross-encoder coverage thresholds for the grader node.

Runs the existing ten labelled probes through the same hybrid retriever used
by the agent route, then prints the maximum cross-encoder score per query.
This keeps threshold calibration aligned with production retrieval rather
than calibrating against dense-only Chroma results.

Usage:
    python scripts/calibrate_grader_threshold.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# torch must load before numpy on Windows; see the matching note in main.py.
import torch  # noqa: F401

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

from sentence_transformers import CrossEncoder  # noqa: E402

from retrieval import BM25Retriever, DenseRetriever, HybridRetriever  # noqa: E402
from src.agent.nodes.grader import COVERED_THRESHOLD, PARTIAL_THRESHOLD  # noqa: E402
from src.agent.policies import RETRIEVER_TOP_K  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BM25_PATH = _REPO_ROOT / "data" / "processed" / "bm25_index.pkl"
CHROMA_PATH = _REPO_ROOT / "data" / "processed" / "chroma_db"
CE_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = RETRIEVER_TOP_K

# Queries grouped by expected coverage level
QUERIES: dict[str, list[str]] = {
    "SHOULD_BE_COVERED": [
        "What is the UK's net-zero target year?",
        "What are the main objectives of the UK Climate Change Act?",
        "What emission reduction commitments did countries make under the Paris Agreement?",
        "What sectors does the UK's carbon budget cover?",
    ],
    "LIKELY_PARTIAL": [
        "How do the UK Climate Change Act targets compare to the Paris Agreement commitments?",
        "Compare the net-zero strategies of the UK and EU — where do they align and diverge?",
        "What role do financial institutions play in climate policy?",
    ],
    "LIKELY_NOT_COVERED": [
        "What is the GDP of France in 2023?",
        "Who won the 2024 US presidential election?",
        "What are the latest iPhone features?",
    ],
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_hybrid_retriever() -> HybridRetriever:
    """Build the production BM25+dense RRF retriever."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found: {BM25_PATH}")
    if not CHROMA_PATH.exists():
        raise FileNotFoundError(f"Chroma index not found: {CHROMA_PATH}")

    bm25 = BM25Retriever.load(BM25_PATH)
    dense = DenseRetriever(persist_dir=CHROMA_PATH)
    dense.warm_up()
    return HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)


def retrieve(retriever: HybridRetriever, query: str, top_k: int) -> list[dict]:
    """Retrieve passages exactly as an agent sub-question does."""
    return retriever.retrieve(query, top_k=top_k)


def classify_score(score: float) -> str:
    """Apply the grader's current production thresholds."""
    if score >= COVERED_THRESHOLD:
        return "covered"
    if score >= PARTIAL_THRESHOLD:
        return "partial"
    return "not_covered"


def main():
    print(f"Loading production hybrid retriever (BM25 + dense RRF), top_k={TOP_K}")
    retriever = build_hybrid_retriever()

    print(f"Loading cross-encoder: {CE_MODEL}")
    encoder = CrossEncoder(CE_MODEL)
    print()

    all_scores: dict[str, list[float]] = {"SHOULD_BE_COVERED": [], "LIKELY_PARTIAL": [], "LIKELY_NOT_COVERED": []}
    classifications: dict[str, list[str]] = {group: [] for group in QUERIES}

    for group, queries in QUERIES.items():
        print(f"{'='*60}")
        print(f"GROUP: {group}")
        print(f"{'='*60}")

        for query in queries:
            chunks = retrieve(retriever, query, TOP_K)
            passages = [(chunk.get("text") or chunk.get("passage") or "").strip() for chunk in chunks]
            passages = [passage for passage in passages if passage]
            if not passages:
                print(f"  [NO PASSAGES] {query!r}")
                continue

            pairs = [(query, passage) for passage in passages]
            scores = encoder.predict(pairs)
            max_score = float(max(scores))
            all_scores[group].append(max_score)
            status = classify_score(max_score)
            classifications[group].append(status)

            print(f"  max={max_score:+6.2f}  status={status:<11}  n_chunks={len(chunks)}")
            print(f"  query: {query!r}")
            print(f"  scores: {[f'{s:+.2f}' for s in scores]}")
            print()

    # Summary + threshold recommendation
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for group, scores in all_scores.items():
        if scores:
            print(f"  {group}: min={min(scores):+.2f}  mean={sum(scores)/len(scores):+.2f}  max={max(scores):+.2f}")

    print(
        f"\nCURRENT THRESHOLDS: covered >= {COVERED_THRESHOLD:+.2f}, "
        f"partial >= {PARTIAL_THRESHOLD:+.2f}, otherwise not_covered"
    )
    for group, statuses in classifications.items():
        counts = {status: statuses.count(status) for status in ("covered", "partial", "not_covered")}
        print(
            f"  {group}: covered={counts['covered']}  partial={counts['partial']}  "
            f"not_covered={counts['not_covered']}"
        )

    covered_scores = all_scores["SHOULD_BE_COVERED"]
    ooc_scores = all_scores["LIKELY_NOT_COVERED"]

    if covered_scores and ooc_scores:
        import numpy as np
        from scipy import stats

        c = np.array(covered_scores)
        oc = np.array(ooc_scores)

        # t-based 95% lower bound of covered distribution
        # mean - t(df, 0.975) * se  where se = std / sqrt(n)
        t_c = stats.t.ppf(0.975, df=len(c) - 1)
        se_c = c.std(ddof=1) / len(c) ** 0.5
        suggested_covered = round(float(c.mean() - t_c * se_c), 2)

        # t-based 95% upper bound of not_covered distribution
        t_oc = stats.t.ppf(0.975, df=len(oc) - 1)
        se_oc = oc.std(ddof=1) / len(oc) ** 0.5
        suggested_partial = round(float(oc.mean() + t_oc * se_oc), 2)

        print("\nSTATISTICS")
        print(f"  covered      n={len(c)}  mean={c.mean():+.2f}  std={c.std(ddof=1):.2f}  t={t_c:.2f}  se={se_c:.2f}")
        print(
            f"  not_covered  n={len(oc)}  mean={oc.mean():+.2f}  "
            f"std={oc.std(ddof=1):.2f}  t={t_oc:.2f}  se={se_oc:.2f}"
        )
        print(
            f"\n  covered lower bound = {c.mean():+.2f} - "
            f"{t_c:.2f}*{se_c:.2f} = {suggested_covered}"
        )
        print(
            f"  OOC upper bound     = {oc.mean():+.2f} + "
            f"{t_oc:.2f}*{se_oc:.2f} = {suggested_partial}"
        )

        if suggested_partial >= suggested_covered:
            print("\nNO THRESHOLD CHANGE RECOMMENDED")
            print(
                "  The confidence bounds overlap or are reversed, so this probe set "
                "cannot produce an ordered three-class threshold."
            )
        else:
            print("\nSUGGESTED THRESHOLDS for grader.py:")
            print(f"  COVERED_THRESHOLD = {suggested_covered}")
            print(f"  PARTIAL_THRESHOLD = {suggested_partial}")

        print("\nNote: t-distribution used (not z=2) because n < 30.")
        print("Add more probe queries to each group to tighten the confidence interval.")


if __name__ == "__main__":
    main()
