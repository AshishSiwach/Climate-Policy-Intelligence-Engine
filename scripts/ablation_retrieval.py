"""
Retrieval ablation — 3 pilot queries × 4 configurations.

Configurations:
  1. BM25 only
  2. Dense only
  3. Hybrid (BM25 + Dense + RRF, no rerank)
  4. Full pipeline (hybrid + cross-encoder rerank)

Metrics per query per config:
  - top-1 hit (expected doc appears at rank 1)
  - top-5 hit (expected doc appears anywhere in top 5)
  - latency (retrieval + rerank if applicable)
  - top-5 doc_ids (for cross-config comparison)

Run:
  uv run python scripts/ablation_retrieval.py
"""

import logging
import time

from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker

logging.basicConfig(level=logging.WARNING)

QUERIES = [
    {
        "query": "What load control licensing requirements does Ofgem propose?",
        "expected_doc": "OFGEM_SMART_SECURE_2024",
    },
    {
        "query": "What aggregate losses did UK banks face under the CBES early action scenario?",
        "expected_doc": "BOE_CBES_RESULTS_2021",
    },
    {
        "query": "What does the IEA project for peak fossil fuel demand?",
        "expected_doc": "IEA_WEO_2025",
    },
]


def run_config(name: str, retrieve_fn, query: str, expected_doc: str) -> dict:
    """Time a retrieval function, return top-5 hit stats."""
    t0 = time.time()
    results = retrieve_fn(query)
    latency_ms = (time.time() - t0) * 1000

    top5 = results[:5]
    top5_docs = [r["doc_id"] for r in top5]
    top1_hit = bool(top5) and top5[0]["doc_id"] == expected_doc
    top5_hit = expected_doc in top5_docs

    return {
        "config": name,
        "latency_ms": latency_ms,
        "top1_hit": top1_hit,
        "top5_hit": top5_hit,
        "top5_docs": top5_docs,
        "top1_page": top5[0]["page_number"] if top5 else None,
    }


def main() -> None:
    print("Loading indices...")
    bm25 = BM25Retriever.load("data/processed/bm25_index.pkl")
    dense = DenseRetriever(persist_dir="data/processed/chroma_db")
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
    reranker = Reranker(top_k=5)

    # Warm up reranker (lazy-loaded) so first-call latency doesn't skew results
    print("Warming up reranker...")
    warmup_candidates = hybrid.retrieve("warmup query about climate policy", top_k=20)
    _ = reranker.rerank("warmup query about climate policy", warmup_candidates)

    all_results = []

    for i, q in enumerate(QUERIES, 1):
        print()
        print("=" * 90)
        print(f"Query {i}: {q['query']}")
        print(f"Expected: {q['expected_doc']}")
        print("-" * 90)

        configs = [
            ("BM25 only", lambda t: bm25.query(t, top_k=5)),
            ("Dense only", lambda t: dense.query(t, top_k=5)),
            ("Hybrid (no rerank)", lambda t: hybrid.retrieve(t, top_k=5)),
            (
                "Full pipeline (hybrid + rerank)",
                lambda t: reranker.rerank(t, hybrid.retrieve(t, top_k=20)),
            ),
        ]

        query_results = []
        for name, fn in configs:
            r = run_config(name, fn, q["query"], q["expected_doc"])
            query_results.append(r)

        print(f"  {'Config':<34} {'Latency':>10} {'Top-1':>7} {'Top-5':>7}  Top-5 doc_ids (unique)")
        print(f"  {'-'*34} {'-'*10} {'-'*7} {'-'*7}  {'-'*40}")
        for r in query_results:
            unique_docs = list(dict.fromkeys(r["top5_docs"]))
            marker1 = "PASS" if r["top1_hit"] else "----"
            marker5 = "PASS" if r["top5_hit"] else "----"
            print(f"  {r['config']:<34} {r['latency_ms']:>8.0f}ms {marker1:>7} {marker5:>7}  {unique_docs}")

        all_results.append({"query": q["query"], "results": query_results})

    # ─── Summary ─────────────────────────────────────────────────────────
    print()
    print("=" * 90)
    print("SUMMARY — hit rate across 3 pilot queries")
    print("=" * 90)
    config_names = [c[0] for c in configs]
    print(f"  {'Config':<34} {'Top-1 hits':>12} {'Top-5 hits':>12} {'Avg latency':>13}")
    print(f"  {'-'*34} {'-'*12} {'-'*12} {'-'*13}")
    for name in config_names:
        top1 = sum(1 for q in all_results for r in q["results"] if r["config"] == name and r["top1_hit"])
        top5 = sum(1 for q in all_results for r in q["results"] if r["config"] == name and r["top5_hit"])
        avg_lat = sum(r["latency_ms"] for q in all_results for r in q["results"] if r["config"] == name) / 3
        print(f"  {name:<34} {top1:>4}/3       {top5:>4}/3       {avg_lat:>10.0f}ms")

    print()
    print("Interpretation guide:")
    print("  - Equal top-5 hit rate → simpler config wins (drop the extra layer)")
    print("  - Rerank changes top-1 → reranker is earning its ~200ms")
    print("  - Rerank doesn't change ordering → drop rerank for v1, revisit Week 5 with ground truth")


if __name__ == "__main__":
    main()
