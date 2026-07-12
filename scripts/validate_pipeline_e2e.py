"""
Week 3 Step 3 — v1 hybrid retrieval pipeline validation.

Runs the 3 CLAUDE.md pilot queries through the v1 pipeline
(built by scripts/build_indices.py) end-to-end:

  BM25 + Dense → RRF fusion → top-5   (no reranker in v1)

Reranker was dropped from v1 after ablation showed no hit-rate gain on the 3
pilot queries. See CLAUDE.md v2 roadmap: reranker re-evaluated in Week 5 with
full ground truth dataset + LLM-as-judge.

Run:
  uv run python scripts/validate_pipeline_e2e.py
"""

import logging
import time

from retrieval import BM25Retriever, DenseRetriever, HybridRetriever

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


def main() -> None:
    print("Loading indices...")
    bm25 = BM25Retriever.load("data/processed/bm25_index.pkl")
    dense = DenseRetriever(persist_dir="data/processed/chroma_db")
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)

    all_pass = True
    for i, q in enumerate(QUERIES, 1):
        print()
        print("=" * 80)
        print(f"Query {i}: {q['query']}")
        print(f"Expected: {q['expected_doc']}")
        print("-" * 80)

        t0 = time.time()
        top5 = hybrid.retrieve(q["query"], top_k=5)
        latency_ms = (time.time() - t0) * 1000

        top_docs = {c["doc_id"] for c in top5}
        passed = q["expected_doc"] in top_docs
        all_pass = all_pass and passed
        top1_hit = "top-1" if top5 and top5[0]["doc_id"] == q["expected_doc"] else "in top-5"

        print(f"{'PASS' if passed else 'FAIL'} — expected doc {top1_hit}, "
              f"retrieval latency {latency_ms:.0f}ms")
        print()
        for c in top5:
            marker = "*" if c["doc_id"] == q["expected_doc"] else " "
            print(f"  {marker} [rrf={c['rrf_score']:.5f}] {c['doc_id']} p{c['page_number']}")
            preview = c["text"].replace("\n", " ")[:110]
            print(f"       {preview!r}")

    print()
    print("=" * 80)
    print(f"OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
