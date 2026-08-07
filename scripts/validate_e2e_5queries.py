"""
Week 3 Step 5 — 5-query manual validation of the full pipeline.

Runs the 5 queries specified in CLAUDE.md through main.py's pipeline
and prints answer, citations, and latency breakdown for manual inspection.

Confirms: the E2E pipeline works for factual, cross-document, and
out-of-corpus queries before Week 3 course submission.

Run:
  uv run python scripts/validate_e2e_5queries.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from main import build_pipeline, run_query
from monitoring import QueryLogger

logging.basicConfig(level=logging.WARNING)

QUERIES = [
    {
        "label": "Factual — Ofgem licensing timeline",
        "query": "What are Ofgem's proposed timelines for load control licence applications?",
    },
    {
        "label": "Factual — CBES loss estimates",
        "query": "What aggregate losses did UK banks face under the CBES early action scenario?",
    },
    {
        "label": "Factual — IEA fossil fuel demand peak",
        "query": "What does the IEA project for the peak of global fossil fuel demand?",
    },
    {
        "label": "Cross-document — CCC vs IEA on net zero pathways",
        "query": "How do the CCC and the IEA differ in their views on the pace of UK net zero delivery?",
    },
    {
        "label": "Out-of-corpus negative",
        "query": "What is the best sourdough starter recipe for beginners?",
    },
]


def main() -> None:
    load_dotenv()
    hybrid, synth = build_pipeline()
    qlogger = QueryLogger(log_path="logs/queries.jsonl")

    for i, q in enumerate(QUERIES, 1):
        print()
        print("=" * 90)
        print(f"Query {i} — {q['label']}")
        print(f"  {q['query']}")
        print("-" * 90)

        brief = run_query(q["query"], hybrid, synth, qlogger)

        if "error" in brief:
            print(f"  FAILURE: {brief['error']}")
            continue

        answer = brief["answer"]
        print(f"  Answer ({len(answer)} chars):")
        print(f"    {answer[:400]}{'...' if len(answer) > 400 else ''}")
        print()
        print(f"  Citations ({len(brief['citations'])}):")
        for c in brief["citations"]:
            print(f"    - {c['doc_id']} p{c['page']}: {c['passage'][:90]!r}")
        if brief["contradictions"]:
            print(f"  Contradictions ({len(brief['contradictions'])}):")
            for c in brief["contradictions"]:
                print(f"    - {c['doc_a']} vs {c['doc_b']}: {c['summary']}")

    print()
    print("=" * 90)
    print(f"All 5 records logged to logs/queries.jsonl")


if __name__ == "__main__":
    main()
