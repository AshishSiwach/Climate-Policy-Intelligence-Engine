"""
Cross-doc A/B evaluation — agent path vs fast path.

Reads data/eval/ground_truth.json, filters to cross-doc queries,
runs both paths, writes results to data/eval/ab_results_crossdoc.jsonl.

Usage:
    uv run python scripts/eval_crossdoc_ab.py --limit 5 --dry-run

Options:
    --dry-run   Print queries without calling APIs.
    --limit N   Run first N cross-doc queries only.

Output record (one JSON line per query):
    {
      "id": str,
      "query": str,
      "fast_answer": str,
      "agent_answer": str,
      "fast_citations": int,          # count
      "agent_citations": int,
      "fast_cost": float,
      "agent_cost": float,
      "fast_latency_s": float,
      "agent_latency_s": float,
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

GROUND_TRUTH_PATH = _REPO_ROOT / "data" / "eval" / "ground_truth.json"
OUTPUT_PATH = _REPO_ROOT / "data" / "eval" / "ab_results_crossdoc.jsonl"

# Cross-doc query_type values accepted from the ground truth file
_CROSS_DOC_TAGS = {"cross_doc", "cross_document"}


def _load_cross_doc_queries(limit: int | None) -> list[dict]:
    """Load cross-doc queries from the ground truth file.

    Falls back to all queries if none are tagged as cross-doc.
    """
    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        data: list[dict] = json.load(f)

    # Filter to cross-doc queries
    cross_doc = [q for q in data if q.get("query_type") in _CROSS_DOC_TAGS]

    if not cross_doc:
        # Fall back to all queries if no cross-doc tag exists
        cross_doc = data

    if limit is not None:
        cross_doc = cross_doc[:limit]

    return cross_doc


def _run_fast_path(query: str) -> dict:
    """Run the fast-path synthesiser and return timing + result."""
    from synthesis.synthesiser import Synthesiser
    from retrieval.hybrid_retriever import HybridRetriever  # type: ignore[import]

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
    }


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


def _run_dry(queries: list[dict]) -> None:
    """Print queries without calling any API."""
    print(f"DRY RUN — {len(queries)} cross-doc quer{'y' if len(queries) == 1 else 'ies'}:")
    for i, q in enumerate(queries, 1):
        print(f"  [{i}] {q.get('id', '?')}: {q.get('question', q.get('query', ''))[:80]}")


def _run_ab(queries: list[dict]) -> None:
    """Run the A/B evaluation and write results to JSONL."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, q in enumerate(queries, 1):
            query_text = q.get("question") or q.get("query", "")
            query_id = q.get("id", f"q_{i}")

            print(f"[{i}/{len(queries)}] Running: {query_id!r} ...")

            try:
                fast = _run_fast_path(query_text)
            except Exception as exc:
                fast = {
                    "fast_answer": f"ERROR: {exc}",
                    "fast_citations": 0,
                    "fast_cost": 0.0,
                    "fast_latency_s": 0.0,
                }

            try:
                agent = _run_agent_path(query_text)
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

            record = {
                "id": query_id,
                "query": query_text,
                **fast,
                **agent,
            }

            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"  fast={fast['fast_latency_s']:.2f}s  "
                f"agent={agent['agent_latency_s']:.2f}s  "
                f"reason={agent['agent_termination_reason']}"
            )

    print(f"\nResults written to {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-doc A/B evaluation: agent vs fast path.")
    parser.add_argument("--limit", type=int, default=None, help="Run first N queries only.")
    parser.add_argument("--dry-run", action="store_true", help="Print queries without calling APIs.")
    args = parser.parse_args()

    queries = _load_cross_doc_queries(limit=args.limit)

    if not queries:
        print("No cross-doc queries found in ground_truth.json.")
        sys.exit(0)

    if args.dry_run:
        _run_dry(queries)
    else:
        _run_ab(queries)


if __name__ == "__main__":
    main()
