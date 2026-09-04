"""
Baseline replay set freeze script — Phase 0 Baseline Hardening.

Reads ``data/eval/ground_truth.json`` (52-question eval set), runs each query
through the existing fast-path synthesiser, and records the results to
``data/eval/baseline_replay_set.jsonl``.

Usage (requires live corpus + OpenAI API key):
    uv run python scripts/freeze_baseline.py

DO NOT run in CI — it requires a live OpenAI API call and a populated corpus.
This script is written and committed but not executed during the build.

Output fields per JSONL record:
    query            — original question
    answer           — synthesiser answer
    citations        — list of chunk_ids cited
    cost_usd         — estimated OpenAI cost for this call
    latency_s        — wall-clock seconds for synthesis
    config_fingerprint — 8-char SHA-256 of settings at time of run
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure src/ is on path when running from project root
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("freeze_baseline")

GROUND_TRUTH_PATH = _REPO_ROOT / "data" / "eval" / "ground_truth.json"
OUTPUT_PATH = _REPO_ROOT / "data" / "eval" / "baseline_replay_set.jsonl"

# Number of retrieved chunks passed to the synthesiser
TOP_K = 5


def main() -> None:
    if not GROUND_TRUTH_PATH.exists():
        logger.error("Ground truth not found at %s — run scripts/migrate_ground_truth.py first", GROUND_TRUTH_PATH)
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set — cannot run live synthesis")
        sys.exit(1)

    # Lazy imports — only needed at runtime, not in CI
    from config.settings import settings_fingerprint
    from retrieval.hybrid_retriever import HybridRetriever
    from synthesis.synthesiser import Synthesiser

    cfg_fp = settings_fingerprint()
    logger.info("Config fingerprint: %s", cfg_fp)

    retriever = HybridRetriever()
    synth = Synthesiser(api_key=api_key)

    with open(GROUND_TRUTH_PATH, encoding="utf-8") as f:
        ground_truth = json.load(f)

    queries = [item["query"] for item in ground_truth]
    logger.info("Loaded %d queries from %s", len(queries), GROUND_TRUTH_PATH)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out_f:
        for i, query in enumerate(queries, 1):
            logger.info("[%d/%d] %s", i, len(queries), query[:80])
            try:
                chunks = retriever.retrieve(query, top_k=TOP_K)
                t0 = time.time()
                result = synth.synthesise(query, chunks)
                latency_s = time.time() - t0

                brief = result["brief"]
                record = {
                    "query": query,
                    "answer": brief.answer,
                    "citations": [c.doc_id for c in brief.citations],
                    "cited_chunk_ids": [getattr(c, "chunk_id", None) for c in brief.citations],
                    "cost_usd": result.get("cost_usd", 0.0),
                    "latency_s": round(latency_s, 3),
                    "config_fingerprint": cfg_fp,
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                written += 1
            except Exception as exc:
                logger.warning("Query %d failed: %s", i, exc)
                record = {
                    "query": query,
                    "answer": None,
                    "citations": [],
                    "cited_chunk_ids": [],
                    "cost_usd": 0.0,
                    "latency_s": 0.0,
                    "config_fingerprint": cfg_fp,
                    "error": str(exc),
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()

    logger.info("Wrote %d records to %s", written, OUTPUT_PATH)


if __name__ == "__main__":
    main()
