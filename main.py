"""
CPIE — CLI entry point.

Usage:
  uv run python main.py "<query>"
  uv run python main.py --top-k 8 "<query>"

Runs the full v1 pipeline:
  query → BM25 + Dense → RRF fusion → top-5 → GPT-4o mini synthesis → JSON brief

Logs one record per query to logs/queries.jsonl (started day one per CLAUDE.md).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from monitoring import QueryLogger, build_query_record
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever
from synthesis import AnalystBrief, Synthesiser

logger = logging.getLogger("cpie.main")

BM25_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_DIR = Path("data/processed/chroma_db")
LOG_PATH = Path("logs/queries.jsonl")

# --- v1 input guardrails (see CLAUDE.md Locked Decisions) -----------------
MAX_QUERY_CHARS = 500          # cost-blow-up defense
DAILY_COST_LIMIT_USD = 5.00    # daily API-spend circuit breaker

QUERY_TOO_LONG_MSG = (
    f"Query exceeds the {MAX_QUERY_CHARS}-character limit. "
    "Rephrase your question more concisely."
)
COST_LIMIT_MSG = (
    "The daily cost limit for this deployment has been reached. "
    "Try again after 00:00 UTC."
)


def _canonical_refusal(msg: str) -> AnalystBrief:
    """Uniform refusal shape so guardrail-triggered responses look like every other brief."""
    return AnalystBrief(answer=msg, citations=[], confidence=0.0, contradictions=[])


def _daily_cost_so_far(log_path: Path) -> float:
    """Sum today's `cost_usd` values from the JSONL log. Cheap enough for v1 traffic."""
    if not log_path.exists():
        return 0.0
    today = date.today().isoformat()
    total = 0.0
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("timestamp", "").startswith(today):
                total += float(rec.get("cost_usd", 0.0) or 0.0)
    return total


def build_pipeline() -> tuple[HybridRetriever, Synthesiser]:
    """Load indices + retriever + synthesiser. One-time setup per CLI invocation."""
    if not BM25_PATH.exists():
        raise SystemExit(
            f"BM25 index not found at {BM25_PATH}. Run: uv run python scripts/build_indices.py"
        )
    if not CHROMA_DIR.exists():
        raise SystemExit(
            f"Chroma index not found at {CHROMA_DIR}. Run: uv run python scripts/build_indices.py"
        )

    bm25 = BM25Retriever.load(BM25_PATH)
    dense = DenseRetriever(persist_dir=CHROMA_DIR)
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
    synth = Synthesiser()
    return hybrid, synth


def run_query(
    query: str,
    hybrid: HybridRetriever,
    synth: Synthesiser,
    qlogger: QueryLogger,
    top_k: int = 5,
    log_path: Path = LOG_PATH,
) -> dict:
    """Run the full pipeline for one query, log the record, return the brief as dict.

    Applies v1 input guardrails BEFORE hitting retrieval/synthesis:
      - query length limit (cost blow-up defense)
      - daily cost circuit breaker (spend cap)
    Both refusals still write a log record (with distinct failure_reason).
    """
    # Guardrail 1 — query length limit
    if len(query) > MAX_QUERY_CHARS:
        brief = _canonical_refusal(QUERY_TOO_LONG_MSG)
        qlogger.log(build_query_record(
            query=query[:MAX_QUERY_CHARS] + "...(truncated for log)",
            retrieved_chunks=[],
            retrieval_latency_ms=0.0,
            synthesis_result={
                "brief": brief, "latency_ms": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "cost_usd": 0.0,
                "confidence_signals": {"score_signal": 0.0, "agreement_signal": 0.0,
                                       "margin_signal": 0.0, "citation_signal": 0.0,
                                       "out_of_corpus": False, "llm_refusal": False},
            },
            model_used=synth.model,
            failure_reason=f"guardrail: query_too_long ({len(query)} chars)",
        ))
        return brief.model_dump()

    # Guardrail 2 — daily cost circuit breaker
    daily_cost = _daily_cost_so_far(log_path)
    if daily_cost >= DAILY_COST_LIMIT_USD:
        brief = _canonical_refusal(COST_LIMIT_MSG)
        qlogger.log(build_query_record(
            query=query,
            retrieved_chunks=[],
            retrieval_latency_ms=0.0,
            synthesis_result={
                "brief": brief, "latency_ms": 0.0, "prompt_tokens": 0,
                "completion_tokens": 0, "cost_usd": 0.0,
                "confidence_signals": {"score_signal": 0.0, "agreement_signal": 0.0,
                                       "margin_signal": 0.0, "citation_signal": 0.0,
                                       "out_of_corpus": False, "llm_refusal": False},
            },
            model_used=synth.model,
            failure_reason=f"guardrail: daily_cost_limit (${daily_cost:.4f} spent)",
        ))
        return brief.model_dump()

    # Normal pipeline
    failure_reason: str | None = None
    synthesis_result = None
    retrieval_latency_ms = 0.0
    chunks: list[dict] = []

    try:
        t0 = time.time()
        chunks = hybrid.retrieve(query, top_k=top_k)
        retrieval_latency_ms = (time.time() - t0) * 1000

        synthesis_result = synth.synthesise(query, chunks)
    except Exception as e:
        failure_reason = f"{type(e).__name__}: {e}"
        logger.exception("Pipeline failure for query: %r", query)

    record = build_query_record(
        query=query,
        retrieved_chunks=chunks,
        retrieval_latency_ms=retrieval_latency_ms,
        synthesis_result=synthesis_result,
        model_used=synth.model,
        failure_reason=failure_reason,
    )
    qlogger.log(record)

    if synthesis_result is None:
        return {"error": failure_reason}

    return synthesis_result["brief"].model_dump()


def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")

    parser = argparse.ArgumentParser(description="CPIE — Climate Policy Intelligence Engine")
    parser.add_argument("query", help="Natural-language question to run against the corpus")
    parser.add_argument("--top-k", type=int, default=5, help="Chunks to pass to synthesiser (default 5)")
    parser.add_argument("--log-path", type=Path, default=LOG_PATH, help="Path for JSONL query log")
    args = parser.parse_args()

    hybrid, synth = build_pipeline()
    qlogger = QueryLogger(log_path=args.log_path)

    brief = run_query(args.query, hybrid, synth, qlogger, top_k=args.top_k, log_path=args.log_path)
    print(json.dumps(brief, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
