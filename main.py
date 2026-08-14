"""
CPIE — CLI entry point.

Usage:
  uv run python main.py "<query>"
  uv run python main.py --top-k 8 "<query>"

Runs the full pipeline:
  query → BM25 + Dense → RRF fusion → top-5 → GPT-5.4-mini synthesis → JSON brief

Logs one record per query to logs/queries.jsonl (started day one per CLAUDE.md).
"""

# isort: skip_file
# torch must be imported before numpy/openai/rank_bm25 on Windows to claim
# its OpenBLAS DLLs first. Reordering this import causes an access violation
# (exit code -1073741819) with no Python traceback. Do not let ruff sort it.
from __future__ import annotations

# Must import before numpy/rank_bm25/openai get a chance to (via the imports
# below). torch bundles its own OpenBLAS/MKL DLLs and has to claim them first
# on Windows — if plain numpy loads its copy first, the later sentence_transformers
# import inside build_pipeline() crashes the process with an access violation
# (exit code -1073741819) and no Python traceback.
import torch  # noqa: F401

import argparse
import json
import logging
import time
import uuid
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from monitoring import QueryLogger, build_query_record
from monitoring.db import insert_query_record as db_insert_query_record
from retrieval import BM25Retriever, HybridRetriever
from retrieval.institution_detector import detect_institutions
from synthesis import AnalystBrief, Synthesiser
from synthesis.query_classifier import classify_query
from synthesis.synthesiser import OUT_OF_CORPUS_ANSWER

# DenseRetriever imported lazily inside build_pipeline() — its dep
# (sentence_transformers/torch) is heavy and crashes some Windows setups
# during test collection. Tests never call build_pipeline; they mock the
# retriever directly, so the import cost only lands at real CLI startup.

logger = logging.getLogger("cpie.main")

BM25_PATH = Path("data/processed/bm25_index.pkl")
CHROMA_DIR = Path("data/processed/chroma_db")
LOG_PATH = Path("logs/queries.jsonl")

# Metadata filtering (Week 5). Detects named institutions
# in the query and pre-filters retrieval to those institutions before RRF.
# Directly fixes cross-doc coverage misses on institution-named queries (2c).
METADATA_FILTER_ENABLED = True

# Pre-retrieval domain gate (see src/synthesis/query_classifier.py).
# Classifies the query with GPT-4o-mini (~$0.00003) before spending retrieval
# + synthesis tokens (~$0.003). Out-of-domain queries are refused immediately.
# Fails-open: any classifier error passes the query through to the normal pipeline.
QUERY_CLASSIFIER_ENABLED = True

# --- Input guardrails (see CLAUDE.md Locked Decisions) --------------------
MAX_QUERY_CHARS = 500  # cost-blow-up defense
DAILY_COST_LIMIT_USD = 5.00  # daily API-spend circuit breaker

QUERY_TOO_LONG_MSG = f"Query exceeds the {MAX_QUERY_CHARS}-character limit. Rephrase your question more concisely."
COST_LIMIT_MSG = "The daily cost limit for this deployment has been reached. Try again after 00:00 UTC."


def _canonical_refusal(msg: str) -> AnalystBrief:
    """Uniform refusal shape so guardrail-triggered responses look like every other brief."""
    return AnalystBrief(answer=msg, citations=[], contradictions=[])


def _safe_db_write(record: dict) -> None:
    """
    Postgres dual-write wrapper. Belt-and-suspenders on db.insert_query_record
    which is already never-raise internally — this outer try/except catches the
    exotic failure modes that don't route through db.py's own try/except (e.g.
    monkeypatched-blowup in tests, module-import errors, pool construction
    exceptions before the DB call itself begins). Pipeline continues on any DB
    failure — JSONL is the primary sink, Postgres is secondary.
    """
    try:
        db_insert_query_record(record)
    except Exception as e:
        logger.warning("Postgres dual-write skipped due to: %s", e)


def _daily_cost_so_far(log_path: Path) -> float:
    """Sum today's `cost_usd` values from the JSONL log."""
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
        raise FileNotFoundError(f"BM25 index not found at {BM25_PATH}. Run: uv run python scripts/build_indices.py")
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma index not found at {CHROMA_DIR}. Run: uv run python scripts/build_indices.py")

    # Lazy import — see module-level note.
    from retrieval import DenseRetriever

    bm25 = BM25Retriever.load(BM25_PATH)
    dense = DenseRetriever(persist_dir=CHROMA_DIR)
    # Force model + Chroma load now so the first query doesn't silently pay
    # the ~15s cost. In app.py this runs inside @st.cache_resource, so the
    # "Loading indices + embedding model" spinner stays up during this call.
    dense.warm_up()
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

    Applies input guardrails BEFORE hitting retrieval/synthesis:
      - query length limit (cost blow-up defense)
      - daily cost circuit breaker (spend cap)
    Both refusals still write a log record (with distinct failure_reason).

    Every record is dual-written: JSONL (primary, always) + Postgres
    (secondary, never-raise). Same record dict feeds both sinks.
    """
    # Generate the query_id up front so it appears in JSONL, Postgres, and
    # (in 4d) the Streamlit UI feedback widget — all reference the same ID.
    query_id = str(uuid.uuid4())

    # Guardrail 1 — query length limit
    if len(query) > MAX_QUERY_CHARS:
        brief = _canonical_refusal(QUERY_TOO_LONG_MSG)
        record = build_query_record(
            query=query[:MAX_QUERY_CHARS] + "...(truncated for log)",
            retrieved_chunks=[],
            retrieval_latency_ms=0.0,
            synthesis_result={
                "brief": brief,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            },
            model_used=synth.model,
            failure_reason=f"guardrail: query_too_long ({len(query)} chars)",
            query_id=query_id,
        )
        qlogger.log(record)
        _safe_db_write(record)
        return {**brief.model_dump(), "query_id": query_id}

    # Guardrail 2 — daily cost circuit breaker
    daily_cost = _daily_cost_so_far(log_path)
    if daily_cost >= DAILY_COST_LIMIT_USD:
        brief = _canonical_refusal(COST_LIMIT_MSG)
        record = build_query_record(
            query=query,
            retrieved_chunks=[],
            retrieval_latency_ms=0.0,
            synthesis_result={
                "brief": brief,
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
            },
            model_used=synth.model,
            failure_reason=f"guardrail: daily_cost_limit (${daily_cost:.4f} spent)",
            query_id=query_id,
        )
        qlogger.log(record)
        _safe_db_write(record)
        return {**brief.model_dump(), "query_id": query_id}

    # Guardrail 3 — pre-retrieval domain gate
    # Classifies the query cheaply (GPT-4o-mini, ~$0.00003) before spending
    # retrieval + synthesis tokens (~$0.003). Fails-open on any API error.
    if QUERY_CLASSIFIER_ENABLED:
        classification = classify_query(query)
        if not classification.in_domain:
            brief = _canonical_refusal(OUT_OF_CORPUS_ANSWER)
            record = build_query_record(
                query=query,
                retrieved_chunks=[],
                retrieval_latency_ms=0.0,
                synthesis_result={
                    "brief": brief,
                    "latency_ms": 0.0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cost_usd": 0.0,
                },
                model_used=synth.model,
                failure_reason=f"guardrail: out_of_domain ({classification.reason})",
                query_id=query_id,
            )
            qlogger.log(record)
            _safe_db_write(record)
            return {**brief.model_dump(), "query_id": query_id}

    # Normal pipeline
    failure_reason: str | None = None
    synthesis_result = None
    retrieval_latency_ms = 0.0
    chunks: list[dict] = []
    institutions: list[str] = []

    try:
        institutions = detect_institutions(query) if METADATA_FILTER_ENABLED else []
        if institutions:
            logger.info("Metadata filter active — institutions detected: %s", institutions)

        t0 = time.time()
        chunks = hybrid.retrieve(query, top_k=top_k, institutions=institutions)
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
        query_id=query_id,
        detected_institutions=institutions,
    )
    qlogger.log(record)  # primary sink — always fires
    _safe_db_write(record)  # secondary sink — never-raise on DB down

    if synthesis_result is None:
        return {"error": failure_reason, "query_id": query_id}

    result = synthesis_result["brief"].model_dump()
    result["query_id"] = query_id  # so Streamlit / callers can attach feedback
    return result


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
