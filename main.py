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
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.agent.router import Path as RoutePath
from src.agent.router import complexity_router
from src.agent.workflow import agent_graph
from src.config.settings import get_settings
from src.monitoring import QueryLogger, build_query_record
from src.monitoring.db import insert_query_record as db_insert_query_record
from src.retrieval import BM25Retriever, HybridRetriever
from src.retrieval.institution_detector import detect_institutions
from src.synthesis import AnalystBrief, Synthesiser
from src.synthesis.output_schema import AnalystBrief as AgentAnalystBrief
from src.synthesis.query_classifier import classify_query
from src.synthesis.synthesiser import OUT_OF_CORPUS_ANSWER

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
    today = datetime.now(timezone.utc).date().isoformat()  # UTC date — matches UTC timestamps in log
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


def _should_use_agent(mode: str, canary_pct: float) -> bool:
    """Return True if the agent path should be used for this request.

    Args:
        mode:       "false" | "canary" | "true" (from AGENT_ROUTE_ENABLED).
        canary_pct: Fraction of requests to route to agent in canary mode.
    """
    if mode == "true":
        return True
    if mode == "canary":
        return random.random() < canary_pct
    return False  # "false" or any unrecognised value → fast path


def _run_fast_path(
    query: str,
    hybrid: HybridRetriever,
    synth: Synthesiser,
    qlogger: QueryLogger,
    top_k: int,
    log_path: Path,
    query_id: str,
) -> dict:
    """Execute the fast path (institution detection → retrieval → synthesis → log).

    This is the complete existing pipeline extracted so agent routing can
    call it explicitly.  Adds ``source: "fast"`` to the returned dict.
    """
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
    qlogger.log(record)
    _safe_db_write(record)

    if synthesis_result is None:
        return {"error": failure_reason, "query_id": query_id, "source": "fast"}

    result = synthesis_result["brief"].model_dump()
    result["query_id"] = query_id
    result["source"] = "fast"
    return result


def _make_agent_initial_state(
    query_id: str,
    query: str,
    task_type: str,
    hybrid: HybridRetriever,
) -> dict:
    """Construct the initial LangGraph state dict."""
    return {
        "request_id": query_id,
        "query": query,
        "task_type": task_type,
        "sub_questions": [],
        "retrievals": {},
        "coverage": {},
        "claims": [],
        "verified_claims": [],
        "steps_used": 0,
        "cost_used_usd": 0.0,
        "_start_time": time.monotonic(),
        "retries_used": {},
        "result": None,
        "termination_reason": None,
        "_retriever": hybrid,
    }


def _agent_state_to_brief(state: dict) -> AgentAnalystBrief:
    """Convert a finished agent state dict to an AgentAnalystBrief."""
    result_dict = state.get("result") or {}
    if result_dict:
        return AgentAnalystBrief(**result_dict)
    return AgentAnalystBrief(
        answer="The agent could not produce an answer for this query.",
        citations=[],
        truncated=True,
        termination_reason=state.get("termination_reason", "max_steps"),
    )


def _log_agent_run(
    query: str,
    query_id: str,
    brief: AgentAnalystBrief,
    latency_s: float,
    cost_usd: float,
    qlogger: QueryLogger,
) -> None:
    """Log a completed agent run to JSONL + Postgres."""
    record = build_query_record(
        query=query,
        retrieved_chunks=[],
        retrieval_latency_ms=0.0,
        synthesis_result={
            "brief": brief,
            "latency_ms": latency_s * 1000,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cost_usd": cost_usd,
        },
        model_used="agent",
        failure_reason=None,
        query_id=query_id,
    )
    qlogger.log(record)
    _safe_db_write(record)


def _evaluate_route(query: str, settings) -> tuple[RoutePath, str, bool]:
    """Classify the query and decide whether to use the agent path.

    Returns (path, task_type, use_agent).
    """
    path, task_type = complexity_router(query)
    use_agent = (
        _should_use_agent(settings.agent.route_enabled, settings.agent.canary_pct)
        if (path == RoutePath.AGENT and task_type == "cross_doc")
        else False
    )
    return path, task_type, use_agent


def _check_guardrails(
    query: str,
    log_path: Path,
) -> tuple[AnalystBrief, str] | None:
    """Evaluate all three input guardrails.

    Returns (brief, failure_reason) if one fires, else None.
    """
    if len(query) > MAX_QUERY_CHARS:
        return (
            _canonical_refusal(QUERY_TOO_LONG_MSG),
            f"guardrail: query_too_long ({len(query)} chars)",
        )
    daily_cost = _daily_cost_so_far(log_path)
    if daily_cost >= DAILY_COST_LIMIT_USD:
        return (
            _canonical_refusal(COST_LIMIT_MSG),
            f"guardrail: daily_cost_limit (${daily_cost:.4f} spent)",
        )
    if QUERY_CLASSIFIER_ENABLED:
        classification = classify_query(query)
        if not classification.in_domain:
            return (
                _canonical_refusal(OUT_OF_CORPUS_ANSWER),
                f"guardrail: out_of_domain ({classification.reason})",
            )
    return None


def _log_guardrail_refusal(
    query: str,
    query_id: str,
    brief: AnalystBrief,
    failure_reason: str,
    synth: Synthesiser,
    qlogger: QueryLogger,
) -> None:
    """Log a guardrail refusal to JSONL + Postgres."""
    query_to_log = (
        query[:MAX_QUERY_CHARS] + "...(truncated for log)"
        if len(query) > MAX_QUERY_CHARS
        else query
    )
    record = build_query_record(
        query=query_to_log,
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
        failure_reason=failure_reason,
        query_id=query_id,
    )
    qlogger.log(record)
    _safe_db_write(record)


def _run_agent_path(
    query: str,
    task_type: str,
    hybrid: HybridRetriever,
    qlogger: QueryLogger,
    query_id: str,
) -> dict:
    """Execute the agent path (LangGraph workflow → AnalystBrief → log).

    Returns a dict in the same shape as the fast path.  Raises on any
    unhandled error so the caller can fall back to the fast path.
    Adds ``source: "agent"`` to the returned dict.
    """
    t_start = time.time()
    initial_state = _make_agent_initial_state(query_id, query, task_type, hybrid)
    final_state = agent_graph.invoke(initial_state)
    latency_s = time.time() - t_start

    if final_state.get("termination_reason") == "fallback_to_fast":
        raise RuntimeError("Planner requested fallback_to_fast — routing to fast path")

    brief = _agent_state_to_brief(final_state)
    _log_agent_run(query, query_id, brief, latency_s, final_state.get("cost_used_usd", 0.0), qlogger)

    result = brief.model_dump()
    result["query_id"] = query_id
    result["source"] = "agent"
    return result


def _run_agent_shadow(
    query: str,
    task_type: str,
    hybrid: HybridRetriever,
    query_id: str,
) -> None:
    """Fire-and-forget agent run in a daemon thread (shadow / dry-run mode).

    The result is discarded — only the LangGraph traces land in
    cpie.agent_traces.  This lets us measure agent behaviour in production
    traffic before enabling the agent path for users.
    """

    def _shadow_task() -> None:
        try:
            initial_state = _make_agent_initial_state(query_id, query, task_type, hybrid)
            final_state = agent_graph.invoke(initial_state)
            logger.info(
                "Shadow agent run complete: query_id=%s termination=%s cost=$%.4f steps=%d",
                query_id,
                final_state.get("termination_reason"),
                final_state.get("cost_used_usd", 0.0),
                final_state.get("steps_used", 0),
            )
        except Exception as exc:
            logger.warning(
                "Shadow agent run failed: query_id=%s error=%s", query_id, exc
            )

    thread = threading.Thread(target=_shadow_task, daemon=True, name=f"shadow-{query_id[:8]}")
    thread.start()


def build_pipeline() -> tuple[HybridRetriever, Synthesiser]:
    """Load indices + retriever + synthesiser. One-time setup per CLI invocation."""
    if not BM25_PATH.exists():
        raise FileNotFoundError(f"BM25 index not found at {BM25_PATH}. Run: uv run python scripts/build_indices.py")
    if not CHROMA_DIR.exists():
        raise FileNotFoundError(f"Chroma index not found at {CHROMA_DIR}. Run: uv run python scripts/build_indices.py")

    # Lazy import — see module-level note.
    from src.retrieval import DenseRetriever

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

    # Guardrails — length limit, daily cost, and domain gate
    refusal = _check_guardrails(query, log_path)
    if refusal is not None:
        brief, failure_reason = refusal
        _log_guardrail_refusal(query, query_id, brief, failure_reason, synth, qlogger)
        return {**brief.model_dump(), "query_id": query_id, "source": "fast"}

    # ── Agent routing ────────────────────────────────────────────────────────
    # Classify complexity AFTER guardrails — avoids spending router tokens on
    # queries that will be refused anyway. Fails-open (returns FAST/factual)
    # so any router error falls through to the fast path transparently.
    settings = get_settings()
    path, task_type, use_agent = _evaluate_route(query, settings)

    if use_agent:
        try:
            return _run_agent_path(query, task_type, hybrid, qlogger, query_id)
        except Exception:
            logger.exception(
                "Agent path failed for query: %r — falling back to fast path", query
            )
            # Fall through to fast path below

    # Shadow mode: run agent in background if the query is agent-eligible but
    # the flag is not fully enabled (or the canary coin flip said fast path).
    if path == RoutePath.AGENT and settings.agent.route_enabled in ("false", "canary"):
        _run_agent_shadow(query, task_type, hybrid, query_id)

    return _run_fast_path(query, hybrid, synth, qlogger, top_k, log_path, query_id)


def run_query_with_progress(
    query: str,
    hybrid: HybridRetriever,
    synth: Synthesiser,
    qlogger: QueryLogger,
    top_k: int = 5,
    log_path: Path = LOG_PATH,
):
    """Generator version of run_query that emits per-node progress events for agent queries.

    Yields dicts of three shapes:
        {"type": "routing", "path": "fast" | "agent"}   — always first
        {"type": "node",    "name": str, "update": dict} — agent only, once per node
        {"type": "result",  "brief": dict}               — always last

    Guardrail refusals (length / daily cost / out-of-domain) yield "routing"="fast"
    then immediately "result" — no "node" events, matching the fast-path UX.
    """
    query_id = str(uuid.uuid4())

    # Guardrails — length limit, daily cost, and domain gate
    refusal = _check_guardrails(query, log_path)
    if refusal is not None:
        brief, failure_reason = refusal
        _log_guardrail_refusal(query, query_id, brief, failure_reason, synth, qlogger)
        r = brief.model_dump()
        r["query_id"] = query_id
        r["source"] = "fast"
        yield {"type": "routing", "path": "fast"}
        yield {"type": "result", "brief": r}
        return

    # ── Agent routing ────────────────────────────────────────────────────────
    settings = get_settings()
    path, task_type, use_agent = _evaluate_route(query, settings)

    if use_agent:
        yield {"type": "routing", "path": "agent"}

        initial_state = _make_agent_initial_state(query_id, query, task_type, hybrid)
        t_start = time.time()
        accumulated: dict = {}

        try:
            for chunk in agent_graph.stream(initial_state, stream_mode="updates"):
                node_name = next(iter(chunk))
                state_update = chunk[node_name]
                accumulated.update(state_update)
                yield {"type": "node", "name": node_name, "update": state_update}
        except Exception:
            logger.exception("Agent streaming failed for query_id=%s — falling back to fast path", query_id)
            yield {"type": "routing", "path": "fast"}
            result = _run_fast_path(query, hybrid, synth, qlogger, top_k, log_path, query_id)
            yield {"type": "result", "brief": result}
            return

        latency_s = time.time() - t_start

        if accumulated.get("termination_reason") == "fallback_to_fast":
            logger.info("Agent streaming: planner requested fallback_to_fast — routing to fast path")
            yield {"type": "routing", "path": "fast"}
            result = _run_fast_path(query, hybrid, synth, qlogger, top_k, log_path, query_id)
            yield {"type": "result", "brief": result}
            return

        brief_agent = _agent_state_to_brief(accumulated)
        agent_cost = accumulated.get("cost_used_usd", 0.0)
        _log_agent_run(query, query_id, brief_agent, latency_s, agent_cost, qlogger)

        result = brief_agent.model_dump()
        result["query_id"] = query_id
        result["source"] = "agent"
        yield {"type": "result", "brief": result}

    else:
        yield {"type": "routing", "path": "fast"}
        if path == RoutePath.AGENT and settings.agent.route_enabled in ("false", "canary"):
            _run_agent_shadow(query, task_type, hybrid, query_id)
        result = _run_fast_path(query, hybrid, synth, qlogger, top_k, log_path, query_id)
        yield {"type": "result", "brief": result}


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
