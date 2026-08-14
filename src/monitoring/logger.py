"""
JSON lines query logger.

One record per query, appended to logs/queries.jsonl.
Written from day one (Week 3, Step 5) — raw data layer for the Week 5
Postgres/Grafana observability stack. JSONL stays as the fallback sink
even after Postgres is wired in (Step 4b) — if Postgres is down, the
pipeline keeps logging.

Fields logged:
  query_id, timestamp, query, retrieved_doc_ids, retrieved_pages,
  rrf_scores, retrieval_latency_ms, detected_institutions,
  synthesis_latency_ms, model_used, prompt_version,
  prompt_tokens, completion_tokens, cost_usd,
  answer, is_refusal, cited_doc_ids,
  citation_count, contradiction_count,
  failure_reason

(Confidence + confidence_signals removed in Week 5 Step 2d — the signals
had at best AUC 0.668 for predicting correctness; shipping a weak signal
was worse than shipping nothing.)
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs/queries.jsonl")

# Prefix of the canonical out-of-corpus refusal. Used to compute is_refusal
# without a separate LLM call (empty-chunks guard and LLM Structured Outputs
# refusal branch both emit this exact string — see synthesiser.py).
_CANONICAL_REFUSAL_PREFIX = "The corpus does not contain sufficient information"


class QueryLogger:
    """Appends one JSON record per query. Never raises — logging must not break the pipeline."""

    def __init__(self, log_path: str | Path = DEFAULT_LOG_PATH) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict[str, Any]) -> None:
        """Append a single record. Errors are logged, never raised."""
        record.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning("QueryLogger failed to write record: %s", e)


def build_query_record(
    *,
    query: str,
    retrieved_chunks: list[dict],
    retrieval_latency_ms: float,
    synthesis_result: dict[str, Any] | None,
    model_used: str,
    failure_reason: str | None = None,
    query_id: str | None = None,
    detected_institutions: list[str] | None = None,
) -> dict[str, Any]:
    """
    Assemble the record from pipeline outputs. Pure function — no I/O.

    Both sinks (JSONL logger and Postgres db.insert_query_record) consume
    this same dict, so there's exactly one shape to keep in sync.

    Fields NEW in Step 4b (backward-compat additions):
      query_id              — client-generated UUID; joins to user_feedback
      answer                — brief.answer for refusal detection + spot-checking
      cited_doc_ids         — the doc_ids the LLM actually cited (distinct
                              from retrieved_doc_ids, which is what the
                              retriever surfaced)
      is_refusal            — computed at build time from `answer` prefix
      detected_institutions — passed by caller if the metadata filter ran
    """
    brief = synthesis_result["brief"] if synthesis_result else None
    answer = brief.answer if brief else None
    cited = [c.doc_id for c in brief.citations] if brief else []
    is_refusal = bool(answer and answer.strip().startswith(_CANONICAL_REFUSAL_PREFIX))

    record: dict[str, Any] = {
        "query_id": query_id or str(uuid.uuid4()),
        "query": query,
        # Retrieval
        "retrieved_doc_ids": [c["doc_id"] for c in retrieved_chunks],
        "retrieved_pages": [c["page_number"] for c in retrieved_chunks],
        "rrf_scores": [round(c.get("rrf_score", 0.0), 6) for c in retrieved_chunks],
        "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        "detected_institutions": list(detected_institutions or []),
        # Synthesis
        "synthesis_latency_ms": (round(synthesis_result["latency_ms"], 1) if synthesis_result else 0.0),
        "model_used": model_used,
        "prompt_version": synthesis_result.get("prompt_version") if synthesis_result else None,
        "prompt_tokens": synthesis_result["prompt_tokens"] if synthesis_result else 0,
        "completion_tokens": synthesis_result["completion_tokens"] if synthesis_result else 0,
        "cost_usd": synthesis_result["cost_usd"] if synthesis_result else 0.0,
        # Answer
        "answer": answer,
        "is_refusal": is_refusal,
        "cited_doc_ids": cited,
        "citation_count": len(brief.citations) if brief else 0,
        "contradiction_count": len(brief.contradictions) if brief else 0,
        # Failure tracking
        "failure_reason": failure_reason,
    }
    return record
