"""
JSON lines query logger.

One record per query, appended to logs/queries.jsonl.
Written from day one (Week 3, Step 5) — raw data layer for the Week 5
Logfire observability integration.

Fields logged (per CLAUDE.md Week 3 Step 5):
  timestamp, query, retrieved_doc_ids, retrieval_scores, rrf_scores,
  retrieval_latency_ms, synthesis_latency_ms, confidence, confidence_signals,
  model_used, prompt_tokens, completion_tokens, cost_usd, failure_reason
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("logs/queries.jsonl")


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
) -> dict[str, Any]:
    """Assemble the JSONL record from pipeline outputs. Pure function — no I/O."""
    brief = synthesis_result["brief"] if synthesis_result else None
    signals = synthesis_result["confidence_signals"] if synthesis_result else {}

    record: dict[str, Any] = {
        "query": query,
        "retrieved_doc_ids": [c["doc_id"] for c in retrieved_chunks],
        "retrieved_pages": [c["page_number"] for c in retrieved_chunks],
        "rrf_scores": [round(c.get("rrf_score", 0.0), 6) for c in retrieved_chunks],
        "retrieval_latency_ms": round(retrieval_latency_ms, 1),
        "synthesis_latency_ms": (
            round(synthesis_result["latency_ms"], 1) if synthesis_result else 0.0
        ),
        "confidence": brief.confidence if brief else None,
        "confidence_signals": signals,
        "model_used": model_used,
        "prompt_tokens": synthesis_result["prompt_tokens"] if synthesis_result else 0,
        "completion_tokens": synthesis_result["completion_tokens"] if synthesis_result else 0,
        "cost_usd": synthesis_result["cost_usd"] if synthesis_result else 0.0,
        "citation_count": len(brief.citations) if brief else 0,
        "contradiction_count": len(brief.contradictions) if brief else 0,
        "failure_reason": failure_reason,
    }
    return record
