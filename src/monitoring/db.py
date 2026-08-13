"""
Postgres writer for the monitoring stack (Week 5 Step 4a).

Two public writers:
  insert_query_record(record: dict) -> None
  insert_feedback(query_id, feedback, comment=None) -> int | None

Two read helpers (mostly for smoke tests and 4a verification):
  fetch_recent_queries(limit: int = 10) -> list[dict]
  fetch_feedback_for_query(query_id) -> list[dict]

Design decisions:
  - psycopg v3 with a module-level ConnectionPool (lazy-init on first use).
    Cheap for our query volume; scales fine to Grafana + Streamlit sharing
    the same pool. min_size=1 avoids startup cost when nothing is querying.
  - JSONL logger stays as the primary sink; Postgres is a secondary,
    dashboard-oriented store. If Postgres is down, the pipeline continues —
    every writer swallows exceptions and logs a warning.
  - Connection creds come from env vars matching docker-compose.yml defaults,
    so `docker compose up postgres` + `python -m ...` just works with no
    config edits.
  - JSONB columns are populated by json.dumps() on the Python list — psycopg
    v3 supports direct Python-to-jsonb but the explicit dumps keeps the
    "what got sent" easy to reason about and identical on all psycopg minor
    versions.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any

logger = logging.getLogger(__name__)

# ─── Connection ─────────────────────────────────────────────────────────

# Defaults intentionally match docker-compose.yml so
# `docker compose up postgres` + module import works with no env setup.
_DEFAULTS = {
    "POSTGRES_HOST":     "localhost",
    "POSTGRES_PORT":     "5432",
    "POSTGRES_USER":     "cpie",
    "POSTGRES_PASSWORD": "cpie_dev_only",
    "POSTGRES_DB":       "cpie",
}


def _dsn() -> str:
    def env(k: str) -> str:
        return os.environ.get(k, _DEFAULTS[k])
    return (
        f"host={env('POSTGRES_HOST')} port={env('POSTGRES_PORT')} "
        f"user={env('POSTGRES_USER')} password={env('POSTGRES_PASSWORD')} "
        f"dbname={env('POSTGRES_DB')}"
    )


# Lazy-init module singleton — no Postgres connection unless something writes.
_pool = None


def _get_pool():
    """Return the module-level connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        # Deferred import — psycopg_pool pulls in psycopg + libpq. Only load
        # when the pipeline actually calls the DB.
        from psycopg_pool import ConnectionPool
        _pool = ConnectionPool(_dsn(), min_size=1, max_size=5, open=True)
    return _pool


def close_pool() -> None:
    """Close the pool. Safe to call multiple times. Used in test teardown."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None


# ─── query_logs writer ──────────────────────────────────────────────────

_QUERY_LOGS_INSERT = """
INSERT INTO cpie.query_logs (
    query_id, ts, query,
    retrieved_doc_ids, retrieved_pages, rrf_scores,
    retrieval_latency_ms, detected_institutions,
    synthesis_latency_ms, model_used, prompt_version,
    prompt_tokens, completion_tokens, cost_usd,
    answer, is_refusal, cited_doc_ids,
    citation_count, contradiction_count,
    failure_reason
) VALUES (
    %(query_id)s, COALESCE(%(ts)s::timestamptz, now()), %(query)s,
    %(retrieved_doc_ids)s, %(retrieved_pages)s, %(rrf_scores)s,
    %(retrieval_latency_ms)s, %(detected_institutions)s,
    %(synthesis_latency_ms)s, %(model_used)s, %(prompt_version)s,
    %(prompt_tokens)s, %(completion_tokens)s, %(cost_usd)s,
    %(answer)s, %(is_refusal)s, %(cited_doc_ids)s,
    %(citation_count)s, %(contradiction_count)s,
    %(failure_reason)s
)
ON CONFLICT (query_id) DO NOTHING
"""


def insert_query_record(record: dict[str, Any]) -> None:
    """
    Write one row into ``cpie.query_logs``.

    Never raises — Postgres failure is logged as a warning and the pipeline
    continues (JSONL sink is the primary fallback). Missing keys in ``record``
    are filled with sensible defaults so partial dicts still insert.

    ``record`` keys consumed (all optional except ``query``):
      query_id, ts (or timestamp), query, retrieved_doc_ids, retrieved_pages,
      rrf_scores, retrieval_latency_ms, detected_institutions,
      synthesis_latency_ms, model_used, prompt_version, prompt_tokens,
      completion_tokens, cost_usd, answer, is_refusal, cited_doc_ids,
      citation_count, contradiction_count, failure_reason
    """
    params = _normalise_record(record)
    try:
        pool = _get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(_QUERY_LOGS_INSERT, params)
    except Exception as e:
        logger.warning("Postgres insert_query_record failed: %s", e)


def _normalise_record(record: dict) -> dict:
    """
    Coerce a raw pipeline record into the shape ``_QUERY_LOGS_INSERT`` expects:
      - fills missing keys with safe defaults
      - json.dumps() every list-valued field for jsonb columns
      - accepts either ``ts`` or ``timestamp`` (JSONL logger uses ``timestamp``)
    """
    out: dict[str, Any] = dict(record)
    out.setdefault("query_id", str(uuid.uuid4()))
    # Accept both `ts` (Postgres) and `timestamp` (JSONL) as source of the time.
    out.setdefault("ts", record.get("timestamp"))
    out.setdefault("query", record.get("query", ""))

    for k in ("retrieved_doc_ids", "retrieved_pages", "rrf_scores",
              "detected_institutions", "cited_doc_ids"):
        v = out.get(k)
        out[k] = json.dumps(v if v is not None else [])

    out.setdefault("retrieval_latency_ms",  0.0)
    out.setdefault("synthesis_latency_ms",  0.0)
    out.setdefault("prompt_tokens",         0)
    out.setdefault("completion_tokens",     0)
    out.setdefault("citation_count",        0)
    out.setdefault("contradiction_count",   0)
    out.setdefault("cost_usd",              0.0)
    out.setdefault("model_used",            None)
    out.setdefault("prompt_version",        None)
    out.setdefault("answer",                None)
    out.setdefault("is_refusal",            False)
    out.setdefault("failure_reason",        None)
    return out


# ─── user_feedback writer ───────────────────────────────────────────────

def insert_feedback(
    query_id: str | uuid.UUID,
    feedback: int,
    comment: str | None = None,
) -> int | None:
    """
    Insert one row into ``cpie.user_feedback``. Returns the new ``feedback_id``,
    or ``None`` if the DB write failed (never raises).

    ``feedback`` MUST be +1 (thumbs up) or -1 (thumbs down); table CHECK
    enforces it too, but we validate here for a clearer error before the DB call.
    """
    if feedback not in (-1, 1):
        raise ValueError(f"feedback must be -1 or +1, got {feedback!r}")
    try:
        pool = _get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """INSERT INTO cpie.user_feedback (query_id, feedback, comment)
                   VALUES (%s, %s, %s) RETURNING feedback_id""",
                (str(query_id), feedback, comment),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        logger.warning("Postgres insert_feedback failed: %s", e)
        return None


# ─── Read helpers (tests + 4a smoke verification) ───────────────────────

def fetch_recent_queries(limit: int = 10) -> list[dict]:
    """
    Return the ``limit`` most recent rows from ``cpie.query_logs`` as dicts.
    Returns ``[]`` on any DB error (never raises).
    """
    try:
        pool = _get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT query_id, ts, query, model_used, prompt_version,
                          is_refusal, citation_count, cost_usd, failure_reason
                   FROM cpie.query_logs
                   ORDER BY ts DESC LIMIT %s""",
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.warning("Postgres fetch_recent_queries failed: %s", e)
        return []


def fetch_feedback_for_query(query_id: str | uuid.UUID) -> list[dict]:
    """Return all feedback rows for a given ``query_id``, ordered oldest-first."""
    try:
        pool = _get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                """SELECT feedback_id, feedback, comment, created_at
                   FROM cpie.user_feedback WHERE query_id = %s
                   ORDER BY created_at""",
                (str(query_id),),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception as e:
        logger.warning("Postgres fetch_feedback_for_query failed: %s", e)
        return []
