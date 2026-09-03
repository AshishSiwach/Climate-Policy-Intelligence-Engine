"""
Agent tracing — Phase 1 Foundations.

Writes one row per agent step to cpie.agent_traces in Postgres.
Uses the same connection pool pattern as src/monitoring/db.py.

Fails silently — tracing must never break the agent.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# Connection defaults match docker-compose.yml and src/monitoring/db.py
_DEFAULTS = {
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_USER": "cpie",
    "POSTGRES_PASSWORD": "cpie_dev_only",
    "POSTGRES_DB": "cpie",
}


def _dsn() -> str:
    def env(k: str) -> str:
        return os.environ.get(k, _DEFAULTS[k])

    return (
        f"host={env('POSTGRES_HOST')} port={env('POSTGRES_PORT')} "
        f"user={env('POSTGRES_USER')} password={env('POSTGRES_PASSWORD')} "
        f"dbname={env('POSTGRES_DB')}"
    )


_pool = None


def _get_pool():
    """Return module-level connection pool, creating it on first use."""
    global _pool
    if _pool is None:
        from psycopg_pool import ConnectionPool

        _pool = ConnectionPool(_dsn(), min_size=1, max_size=5, open=True)
    return _pool


def close_pool() -> None:
    """Close the pool. Safe to call multiple times."""
    global _pool
    if _pool is not None:
        try:
            _pool.close()
        except Exception:
            pass
        _pool = None


_INSERT_SQL = """
INSERT INTO cpie.agent_traces (
    trace_id, step_no, node_name,
    input_summary, output_summary, tool_called,
    latency_ms, cost_usd, termination_reason
) VALUES (
    %s::uuid, %s, %s,
    %s::jsonb, %s::jsonb, %s,
    %s, %s, %s
)
"""


def emit_span(
    trace_id: str,
    step_no: int,
    node_name: str,
    input_summary: dict,
    output_summary: dict,
    tool_called: str | None,
    latency_ms: int,
    cost_usd: float,
    termination_reason: str | None = None,
) -> None:
    """Write one row to cpie.agent_traces. Fails silently."""
    try:
        pool = _get_pool()
        with pool.connection() as conn, conn.cursor() as cur:
            cur.execute(
                _INSERT_SQL,
                (
                    trace_id,
                    step_no,
                    node_name,
                    json.dumps(input_summary),
                    json.dumps(output_summary),
                    tool_called,
                    latency_ms,
                    cost_usd,
                    termination_reason,
                ),
            )
    except Exception as exc:
        logger.warning("emit_span failed (tracing is non-critical): %s", exc)
