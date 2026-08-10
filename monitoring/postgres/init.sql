-- CPIE query monitoring — Postgres schema
--
-- This file is mounted into the postgres container at
-- /docker-entrypoint-initdb.d/init.sql and runs automatically on the
-- container's FIRST startup (when the postgres data volume is empty).
--
-- Safe to re-run manually against a live DB — every statement is
-- IF NOT EXISTS or CREATE INDEX IF NOT EXISTS.
--
-- Two tables in schema `cpie`:
--   query_logs      — one row per pipeline query
--   user_feedback   — one row per human thumbs vote via Streamlit
--
-- Design decisions and full rationale in Week 5 Step 4a notes.

CREATE SCHEMA IF NOT EXISTS cpie;

-- gen_random_uuid() lives in pgcrypto on older Postgres; native in PG13+.
-- Included defensively — no-op on modern versions.
CREATE EXTENSION IF NOT EXISTS pgcrypto;


-- ─── query_logs ────────────────────────────────────────────────────────
-- One row per query hitting the pipeline (production traffic).
-- Fields mirror the current JSONL logger schema PLUS `answer`,
-- `cited_doc_ids`, and `is_refusal` which JSONL doesn't currently capture.

CREATE TABLE IF NOT EXISTS cpie.query_logs (
    query_id              UUID           PRIMARY KEY,
    ts                    TIMESTAMPTZ    NOT NULL DEFAULT now(),
    query                 TEXT           NOT NULL,

    -- Retrieval
    retrieved_doc_ids     JSONB          NOT NULL DEFAULT '[]'::jsonb,
    retrieved_pages       JSONB          NOT NULL DEFAULT '[]'::jsonb,
    rrf_scores            JSONB          NOT NULL DEFAULT '[]'::jsonb,
    retrieval_latency_ms  REAL           NOT NULL DEFAULT 0,
    detected_institutions JSONB          NOT NULL DEFAULT '[]'::jsonb,

    -- Synthesis
    synthesis_latency_ms  REAL           NOT NULL DEFAULT 0,
    model_used            TEXT,
    prompt_version        TEXT,
    prompt_tokens         INT            NOT NULL DEFAULT 0,
    completion_tokens     INT            NOT NULL DEFAULT 0,
    cost_usd              NUMERIC(12, 8) NOT NULL DEFAULT 0,

    -- Answer
    answer                TEXT,
    is_refusal            BOOLEAN        NOT NULL DEFAULT FALSE,
    cited_doc_ids         JSONB          NOT NULL DEFAULT '[]'::jsonb,
    citation_count        INT            NOT NULL DEFAULT 0,
    contradiction_count   INT            NOT NULL DEFAULT 0,

    -- Failure tracking (nullable on the happy path)
    failure_reason        TEXT
);

-- Indexes tuned to the dashboards we plan in 4c
CREATE INDEX IF NOT EXISTS idx_ql_ts              ON cpie.query_logs (ts DESC);
CREATE INDEX IF NOT EXISTS idx_ql_model_used      ON cpie.query_logs (model_used);
CREATE INDEX IF NOT EXISTS idx_ql_prompt_version  ON cpie.query_logs (prompt_version);
CREATE INDEX IF NOT EXISTS idx_ql_is_refusal_ts   ON cpie.query_logs (is_refusal, ts DESC);
-- Partial index: failures are rare (<1%), keeps this index small.
CREATE INDEX IF NOT EXISTS idx_ql_failure         ON cpie.query_logs (failure_reason)
    WHERE failure_reason IS NOT NULL;


-- ─── user_feedback ─────────────────────────────────────────────────────
-- One row per human thumbs vote from the Streamlit widget.
-- ON DELETE CASCADE — if a query row is ever purged, feedback goes with it.

CREATE TABLE IF NOT EXISTS cpie.user_feedback (
    feedback_id  SERIAL       PRIMARY KEY,
    query_id     UUID         NOT NULL REFERENCES cpie.query_logs(query_id) ON DELETE CASCADE,
    feedback     SMALLINT     NOT NULL CHECK (feedback IN (-1, 1)),   -- +1 up, -1 down
    comment      TEXT,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_uf_query_id    ON cpie.user_feedback (query_id);
CREATE INDEX IF NOT EXISTS idx_uf_created_at  ON cpie.user_feedback (created_at DESC);


-- ─── Sanity check: log DDL applied ─────────────────────────────────────
-- Postgres notices show up in `docker compose logs postgres` so you can
-- confirm at a glance that init ran.
DO $$
BEGIN
    RAISE NOTICE 'cpie schema initialised: query_logs + user_feedback + indexes';
END $$;
