-- Agent traces table — Phase 1 Foundations
-- Run after init.sql (which creates the cpie schema).

CREATE TABLE IF NOT EXISTS cpie.agent_traces (
    id                 SERIAL PRIMARY KEY,
    trace_id           UUID NOT NULL,
    step_no            INT NOT NULL,
    node_name          TEXT NOT NULL,
    input_summary      JSONB,
    output_summary     JSONB,
    tool_called        TEXT,
    latency_ms         INT,
    cost_usd           NUMERIC(10, 6),
    termination_reason TEXT,
    created_at         TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_traces_trace_id ON cpie.agent_traces(trace_id);
