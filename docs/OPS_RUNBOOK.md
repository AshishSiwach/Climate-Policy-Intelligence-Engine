# CPIE Agent Route — Ops Runbook

## Toggling the agent route

Set `AGENT_ROUTE_ENABLED` in `.env`:

| Value    | Behaviour |
|----------|-----------|
| `false`  | Shadow mode (safe default). The agent runs silently for eligible queries and logs traces to `cpie.agent_traces`, but the fast-path answer is always returned to the user. |
| `canary` | Agent path is active for 10% of cross_doc queries (random per-request coin flip). 90% of cross_doc traffic and all other traffic still goes to the fast path. |
| `true`   | Agent path is active for all cross_doc queries. Use only after the eval gate has passed (Correctness ≥ 3.50, Completeness ≥ 3.25). |

Restart the app after changing (Streamlit does not hot-reload env vars):

```bash
# docker compose
docker compose restart app

# or direct
kill $(pgrep -f "streamlit run app.py") && streamlit run app.py
```

---

## Disabling the agent route immediately

```bash
# 1. Edit .env
AGENT_ROUTE_ENABLED=false

# 2. Restart
docker compose restart app
```

The fast path is always available as a fallback. Any in-flight agent runs
(daemon threads in shadow mode) will complete or be abandoned when the process
restarts.

---

## Inspecting a trace

All agent workflow steps write rows to `cpie.agent_traces`. Each row is one
node invocation within a trace identified by `trace_id` (= `query_id` from
`cpie.query_logs`).

### List recent traces

```sql
SELECT
    trace_id,
    MIN(created_at)                AS started_at,
    MAX(step_no)                   AS total_steps,
    SUM(cost_usd)                  AS total_cost_usd,
    MAX(termination_reason)        AS termination_reason
FROM cpie.agent_traces
WHERE created_at >= NOW() - INTERVAL '1 hour'
GROUP BY trace_id
ORDER BY started_at DESC
LIMIT 20;
```

### Inspect one trace in full

```sql
SELECT
    step_no,
    node_name,
    latency_ms,
    cost_usd,
    termination_reason,
    input_summary,
    output_summary,
    created_at
FROM cpie.agent_traces
WHERE trace_id = '<paste-trace-id-here>'
ORDER BY step_no;
```

### Find traces that hit the step budget

```sql
SELECT trace_id, MAX(step_no) AS steps, MAX(termination_reason) AS reason
FROM cpie.agent_traces
WHERE termination_reason = 'max_steps'
  AND created_at >= NOW() - INTERVAL '24 hours'
GROUP BY trace_id
ORDER BY steps DESC;
```

### Coverage retry rate (last 24 h)

```sql
SELECT
    100.0 * COUNT(DISTINCT retry_traces.trace_id)::float
        / NULLIF(COUNT(DISTINCT all_traces.trace_id), 0) AS retry_pct
FROM (
    SELECT DISTINCT trace_id FROM cpie.agent_traces
    WHERE created_at >= NOW() - INTERVAL '24 hours'
) AS all_traces
LEFT JOIN (
    SELECT DISTINCT trace_id FROM cpie.agent_traces
    WHERE created_at >= NOW() - INTERVAL '24 hours'
      AND node_name = 'retry_retriever'
) AS retry_traces ON all_traces.trace_id = retry_traces.trace_id;
```

### Mean cost per agent-routed query (last 24 h)

```sql
SELECT ROUND(AVG(trace_cost)::numeric, 6) AS mean_cost_usd
FROM (
    SELECT trace_id, SUM(cost_usd) AS trace_cost
    FROM cpie.agent_traces
    WHERE created_at >= NOW() - INTERVAL '24 hours'
      AND cost_usd IS NOT NULL
    GROUP BY trace_id
) sub;
```

---

## Cross-referencing with query logs

`trace_id` in `cpie.agent_traces` equals `query_id` in `cpie.query_logs`.
To see the original query text for a trace:

```sql
SELECT q.query, q.ts, q.cost_usd, q.model_used
FROM cpie.query_logs q
WHERE q.query_id = '<paste-trace-id-here>';
```

---

## Reproducing a specific query

Given a `request_id` / `query_id`, replay the query through the agent path:

```python
# replay_agent.py — run from the project root
import uuid
from dotenv import load_dotenv
load_dotenv()

from src.agent.workflow import agent_graph

REQUEST_ID = "<paste-request-id-here>"
QUERY = "<paste-query-text-here>"

initial_state = {
    "request_id": REQUEST_ID,
    "query": QUERY,
    "task_type": "cross_doc",
    "sub_questions": [],
    "retrievals": {},
    "coverage": {},
    "claims": [],
    "verified_claims": [],
    "steps_used": 0,
    "cost_used_usd": 0.0,
    "time_used_s": 0.0,
    "retries_used": {},
    "result": None,
    "termination_reason": None,
    # Inject retriever — requires the full index stack to be built
    # "_retriever": hybrid,   # uncomment and build via build_pipeline()
}

final_state = agent_graph.invoke(initial_state)
print("termination:", final_state["termination_reason"])
print("steps used:", final_state["steps_used"])
print("cost (usd):", final_state["cost_used_usd"])
print("answer:", (final_state.get("result") or {}).get("answer", "(none)"))
```

To obtain the original query text from only the request_id:

```sql
SELECT query, ts FROM cpie.query_logs WHERE query_id = '<paste-id>';
```

---

## Grafana

The agent-route dashboard is at: `http://localhost:3000/d/cpie-agent-route`

The dashboard file lives at: `monitoring/grafana/dashboards/agent_dashboard.json`

Key panels:
- **Agent step distribution** — bar chart of steps per run
- **Termination reason breakdown** — pie chart of complete / max_steps / max_cost / max_time / fallback_to_fast
- **Per-node latency** — bar chart of mean ms per node
- **Coverage retry rate** — % of runs that needed a retry_retriever step
- **Mean cost per query** — in USD

---

## Kill criteria (from `docs/AGENT_ROUTE_REASONING.md`)

Any route that fails to hit its success gate on the expanded eval set:

1. **Do not ship it enabled.** Set `AGENT_ROUTE_ENABLED=false` and restart.
2. **Do not tune the gate downward** to make it pass.
3. **Document the failure** in a follow-up section of `docs/AGENT_ROUTE_REASONING.md`
   with the observed numbers and a hypothesis for why the mechanism didn't
   translate to metric gains.

### Cross-document route success gates

| Metric | Baseline | Gate |
|--------|----------|------|
| Correctness (cross-doc) | 3.00 / 5 | ≥ 3.75 |
| Completeness (cross-doc) | 2.50 / 5 | ≥ 3.75 |
| Recall@5 (cross-doc) | 0.750 | ≥ 0.90 |
| Faithfulness regression vs fast path | — | ≤ 0.10 |

Run the A/B eval to check:

```bash
uv run python scripts/eval_crossdoc_ab.py --judge
```

If the agent does not clear the gates: disable the route, document the failure,
and open a follow-up issue. Publishing "we tried it and it didn't move the
metrics enough" is a better outcome than shipping a degraded experience.
