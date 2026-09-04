# Phase 0 / Phase 1 Verification Report

Generated: 2026-09-04

## Phase 0 — Baseline Hardening

| Check | Status | Notes |
|---|---|---|
| Hardened citation verifier | PASS | `src/evidence/citations.py` — chunk_id binding, fail-closed, doc_id/page from chunk |
| Adversarial citation tests (6 cases) | PASS | `tests/synthesis/test_citation_verifier_adversarial.py` — all 6 cases present and passing |
| Centralised settings + fingerprint | PASS | `src/config/settings.py` — Pydantic Settings, `settings_fingerprint()` as 8-char SHA-256 hex |
| config_fingerprint in logger | PASS | `src/monitoring/logger.py` — `_get_config_fingerprint()` added to every `build_query_record()` record |
| config_fingerprint in db | PASS | `src/monitoring/db.py` — `config_fingerprint` in `_QUERY_LOGS_INSERT` SQL and `_normalise_record()` |
| Index manifest | PASS | `src/ingestion/index_manifest.py` — `write_manifest()` writes `data/processed/index_manifest.json` with `content_hash` per doc |
| Tombstone logic | PASS | `src/ingestion/dlt_pipeline.py` — `delete_orphaned_chunks()` called before `pipeline.run()`; manifest written after |
| Tombstone tests (4) | PASS | `tests/ingestion/test_tombstone.py` — 4 tests, all passing (171 total pass) |
| Baseline replay script | PASS | `scripts/freeze_baseline.py` present (manual-only, not executed) |

## Phase 1 — Agent Foundations

| Check | Status | Notes |
|---|---|---|
| AgentState + Pydantic models | PASS | `src/agent/state.py` — AgentState TypedDict; `src/evidence/claims.py` — SubQuestion, Claim, Coverage, EvidenceRef |
| Policies constants | PASS | `src/agent/policies.py` — MAX_STEPS=8, MAX_COST_USD=0.05, MAX_TIME_S=60.0, RETRY_LIMIT=2 |
| Router (fail-open verified) | PASS | `src/agent/router.py` — `complexity_router()` wraps entire inner function in `try/except Exception`, returns `(Path.FAST, "factual")` on any error |
| Router labels (30 queries, diversity) | PASS | `data/eval/router_labels.json` — 30 entries; cross_doc=9, contradiction=4, summary=4, unsupported=3, factual=5, numeric=5; no duplicates |
| Planner node | PASS | `src/agent/nodes/planner.py` — retry on malformed JSON, falls back to `termination_reason="fallback_to_fast"` |
| Retriever node | PASS | `src/agent/nodes/retriever.py` — retrieves per sub-question, fails gracefully if no retriever injected |
| Grader node (fail-permissive verified) | PASS | `src/agent/nodes/grader.py` — on LLM/parse failure, all sub-questions default to "covered" |
| LangGraph workflow (budget enforcement) | PASS | `src/agent/workflow.py` — `check_budget()` as conditional edge after planner (before retriever) and after retriever (before grader); graph is acyclic, cannot loop forever |
| Tracing + SQL | PASS | `src/observability/tracing.py` + `monitoring/postgres/init_agent_traces.sql` — `emit_span()` writes to `cpie.agent_traces`, fails silently |
| Shadow runner | PASS | `scripts/shadow_run.py` present (manual-only) |
| Tests — 35 new, all passing | PASS | `tests/agent/test_router.py` (10), `tests/agent/test_termination.py` (9), `tests/agent/test_nodes/test_planner.py` (11), `tests/agent/test_nodes/test_grader_batched.py` (5) — all pass |

## Cross-checks

| Check | Result |
|---|---|
| langgraph confined to workflow.py | YES — only `from langgraph.graph import END, StateGraph` in `workflow.py`; node file mentions are comments, not imports |
| Grader fail-permissive | YES — `_call_grader()` returns `None` on any exception; `run_grader()` then calls `_default_coverage()` setting all to "covered" |
| Router fail-open | YES — `complexity_router()` catches `Exception` and returns `(Path.FAST, "factual")`; inner `_call_router()` raises on any error so the outer wrapper always catches |
| Budget checked before every node | YES — `check_budget()` runs as a conditional edge after planner (gates retriever) and after retriever (gates grader); no pre-planner check needed as budgets start at zero and the graph has no back-edges |
| doc_id/page read from chunk not model | YES — `citations.py` lines 97-104 pull `doc_id` and `page_number` from `matched_chunk`, discarding model-supplied values |

## Gap-fills applied

None. All Phase 0 and Phase 1 deliverables were present and correct on branch `phase-1-foundations`. No code changes were required.

## Test suite

pytest: 171/171 PASS
ruff check: PASS
ruff format: FAIL — 1 pre-existing violation in `src/retrieval/query_rewriter.py` (locked retrieval file, not touched per brief)

## Escalated to user (design-blocked)

None. All items in scope were verifiable and correct.

## Verdict

READY FOR USER REVIEW
