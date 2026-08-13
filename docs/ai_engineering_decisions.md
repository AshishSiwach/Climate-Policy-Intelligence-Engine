# CPIE — AI Engineering Decisions

This document records every significant technical decision made during the CPIE build,
together with the evidence that drove it. Where a component was built, measured, and
then removed, that is recorded here too — the evidence trail is part of the engineering.

Full measurement data: `docs/week5_failure_analysis.md` and `data/eval/results/`.

---

## 1. Problem Framing and Scope

### Decision: Single-turn domain RAG, not a deep research system

CPIE is scoped as a **single-turn domain RAG** system: one retrieval pass, one LLM
call, structured brief out. Deep-research patterns (query decomposition, agentic loops,
iterative retrieval, multi-minute latencies, report-length output) are explicitly
deferred to future versions.

**Why:** Deep research architecture compounds failure modes at every loop iteration.
For a first version over a fixed 12-document corpus, the appropriate benchmark is
"does single-turn RAG work well?", not "can we replace a research team?". Shipping
a clean, measurable single-turn system is more honest and more useful than a complex
agentic system with unmeasured failure modes.

**Consequence for interpreting failures:** Any failure is a single-turn-RAG failure
(prompt calibration, top-k coverage, threshold calibration), not an architecture gap.
This framing prevents misattributing retrieval misses to "not agentic enough".

### Decision: Trusted corpus, no user uploads

All 12 source PDFs are curated public documents from named institutions (Ofgem, DESNZ,
CCC, IEA, BoE, ESO). User-uploaded documents are not accepted.

**Why:** Accepting user content would require corpus-side prompt-injection scanning
(users could upload a PDF that instructs the LLM to ignore system instructions),
content-provenance metadata per document, and per-user chunk isolation in both
Chroma and BM25. These are future concerns. A trusted, curated corpus keeps the failure
surface small and evaluation meaningful.

---

## 2. Corpus Preparation Decisions

### Decision: Sliding window chunking as baseline (400 tokens / 80-token overlap)

All 12 documents are chunked with a 400-token window and 80-token overlap using the
`cl100k_base` tokeniser (tiktoken). A 50-token minimum floor discards fragments; a
512-token hard ceiling catches any chunk that grows after heading injection.

**Why sliding window and not document-aware chunking:** Every chunking-strategy claim
needs empirical validation. Shipping document-aware chunking (Ofgem by paragraph,
IEA by section) without measuring it against a baseline would be "changed stance
without evidence." Sliding window is the standard scientific baseline. Document-aware
chunking is a future experiment to be measured against this baseline.

**Parameter justification:** 400/80 was validated in the Week 2 audit notebook.
Avg token count across all chunks: 397–400. Zero chunks exceeded the 512-token
ceiling. The 80-token overlap keeps sentence boundaries intact across chunk
boundaries.

### Decision: Three-tier table handling strategy

Tables are not treated uniformly. After auditing all 12 documents:

**Tier 1 — Active stripping before chunking**
Two documents contain layout noise that poisons chunks if left in:
- **ESO Beyond 2030**: Strip interactive PDF nav elements ("Navigation", "Download a
  pdf"), duplicated map headers, fragmented social handles.
- **Ofgem SSES**: Strip running header pattern ("Consultation Smart Secure Electricity
  Systems...NN") and `OFFICIAL OFFICIAL` security stamps.

These produce garbage tokens that contaminate BM25's term statistics and embed
as off-topic dense vectors.

**Tier 2 — Section heading injection on table pages**
Five documents contain real numeric tables that analysts will query (WEO 2025 LCOE
tables, Seventh Carbon Budget sector trajectories, BoE financial exposure tables,
CCC Progress status tables). For pages where `page.find_tables()` returns confirmed
real tables (fill-ratio filtered — raw counts overcounted by 40–60% due to header
detection), prepend the nearest section heading (walking back up to 3 pages).

This makes table chunks retrievable by BM25 keyword search and gives the dense
retriever a topic anchor. Without heading injection, a table chunk containing only
numbers and column headers is nearly invisible to both retrievers.

**Tier 3 — Keep as-is**
Remaining documents have either prose-cell tables (CBES, Measuring Climate Risks)
or tables whose content duplicates surrounding prose. No special handling needed.

**Note on fill-ratio filtering:** `page.find_tables()` raw counts significantly
overcount on several documents (WEO 2025: 338 raw → 122 real; ESO Beyond2030:
118 raw → 53). Applied a fill-ratio filter (discard detections where >70% of data
cells are empty) to avoid injecting headings onto false-positive "table" pages.

### Decision: dlt + DuckDB as the ingestion pipeline

The ingestion pipeline uses `dlt` to write chunks into a DuckDB table (`cpie.chunks`)
rather than writing JSON files to disk.

**Why dlt + DuckDB:**
- `build_indices.py` reads directly from DuckDB with a single SQL query — one
  canonical source of truth for both BM25 and Chroma, no JSON sync issues.
- `dlt`'s `write_disposition="merge"` on `chunk_id` gives incremental ingestion
  for free: re-running on a changed corpus only updates changed chunks.
- Matches LLM Zoomcamp course tooling — peer reviewers recognise the pattern.
- DuckDB (17 MB) can be committed to git; JSON files (scattered) cannot.

---

## 3. Retrieval Architecture Decisions

### Decision: Hybrid retrieval (BM25 + dense + RRF), not vector-only

CPIE uses three retrieval components fused with Reciprocal Rank Fusion:

- **BM25** (`rank-bm25`, BM25Okapi): handles exact keyword matches — institution
  names ("Ofgem", "FCA"), policy codes ("SSES", "ZEV"), numeric values.
- **Dense** (`BAAI/bge-base-en-v1.5`, Chroma): handles semantic equivalence —
  "aggregate losses" matching chunks that say "corporate losses", paraphrase matching.
- **RRF k=60**: fuses ranked lists without score normalisation, tolerant of scale
  differences between BM25 term-frequency scores and cosine similarities.

**Why not vector-only:** Climate policy documents are dense with named entities,
regulation codes, and numerical values that exact-match retrieval handles more
reliably than embedding similarity. A query for "Ofgem Class B licence" will surface
relevant chunks via BM25 regardless of how the embedding model represents "Class B"
in 768-dimensional space. Hybrid consistently beats either retriever alone on
this corpus.

### Decision: BAAI/bge-base-en-v1.5 (768-dim) over all-MiniLM-L6-v2 (384-dim)

Measured in the Week 2 audit notebook on 5 queries across 3 representative documents:

| Model | Mean cosine sim | Top-5 manual relevance |
|---|---|---|
| all-MiniLM-L6-v2 | 0.278 | 3/5 relevant |
| BAAI/bge-base-en-v1.5 | **0.543** | 5/5 relevant |

BGE-base's higher-dimensional space (768 vs 384) captures domain-specific
co-occurrence patterns in regulatory text that the smaller model loses. The
performance gap justified the ~2× indexing time increase.

**Note:** Both models run on CUDA (RTX 4050) locally. CPU-only mode (containers)
is ~4× slower but functionally identical.

### Decision: RRF k=60 (Cormack et al. 2009 default)

RRF score = `1 / (k + rank)`. Tested k=10, k=30, k=60 on 3 representative queries.
All three returned identical top-5 rankings. Selected k=60 as the literature default
(Cormack, Clarke & Buettcher 2009) — conservative fusion appropriate for a small
single-domain corpus where retriever agreement is high.

**Why RRF over score normalisation:** BM25 scores and cosine similarities have
different scales and non-comparable distributions. Score normalisation introduces
hyperparameters that need tuning per corpus. RRF is parameter-free apart from k,
robust, and achieves near-optimal fusion without normalisation.

### Decision: Institution metadata filter

Before retrieval, the query text is scanned for named institutions (Ofgem, FCA, IEA,
BoE, CCC, DESNZ, ESO). If detected, Chroma retrieval is pre-filtered to those
institutions via its native `where` clause, and BM25 results are post-filtered to
matching `institution` metadata.

**A/B evidence (47-query ground truth, gpt-5.4-mini judge):**

| Metric | Without filter | With filter | Delta |
|---|---|---|---|
| Recall@5 | ~0.88 | 0.907 | +0.03 |
| Cross-doc Completeness | ~2.0 | ~3.0 | +1.0 |
| Refusal_appropriateness | ~4.55 | 4.81 | +0.26 |
| Retrieval latency | ~155ms | ~175ms | +18ms |

The +1.0 cross-doc Completeness gain was decisive. The filter also fixed a concrete
fabrication: a query naming only "SMR" (Small Modular Reactors, an IEA topic) was
returning BoE CBES chunks about "stress testing" — the filter prevents retrieval from
bleeding across institutions on named queries.

### Decision: Reranker NOT active (preserved but not wired)

A cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) was built and
ablated at two scales:

**Week 3 ablation (3 pilot queries, 4 configs: BM25 / Dense / Hybrid / Full):**
All 4 configs got 3/3 top-1 hits. Reranker added 172ms with zero improvement.

**Week 5 re-evaluation (47 queries × 4 configs, LLM-as-judge):**

| Config | Correctness | Retrieval latency |
|---|---|---|
| BM25 only | ~3.8 | ~50ms |
| Dense only | ~3.7 | ~80ms |
| Hybrid (BM25+Dense+RRF) | ~4.0 | ~155ms |
| Full (Hybrid + Rerank) | ~4.0 | **~800ms** |

The reranker added **5.2× retrieval latency** with zero aggregate Correctness gain.
The only per-type signal: numeric queries showed +1.0 Correctness (n=4, small sample).

**Decision:** Reranker not active. Code preserved in `src/retrieval/reranker.py`
for a future experiment: per-query-type activation (reranker fires only when a query
classifier identifies the query as numeric/table-lookup).

**Principle applied:** When a measured component fails to improve the target metric,
do not ship it. 5.2× latency is a real cost; "might help on some queries" is not
a ship signal without evidence.

---

## 4. LLM Selection and Synthesis Decisions

### Decision: GPT-5.4-mini for synthesis (upgraded from GPT-4o-mini)

Initial model selection (Week 2): GPT-4o-mini beat Claude Haiku 3.5 on 10 pilot
queries at equivalent cost tier (both ~$0.15/1M input tokens at the time).

Week 5 model upgrade experiment: tested GPT-5.4-mini against GPT-4o-mini on the
52-query ground truth.

**Judge strictness confound discovered and controlled:**
Running A → C (change synth model AND judge simultaneously) showed apparent
Correctness drop of −0.527. Investigating: judge alone accounted for −0.615
(gpt-5.4 is a much stricter judge than gpt-5.4-mini). Running the tiebreaker (B:
same synth, different judge) isolated:

- Judge strictness effect: −0.615 Correctness
- Model effect (B vs C, same strict judge): **+0.089 Correctness**

**Apples-to-apples results (gpt-5.4 judge, 52 queries):**

| Metric | GPT-4o-mini | GPT-5.4-mini | Delta |
|---|---|---|---|
| Correctness | 3.519 | 3.608 | +0.089 |
| Faithfulness | 4.443 | 4.692 | **+0.249** |
| Completeness | 2.880 | 3.038 | +0.158 |
| Synthesis latency | ~4,050ms | ~3,213ms | **−837ms (−26%)** |

GPT-5.4-mini won on every metric including latency. Cost: 6× per query (~$0.0005
→ ~$0.003). At demo scale (100 queries/day) that is ~$9/month — accepted.

**Process discipline lesson:** Always run the judge-only tiebreaker BEFORE comparing
model A vs model B across different judges. In this experiment, the judge strictness
effect was ~7× the model effect. Running A→C alone would have called it a regression.

### Decision: Pydantic Structured Outputs, not prompt-parsed JSON

`AnalystBrief` is a Pydantic model validated by OpenAI's Structured Outputs mode.
The schema (`answer`, `citations[]`, `contradictions[]`) is enforced server-side;
if the LLM emits a refusal (`message.refusal`), the pipeline catches it without
attempting JSON parsing.

**Why not prompt-parsed JSON:** Prompt instructions to "return JSON" produce
format errors on edge cases (truncated output, model confusion about nested schemas).
Structured Outputs eliminates the format-validation failure mode entirely and makes
the CRAG correction layer clean: `message.refusal is not None` is a boolean, not
a JSON parse attempt on a refusal string.

### Decision: Prompt versioning with A/B infrastructure

Three prompt variants were authored and tested against the 52-query ground truth
with the same judge and same retrieval config:

| Variant | Description | Correctness | Completeness | Faithfulness |
|---|---|---|---|---|
| v1 (baseline) | Standard RAG prompt | ~4.02 | ~3.15 | ~4.37 |
| v2_crossdoc | + "compare and contrast" for multi-source | 4.096 | 3.385 | 4.308 |
| **v2_numeric** | + "extract verbatim values + page citation" | **4.135** | 3.327 | **4.365** |

`v2_numeric` shipped as the default: best aggregate Correctness, no regressions
on any metric. The verbatim-extraction instruction turned out to be a generally-good
prompt improvement, not only useful for numeric queries.

`v2_crossdoc` is preserved in `PROMPT_REGISTRY` — per-type activation (fires only
on queries classified as `cross_document`) is a future roadmap item.

**Why prompt versioning matters:** Without versioned prompts logged per-query, any
prompt change makes all historical eval data non-comparable. `prompt_version` is
logged in both JSONL and Postgres so every score can be traced to the exact prompt
that generated it.

---

## 5. Correction Layer Design (CRAG-style)

### Decision: LLM refusal as the primary correction signal

CPIE implements a coarse Corrective RAG pattern between retrieval and synthesis:

- **CORRECT path**: LLM generates a response (`message.refusal is None`) →
  synthesise `AnalystBrief` and return
- **INCORRECT path**: LLM emits `message.refusal` (Structured Outputs refusal),
  OR retrieval returns zero chunks → return canonical refusal brief
  (`"The corpus does not contain sufficient information..."`)

Both paths are logged with distinct `failure_reason` values for Grafana monitoring.

**Why LLM refusal, not an RRF threshold:**

An RRF threshold (`top_rrf < 0.020`) was built first and deleted after measurement
(Week 5 Step 2b). The threshold was catching only retriever-disagreement (top-1 chunk
ranked high by only one retriever), not corpus-relevance. Evidence:

- 5 positive queries (correct answer exists in corpus) were blocked by the threshold
- Only 3 negative queries (out-of-corpus) were correctly caught
- Threshold precision on negatives: 3/8 = 37.5%
- All 5 false-blocked queries had `hit@5=1` — retrieval found the right chunk,
  the threshold prevented the LLM from seeing it

The LLM's own refusal signal is more accurate than the RRF threshold on this corpus.
5 of 6 negatives that escaped the threshold were correctly refused by the LLM
(refusal_appropriateness = 5); only 1 fabricated.

### Decision: Pipeline-derived confidence removed

A confidence score (HIGH/MEDIUM/LOW) was planned as a user-facing signal.
Four signals were measured on 47 ground-truth queries:

| Signal | AUC vs Correctness ≥ 4 |
|---|---|
| top_rrf | 0.668 |
| citation_count | ~0.54 |
| retrieval_latency | ~0.51 (anti-correlated) |
| rrf_spread | ~0.51 |

Best single-signal AUC: **0.668** (95% CI overlapping random = 0.50). Three of four
signals were noise or anti-correlated with correctness. A multi-signal logistic
regression did not meaningfully improve over the best single signal at n=47.

**Decision:** Confidence removed from `AnalystBrief`, removed from logger, removed
from all user-facing copy. Every answer carries a standing "verify against sources"
caveat instead. Shipping a weak signal as a user promise (HIGH/MEDIUM/LOW) was
worse than shipping nothing — it would mislead analysts into over-trusting answers
that happened to be correct.

**Re-introduction conditions:** n ≥ 100 ground-truth queries, semantic_sim
(query↔top-1 chunk cosine) and doc_aware_margin added as signals, AUC ≥ 0.75
on held-out fold.

---

## 6. Evaluation Framework Decisions

### Decision: LLM-as-judge with a 4-dimensional rubric

A single aggregate score masks failure modes. CPIE uses four dimensions:

| Dimension | What it catches | Why separate |
|---|---|---|
| **Correctness** | Answer is factually right vs source | Core quality signal |
| **Faithfulness** | Answer is grounded in retrieved chunks | Hallucination detection |
| **Completeness** | All relevant aspects of the question addressed | Recall-side quality |
| **Refusal_appropriateness** | Refusals on out-of-corpus queries score well; false refusals on positives score badly | Corpus boundary enforcement |

**Judge model:** GPT-5.4-mini for regular monitoring (~5% same-model bias vs
GPT-5.4-mini synthesis, acceptable). GPT-5.4 for rigorous audits (flip
`JUDGE_MODEL` in `judge.py`; 5× cost, ~7× stricter).

**Why not ROUGE/BLEU:** Reference-free metrics that measure lexical overlap with a
reference answer do not detect faithful-but-wrong synthesis or hallucinated citations.
LLM-as-judge with explicit rubric criteria better reflects analyst utility.

### Decision: 52 QA pairs written before running the system

Ground truth QA pairs were written from genuine reading of source documents before
the system was run on any of them. The full dataset includes:

| Query type | Count | Purpose |
|---|---|---|
| Factual | 29 | Core coverage — does the system retrieve and synthesise correctly? |
| Numeric | 8 | Tests verbatim value extraction from tables and prose |
| Cross-document | 4 | Tests retrieval coverage across multiple sources |
| Negative (out-of-corpus) | 9 | Tests corpus boundary enforcement — should refuse |
| Summarisation | 2 | Tests multi-chunk synthesis |
| Table-only probes | 5 | Tests whether table chunks are retrievable |

**Why negatives are critical:** A system that refuses nothing gets a superficially
high Correctness score. The 9 out-of-corpus negatives force the system to demonstrate
that it knows what it does not know.

**Why write before running:** Running the system first reveals what it can answer;
writing ground truth afterward biases toward questions the system happens to get right.
All 52 pairs were written independently of system output.

### Decision: Judge strictness confound treated as a first-class measurement risk

During the Week 5 model comparison (Step 3b), changing the synthesis model and the
judge model simultaneously produced a misleading apparent regression. The judge
strictness effect (−0.615 Correctness) was 7× larger than the model quality effect
(+0.089 Correctness).

**Protocol adopted:** Any eval comparison that changes the synthesis model must hold
the judge constant, OR run a judge-only tiebreaker (change judge but not synth) to
isolate the strictness delta before interpreting cross-model scores.

This is documented in `judge.py` and in `model_selection.md` as a standing process
rule.

---

## 7. Query Rewriting — Built, Measured, and Removed

A query rewriting module (`src/synthesis/query_rewriter.py`, removed) was built
to address two failure modes: vocabulary mismatch (query uses different terminology
than the document) and underspecified queries.

**Implementation:** GPT-4o-mini, temperature=0, in-process cache, generated
paraphrase-style rewrites alongside the original query. Both variants were passed
to retrieval; their RRF scores were combined before final fusion.

**A/B measurement (52-query ground truth, gpt-5.4-mini judge):**

| Metric | Without rewriting | With rewriting | Delta |
|---|---|---|---|
| Correctness (overall) | 4.135 | 4.115 | −0.020 |
| Recall@5 | 0.907 | 0.857 | **−0.050** |
| Retrieval latency | 193ms | 585ms | **+3× latency** |
| Cross-doc Correctness | 3.50 | 2.75 | **−0.75 (target metric)** |

The cross-document target — the failure mode rewriting was meant to fix — regressed
by 0.75 Correctness points. Root cause: semantic-preserving paraphrases produced
rewrites that were semantically near-identical to the original. In RRF fusion, both
the original and the paraphrase voted for the same chunks, concentrating scores on
those chunks and reducing diversity. This is the opposite of what multi-query
retrieval should do.

**Decision:** Full removal. Same evidence discipline as reranker and confidence —
when the target fails to improve, do not ship.

**Re-introduction conditions:** (a) rewriter that produces *lexically different*
variants (HyDE-style: generate a hypothetical answer and embed that, or explicit
synonym expansion) rather than paraphrases that preserve keywords; (b) per-query-type
activation (only fire on underspecified / vocab_mismatch queries, which requires
a query classifier); (c) n ≥ 100 ground truth.

---

## 8. Monitoring and Observability Decisions

### Decision: Postgres + Grafana over Logfire

Logfire was the original planned monitoring stack. Replaced with Postgres + Grafana
to match LLM Zoomcamp Module 5 tooling. Both stacks solve the same problem; the
decision was course alignment and peer recognisability over technical superiority.

**Concrete benefit of Postgres:** The `cited_doc_ids` field (JSONB array) enables
the "Top-cited docs" Grafana panel (`jsonb_array_elements_text`) — a query that
answers "which sources does the system actually use?" This is difficult to express
in a trace-based tool like Logfire.

### Decision: Dual-write architecture — JSONL primary, Postgres secondary

Every query writes to two sinks:

1. **JSONL** (`logs/queries.jsonl`): append-only, never-fail, no external dependency.
   Primary sink. Used as the daily cost circuit breaker input (summing
   today's `cost_usd` entries).
2. **Postgres** (`cpie.query_logs`): feeds Grafana dashboards and joins to
   `user_feedback`. Secondary sink — never raises if Postgres is down.

**Why both:** JSONL survives Postgres being down (container restart, network
partition). Postgres enables SQL queries that JSONL cannot express efficiently
(time-series aggregations, joins to feedback table, JSONB unnesting). The never-raise
wrapper on all Postgres writes means Postgres being unavailable never breaks the
pipeline.

### Decision: Never-raise pattern for all monitoring writes

All monitoring writes (JSONL, Postgres) catch exceptions internally and log a
warning — they never propagate exceptions to the caller. The pipeline has one job:
return an `AnalystBrief`. Monitoring is instrumentation, not a pipeline step.

```python
try:
    db_insert_query_record(record)
except Exception as e:
    logger.warning("Postgres dual-write skipped: %s", e)
```

**Why this matters:** A Postgres connection pool failure at startup would otherwise
kill the first query in a fresh container. With never-raise, the system degrades
gracefully (JSONL only) rather than failing completely.

### Decision: User feedback stored in Postgres, not inferred from LLM scores

Thumbs up/down from the Streamlit widget is stored in `cpie.user_feedback` with a
`query_id` foreign key to `query_logs`. This is distinct from LLM-as-judge scores.

**Why separate:** LLM-as-judge evaluates correctness on ground truth queries in a
controlled eval run. User feedback captures real-world query satisfaction on live
traffic. They measure different things. Mixing them would produce a signal that
reflects neither accurately.

**FK integrity:** `user_feedback.query_id` references `query_logs.query_id`. A
bug was found during review: guardrail-triggered responses (query too long,
cost limit hit) returned `brief.model_dump()` without the `query_id` that was
written to Postgres. `app.py`'s `setdefault` generated a different UUID — feedback
clicks hit FK violations and failed silently. Fixed by explicitly adding `query_id`
to all guardrail return paths, consistent with the normal pipeline path.

---

## 9. Safety and Guardrails Decisions

### Decision: Four input guardrails required before public demo

1. **Query length limit (500 chars):** Defends against cost blow-up from LLM
   prompt injection via the query field. The limit is enforced before retrieval so
   no API call is made on a rejected query.

2. **Prompt injection line in system prompt:** The system prompt explicitly marks
   the user query as untrusted and instructs the model to ignore any instructions
   inside it. This is defence-in-depth against jailbreaks via the query field.

3. **Daily cost circuit breaker ($5):** `_daily_cost_so_far()` reads today's
   `cost_usd` entries from the JSONL log before each query. If cumulative spend
   exceeds $5, the query is refused with a distinct `failure_reason`. JSONL is used
   (not Postgres) because this check must work even when Postgres is down.

4. **OpenAI SDK resilience:** `timeout=30s`, `max_retries=2` on every API call.
   Defends against the tool/API timeout failure mode that would otherwise block the
   Streamlit UI indefinitely.

### Decision: No auth

The Streamlit app is anonymous and single-user. No session isolation, no user
accounts, no per-user rate limits.

**Why:** Auth adds scope (session management, credential storage, revocation) that
is disproportionate to a single-user demo. The corpus is not sensitive (all documents
are public). The daily cost circuit breaker caps the cost risk of an open demo.

**Condition for auth:** multi-tenant deployment with per-user query history,
per-user rate limits, or non-public corpus.

---

## 10. Infrastructure Decisions

### Decision: uv for dependency management

`uv` replaces pip for dependency resolution, lock file management, and virtual
environment creation. Resolves in milliseconds (vs seconds for pip), produces a
deterministic lock file, and has first-class support for multiple package indexes
(used to separate the CUDA torch dev wheel from the CPU wheel for containers).

### Decision: CPU torch wheel in containers, CUDA locally

Local development uses CUDA (RTX 4050) via the `pytorch-cu124` index. Container
builds use the CPU wheel via `--extra-index-url https://download.pytorch.org/whl/cpu`.
The `requirements.txt` generated for Docker strips CUDA packages and appends the
CPU override — the Dockerfile uses pip with that file rather than uv, keeping the
container build portable to CI environments without GPU access.

### Decision: Non-root container user

The Dockerfile creates a `cpie` user (UID 1000) and runs Streamlit as that user.
Running as root in a container is a security anti-pattern: if a path traversal or
command injection vulnerability exists, a root process has full container-filesystem
access. UID 1000 limits blast radius.

### Decision: data/raw and data/processed mounted as volumes, not baked into image

The 12 PDFs (`data/raw/`, ~99MB) and the processed indices (`data/processed/`, ~100MB
Chroma + pickle) are excluded from the Docker image and mounted at runtime as volumes.
Baking them in would make the image ~600MB larger, slow every rebuild, and embed
copyrighted (if public) documents in the image layer history.

---

## Summary Table: Decisions and Their Status

| Component | Status | Key evidence |
|---|---|---|
| Sliding window chunking 400/80 | Shipped as baseline | Validated Week 2 audit |
| Table heading injection (Tier 2) | Shipped | Manual audit — tables invisible without it |
| BAAI/bge-base-en-v1.5 | Shipped | 0.543 vs 0.278 mean cosine vs MiniLM |
| BM25 + Dense + RRF k=60 | Shipped | Hybrid consistently best on 3-query pilot |
| Institution metadata filter | Shipped | +1.0 cross-doc Completeness, +0.03 Recall@5 |
| GPT-5.4-mini synthesis | Shipped (upgraded from 4o-mini) | +0.089 Correctness, −26% latency (controlled A/B) |
| v2_numeric prompt | Shipped as default | Best aggregate Correctness, no regressions |
| CRAG correction layer (refusal branch) | Shipped | LLM refusal more accurate than RRF threshold |
| Dual-write JSONL + Postgres | Shipped | Never-raise, Grafana SQL analytics |
| Postgres + Grafana monitoring | Shipped | 8 panels, auto-provisioned |
| Streamlit feedback widget | Shipped | FK-linked to query_logs via query_id |
| RRF threshold short-circuit | **Removed** | 37.5% precision — blocked 5 positives to catch 3 negatives |
| Pipeline-derived confidence | **Removed** | AUC 0.668 — overlapping random on 47 queries |
| Reranker | **Not active** (code preserved) | 5.2× latency, zero aggregate Correctness gain |
| Query rewriting | **Removed** | Cross-doc Correctness −0.75, Recall@5 −5pp, 3× latency |
| v2_crossdoc prompt | In registry (not default) | Cross-doc target regressed −0.25; future per-type activation |
| Document-aware chunking | Deferred | No baseline to compare against yet |
| Query classification | Deferred | Required for per-type reranker + prompt activation |
| Retriever-agreement gate | Deferred | Correct replacement for deleted RRF threshold |
| Confidence layer | Deferred | Needs n≥100 GT + semantic_sim signal + AUC≥0.75 |
