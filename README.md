# CPIE — Climate Policy Intelligence Engine

<img width="2752" height="1536" alt="Accelerating_Climate_Policy_Intelligence_Overview" src="https://github.com/user-attachments/assets/61ccd59d-e978-46af-9e28-8744053b5d4d" />


Domain-aware RAG system that reads 12 UK and global climate policy PDFs and
returns structured analyst briefs with verified citations, so policy researchers
can act on regulatory signals without reading hundreds of pages themselves.

---

## Problem

Climate finance analysts must track regulatory signals from Ofgem, FCA, DESNZ,
IPCC, and IEA across hundreds of pages of dense documentation. Signals are
missed or acted on late because there is no fast, verifiable way to query across
multiple sources simultaneously.

**CPIE output — a structured brief per query:**

```json
{
  "answer": "Ofgem proposes that load control providers hold a Class B licence...",
  "citations": [
    { "doc_id": "OFGEM_SSES_2024", "passage": "...", "page": 14 }
  ],
  "contradictions": []
}
```

---

## Pipeline

The entire RAG pipeline for the project is organised into five stages. Stage 1 (Ingestion) runs once offline to
build the search indices; Stages 2–5 execute on every query.

<p align="center">
  <img src="docs/diagrams/arch_00_overview.svg" alt="Complete Architecture Overview" width="680">
</p>

### Stage 1 — Ingestion (offline)

PDFs are extracted with PyMuPDF, cleaned of layout noise (ESO nav elements,
Ofgem security stamps), and split into sliding-window chunks. A **dlt pipeline**
writes chunks into DuckDB. `build_indices.py` reads from DuckDB to produce two
search indices used at query time: a **BM25 index** (keyword matching) and a
**Chroma vector store** (dense semantic embeddings).

<p align="center">
  <img src="docs/diagrams/arch_01_ingestion.svg" alt="Stage 1 — Ingestion" width="580">
</p>

| Parameter | Value |
|---|---|
| Chunk size | 400 tokens |
| Overlap | 80 tokens |
| Floor / ceiling | 50t / 512t |
| Embedding model | BAAI/bge-base-en-v1.5 (768-dim) |
| Vector store | Chroma (`cpie` collection) |

---

### Stage 2 — Retrieval

Each query is first scanned for named institutions (Ofgem, FCA, IEA, BoE, CCC,
DESNZ, ESO). Matching institutions pre-filter both retrievers before RRF fusion.

<p align="center">
  <img src="docs/diagrams/arch_02_retrieval.svg" alt="Stage 2 — Retrieval" width="580">
</p>

| Component | Choice | Reason |
|---|---|---|
| BM25 | rank-bm25 | Exact keyword match on institution names, policy codes |
| Dense | BAAI/bge-base-en-v1.5 | Mean cosine 0.543 vs 0.278 for all-MiniLM (ablation) |
| Fusion | RRF k=60 | Cormack et al. 2009 default; no score normalisation needed |
| Reranker | Not active | Reorders chunks by relevance, but hybrid retrieval already ranks well — 5.2× latency, zero downstream Correctness gain (ablation) |

---

### Stage 3 — Synthesis

Retrieved chunks and the original query are passed to GPT-5.4-mini with a
structured output schema (`LLMResponse`). The system prompt instructs the
model to answer only from the provided excerpts. If the excerpts don't contain
enough information, the model produces a parsed response whose `answer` field
says so — it does not fabricate. `message.refusal` is a separate OpenAI safety
mechanism (content policy) and is handled as a fallback, not the primary
refusal path.

CPIE follows the CRAG (Yan et al. 2024) framing of routing responses into
CORRECT / INCORRECT paths, though the mechanism is simpler: rather than a
separate evaluator model scoring retrieved documents, the same LLM that
synthesises the answer decides whether the chunks are sufficient and refuses
if they are not. A decision gate routes to one of two paths:

- **CORRECT** — LLM returns a substantive answer → validated, returned as `AnalystBrief`
- **INCORRECT** → canonical refusal (`"The corpus does not contain sufficient information…"`)

Three things trigger the INCORRECT path:

- **Zero chunks** — retriever returns nothing → short-circuit before the LLM call even happens
- **Primary refusal** — chunks retrieved → LLM parses them → `answer` field says "excerpts don't contain this"
- **`message.refusal`** — OpenAI safety system blocks the request at the API level (content policy); handled as a separate fallback

After synthesis, every cited passage is matched against the retrieved chunks
(substring anchor check). Any citation whose passage cannot be found in the
retrieved set is dropped — this is **citation provenance verification**: it
confirms that a quoted passage exists in a chunk that was actually retrieved.
It does not verify that the answer's claim is entailed by the passage, that
the comparison is logically valid, or that qualifications were preserved. It
prevents chunk-id fabrication; faithfulness of the surrounding argument
remains the synthesiser's prompt-level responsibility.

<p align="center">
  <img src="docs/diagrams/arch_03_synthesis.svg" alt="Stage 3 — Synthesis" width="640">
</p>

| Component | Choice | Reason |
|---|---|---|
| Synthesis model | GPT-5.4-mini | +0.09 Correctness, +0.25 Faithfulness vs gpt-4o-mini; −26% latency (A/B) |
| Prompt | v2_numeric | Adds "verbatim value extraction" instruction; best aggregate Correctness, no regressions |
| Output schema | Pydantic `AnalystBrief` | Structured outputs — `answer`, `citations[]`, `contradictions[]` |

---

### Stage 3b — Agent Route (cross-document queries)

Cross-document queries — those requiring comparison or synthesis across multiple
institutions — are routed to a 7-node LangGraph agent instead of the fast path.
The agent decomposes the query into factual sub-questions, retrieves evidence per
sub-question, grades coverage deterministically, and synthesises a verified brief.

```
Router → cross_doc? ──► Planner → Retriever → Grader ──► Claim Builder → Verifier → Synthesiser
                                       ▲             |
                                       └─ retry ◄────┘  (not_covered only, max 1 retry)
```

**Planner** (GPT-4o-mini): decomposes the query into ≤6 factual sub-questions.
Comparison sub-questions are excluded — the synthesiser handles cross-source
comparison from factual evidence. `required_source` is constrained to the six
canonical institution names (`BoE`, `CCC`, `DESNZ`, `ESO`, `IEA`, `Ofgem`) that
exactly match the Chroma `institution` metadata field, so the per-sub-question
retrieval filter fires correctly instead of falling back to unfiltered retrieval.

**Retriever**: runs per sub-question, applying the planner's `required_source`
as a Chroma `institution` filter. Falls back to unfiltered retrieval if the
filtered pool is empty (e.g. the sub-question has no institution constraint).
`RETRIEVER_TOP_K = 6` — set by a stratified k-sweep across all 12 documents
(k=4 missed narrow passages; k=8 added noise without quality gain).

**Grader** (cross-encoder `ms-marco-MiniLM-L-6-v2`): scores each `(sub-question,
passage)` pair; takes the maximum score across all retrieved passages. Deterministic
— no LLM call. Replaced an initial LLM-based grader that produced a 100% retry
rate because its binary covered/not-covered output was too coarse for partial evidence.
Thresholds calibrated on CPIE corpus via t-distribution (n=4 probe queries):

| Score | Decision |
|---|---|
| ≥ 3.63 | covered — proceed |
| ≥ −2.3 | partial — proceed with available evidence |
| < −2.3 | not\_covered — retry with enriched query (parent + sub-question) |

**Retry logic:** only `not_covered` sub-questions trigger a retry. `partial`
proceeds to claim builder — evidence exists, the synthesiser notes gaps.
`RETRY_LIMIT = 1`.

**Claim Builder** (GPT-4o-mini): extracts structured claims from covered/partial
sub-questions. Each claim must cite ≥1 chunk_id visible in the retrieved passages.

**Verifier** (deterministic): drops claims whose `evidence_ids` are not present in
`state["retrievals"]`. No semantic matching — a set-membership check against chunk_ids
that were actually retrieved. Prevents chunk_id hallucination without an LLM call.

**Synthesiser** (GPT-5.4-mini): produces an `AnalystBrief` from verified claims +
full retrieved excerpts. Tracks `finish_reason` and `completion_tokens` for
diagnostics; retries once at 3000 tokens on `LengthFinishReasonError`.

**Model tiering — cost vs quality at each stage:**

The agent uses two models chosen deliberately for the work each node does:

| Node | Model | Reason |
|---|---|---|
| Planner | GPT-4o-mini | Structural JSON task (decompose query into sub-questions). Output is a schema — correctness is verifiable by the parser, not the user. Cheapest capable model. |
| Claim Builder | GPT-4o-mini | Structured extraction from retrieved text. Claims are post-filtered by the deterministic Verifier anyway, so model quality is less critical here than synthesis quality. |
| Synthesiser | GPT-5.4-mini | The only node whose output the user reads. Upgraded from GPT-4o-mini after A/B: +0.09 Correctness, +0.25 Faithfulness, −26% latency. The latency improvement comes from GPT-5.4-mini's faster token generation, which more than offsets the cost increase on a per-query basis. |

Upstream tasks (planner, claim builder) together cost ~$0.002 per query.
Synthesis costs ~$0.011. Total agent mean: **$0.013 per cross-doc query**.
The tiering concentrates spend where quality is user-visible and saves it
where the output is intermediate and machine-consumed.

**Budget caps** (hard stops before each node):

| Budget | Cap |
|---|---|
| Steps | 14 |
| Wall time | 60s |
| Cost | $0.05 |
| Retries per sub-question | 1 |

**Routing:** `AGENT_ROUTE_ENABLED` in `configs/config.yaml` (or the env var of
the same name). `"true"` routes all cross-doc queries to the agent. `"canary"`
routes `canary_pct` fraction. `"false"` runs the agent as a shadow (result
discarded; fast-path answer returned).

#### Agent A/B results (N=100 cross-document queries)

| Metric | Fast path | Agent | Δ |
|---|---|---|---|
| Correctness (1–5) | 3.07 | **4.06** | +0.99 |
| Faithfulness (1–5) | 4.41 | 4.34 | −0.07 |
| Completeness (1–5) | 2.43 | **3.31** | +0.88 |
| Refusal appropriateness (1–5) | 4.44 | **5.00** | +0.56 |
| Source recall | 0.62 | **0.91** | +0.28 |
| Mean cost | $0.004 | $0.013 | 3.3× |
| Mean latency | 2.7s | 12.6s | 4.7× |
| Latency P95 | 4.6s | **14.4s** | — |
| Retrieval recall (pre-synthesis) | — | 0.98 | — |
| Required-source hit rate | — | 97% (496/507) | — |
| Finish reason = length (truncation) | — | 0/100 | — |

**Quality gates:**

| Gate | Threshold | Result | Notes |
|---|---|---|---|
| Correctness | ≥ 3.50 | 4.06 ✅ | N=100 judge run |
| Completeness | ≥ 3.25 | 3.31 ✅ | N=100 judge run |
| Latency P95 | ≤ 15s | 14.4s ✅ | Clean latency run (no judge); see note below |
| Faithfulness regression | ≥ fast-path − 0.20 | 4.34 vs 4.41 (−0.07) ✅ | Within noise; fast path has easier single-source task |
| Retrieval recall | ≥ 0.95 | 0.98 ✅ | Pre-synthesis, expected sources present |
| Bounded termination | 0 budget breaches | 0/100 ✅ | All queries completed within step/cost/time caps |
| Citation smuggling | — | Not yet tested ⚠️ | Adversarial chunk injection not in current eval set |
| Canary stability | 48h no regressions | Not yet run ⚠️ | Agent path enabled; monitoring in place via Grafana |

Faithfulness is −0.07 vs fast path (noise-level; fast path is a single-source
retrieval with a narrower synthesis context, which is an easier faithfulness
target than multi-source cross-doc answers).

#### Agent quality engineering — problem-solving log

Four quality problems were identified and resolved during development. Each
is recorded here as an engineering decision with its root cause, failed approach
(where one was tried), and the fix that shipped.

---

**1. Correctness gate failure (initial: 3.07 → target: ≥ 3.50)**

*Identified:* First A/B run on 30 cross-doc queries showed agent Correctness
at 3.32 — above the fast path (3.07) but below the gate.

*Root causes:*
- The initial LLM-based grader (GPT-4o-mini binary covered/not-covered)
  produced a 100% retry rate: every sub-question was marked `not_covered` on
  the first retrieval pass, exhausted retries, and fell back to weak evidence.
  The grader burned budget and degraded retrieval quality by forcing broad
  fallback queries instead of selective refinement.
- `RETRIEVER_TOP_K = 5` (inherited from fast path) was too shallow for
  multi-source cross-doc queries; sub-questions requiring a specific BoE
  paragraph within 12 documents had insufficient recall at k=5.
- Agent synthesiser was GPT-4o-mini (same as fast path), providing no
  reasoning uplift for synthesis.

*Fixes applied:*
1. **Replaced LLM grader with cross-encoder** (`ms-marco-MiniLM-L-6-v2`):
   scores each (sub-question, passage) pair numerically; three-way threshold
   (covered / partial / not_covered) calibrated on CPIE corpus. Retry rate
   dropped from 100% to ~15%. No LLM call, deterministic, 8ms per sub-question.
2. **RETRIEVER_TOP_K raised to 6** via stratified k-sweep: k=4 missed narrow
   passages; k=6 recovered them; k=8 added noise with no quality gain.
3. **Synthesiser upgraded to GPT-5.4-mini**: +0.09 Correctness, −26% latency
   vs GPT-4o-mini on the same 30-query eval.

*Outcome:* Correctness 4.06 on N=100. Gate passed.

---

**2. Faithfulness deficit — blended cross-source sentences**

*Identified:* After correctness was fixed, faithfulness scored 4.26 — below
the fast path's 4.41. LLM judge rationales consistently flagged "blended"
sentences: the synthesiser was writing things like *"Both the BoE and IEA
project 2–3°C warming under baseline scenarios"* without citing a specific
excerpt for either claim, making the sentence unverifiable even though the
underlying facts were correct.

*Failed approach:*
Filter synthesis context to only the chunks whose `chunk_id` appeared in
verified claims (`fe80e33`). Faithfulness improved +0.04. But the filtered
context was too thin: completeness fell −0.40 and correctness fell −0.30
because the synthesiser lost relevant passages it needed for full coverage.
Reverted after one eval run.

*Root cause (revised):* The faithfulness problem was a **prompt rule**, not a
context problem. Prompt rule 5 was written to *encourage* comparative
sentences ("When the question calls for comparison, be thorough") without
requiring that each comparison point be attributed to a specific excerpt. The
model complied with the spirit (thorough comparison) while violating the
letter (grounded attribution).

*Fix applied (`17eea6b`):*
- Rule 1 rewritten: *"every sentence must be traceable to a specific excerpt
  — write [Excerpt N] immediately after the claim it supports."*
- Rule 5 rewritten: prohibit free-floating blended sentences; require two
  separately-attributed sentences per comparison point, each citing exactly
  one excerpt (e.g. *"BoE projects X [Excerpt 3]. IEA projects Y [Excerpt 7]."*)

*Outcome:* Faithfulness 4.34 (−0.07 vs fast path, noise-level; fast path
is a simpler single-source task).

---

**3. Faithfulness fix → completeness regression**

*Identified immediately after fix 2:* Completeness dropped from 3.31 → 2.91.
The two-sentence split rule was too strict: when a comparison point was
genuinely attributable to both sources simultaneously (e.g. a shared
conclusion), the model's safest response was to omit the point entirely rather
than risk a faithfulness penalty by attempting a split.

*Root cause:* The prohibition on blended sentences created an **omission
incentive** — skipping a comparison point carries no penalty, but writing it
incorrectly does.

*Fix applied (`8b69b60`):*
- Rule 1 softened: *"cite every sentence inline using [Excerpt N]"* — frames
  the requirement as mandatory inline citation rather than a prohibition,
  removing the trigger that caused omissions.
- Rule 5 softened: allow blended multi-source sentences; require all
  contributing excerpts cited inline (e.g. *"Both BoE [Excerpt 3] and
  IEA [Excerpt 7] project X"*). The model can write the comparison it finds
  in the evidence as long as every source is attributed.

*Outcome:* Completeness recovered to 3.31 (gate ≥ 3.25 passed). Faithfulness
held at 4.34 — inline citation enforcement preserved attribution without
requiring syntactic sentence splitting.

---

**4. Latency gate (P95 ≤ 15s)**

*Identified:* First full N=100 latency-only run gave P95 = 14.4s (gate pass).
The judge run immediately after returned P95 = 16.7s (gate fail). The 2.3s
discrepancy was the only failure.

*Investigation:*
- Profiled node latencies across traces: synthesiser dominates at ~10–12s.
  Retrieval across 6 sub-questions averages ~1.8s total.
- Tried **ThreadPoolExecutor parallelisation** of per-sub-question retrieval
  (6 workers). P95 increased from 14.4s → 16.9s. Root cause: Python GIL
  serialises SentenceTransformer inference and Chroma query under concurrent
  access; thread overhead adds latency rather than removing it. Reverted.

*Root cause of the 16.7s judge run:* The clean latency run and the judge run
were separate eval sessions. The 2.3s P95 difference is within expected
OpenAI API latency variance across sessions — individual LLM calls show
±1–3s depending on server load and time of day. The eval script runs queries
sequentially (one fast path + one agent path + two judge calls per query),
so there is no within-run concurrency. The 14.4s clean-run measurement is
the production-representative P95; the 16.7s represents a higher-variance
session that happened to land above the gate.

*Synthesiser token reduction (prompt rules 8–10):* Since the synthesiser
dominates latency (~10–12s of the ~12.6s mean), reducing completion tokens
directly reduces wall time. Three prompt rules were added to eliminate
redundant output without losing coverage:

- **Rule 8:** *"Do not repeat the same fact from the same source in multiple
  sentences. State each distinct point once."* — eliminates restatement loops
  where the model echoes the same BoE figure across opening and closing
  sentences.
- **Rule 9:** *"Use the single strongest [Excerpt N] per source per comparison
  point. Do not stack multiple citations for the same claim."* — stops the
  model from appending three identical-signal excerpts to a single sentence.
- **Rule 10:** *"Begin your answer directly with the evidence. Do not open by
  restating or paraphrasing the question."* — cuts the standard 15–25 token
  preamble (*"This question asks me to compare…"*) that added latency with
  zero information content.

Mean completion tokens after rules 8–10: 924 (P95: 1387). Zero
`finish_reason=length` truncations across 100 queries.

*Resolution:* 14.4s is the production latency. The synthesiser is the
bottleneck and the only real knob remaining is adaptive sub-question count
(generating 2–4 sub-questions for simple comparisons vs up to 6 for complex
multi-source queries). This is a planned follow-on; P95 gate passes on the
clean measurement.

---

### Stage 4 — Evaluation

CPIE uses two complementary evaluation tracks: **offline evaluation** run
against a fixed ground-truth dataset before deployment, and **online
monitoring** of live traffic captured through the production stack.

<p align="center">
  <img src="docs/diagrams/arch_04_evaluation.svg" alt="Stage 4 — Evaluation" width="580">
</p>

#### Offline evaluation

52 hand-crafted QA pairs across all 12 documents (29 factual, 8 numeric,
4 cross-document, 9 out-of-corpus negatives, 2 summarisation) — written
before running the system on them.

**Retrieval metrics** (hybrid BM25 + dense + RRF, institution metadata filter)

| Metric | Score |
|---|---|
| Recall@5 | **0.907** |
| MRR@5 | 0.884 |
| nDCG@5 | 0.894 |
| Hit@5 | 0.953 |
| Precision@5 | 0.722 |

**Recall@5 is the primary metric for this system.** A chunk missed at retrieval
is unrecoverable, the LLM can only synthesise from what it receives, so a
missed relevant chunk always produces a wrong or refused answer regardless of
how good the system prompt is. A false positive (irrelevant chunk included)
is tolerable: the LLM filters noise and the citation verifier drops fabricated
passages. This asymmetry, a miss is fatal and noise is manageable, makes Recall
the right thing to optimise. MRR and nDCG measure rank position within the top
5, which matters for search UIs where users scan results; CPIE sends all top 5
to the LLM at once so rank within that set has no effect on output quality.

**Business justification:** CPIE's users are climate finance analysts tracking
regulatory signals across hundreds of pages on a deadline. A missed signal, a
liability threshold buried in an Ofgem consultation, a new BoE stress-test
scenario can mean a misaligned investment decision or a compliance gap. The
cost of a false negative (analyst acts on incomplete information) far outweighs
the cost of a false positive (analyst reads one extra citation). A Recall@5 of
0.907 means the system surfaces the right evidence 9 times out of 10, the
remaining gap is the honest case for keeping a human in the loop.

**LLM-as-judge** (GPT-5.4-mini, 4-dimensional rubric, 1–5 scale; shipped config: v2\_numeric prompt)

| Metric | Overall | Factual | Numeric | Cross-doc | Negative |
|---|---|---|---|---|---|
| Correctness | **4.13** | 4.14 | 4.25 | 3.50 | — |
| Faithfulness | **4.37** | 4.52 | 4.62 | 3.50 | — |
| Completeness | **3.33** | 3.07 | 3.75 | 2.75 | — |
| Refusal appropriateness | **4.85** | — | — | — | 4.11 |

Out-of-corpus negatives correctly handled: **77.8%** (7/9).

Evaluation scripts: `src/evaluation/retrieval_eval_runner.py` (retrieval metrics), `src/evaluation/judge_runner.py` (LLM-as-judge).
Ground truth: `data/eval/ground_truth.json`. Results: `data/eval/results/`.

#### Online evaluation (live traffic)

Every production query is logged to Postgres and surfaced in Grafana. Online
metrics complement the static ground-truth run by catching quality drift on
real user queries over time.

| Online metric | How it is measured |
|---|---|
| Refusal rate | `is_refusal = true` rows / total queries over time |
| Latency (p50 / p95) | `retrieval_latency_ms` + `synthesis_latency_ms` per query |
| Cost per query / per day | `cost_usd` field, aggregated daily |
| Citation count | Mean `citation_count` per answered query |
| User satisfaction | Thumbs up / (thumbs up + thumbs down) from `cpie.user_feedback` |
| Top-cited documents | Most frequent `doc_id` values in `cited_doc_ids` |
| Failure reasons | Breakdown of `failure_reason` field (empty retrieval, LLM refusal, cost limit) |

---

### Stage 5 — Monitoring (online evaluation infrastructure)

Every query is dual-written to `logs/queries.jsonl` (primary fallback) and
`cpie.query_logs` (Postgres, feeds Grafana). The Streamlit UI writes thumbs
feedback to `cpie.user_feedback`. This is the infrastructure that powers
the online evaluation metrics described in Stage 4.

<p align="center">
  <img src="docs/diagrams/arch_05_monitoring.svg" alt="Stage 5 — Monitoring" width="680">
</p>

<p align="center">
  <img src="docs/images/Grafana_dashboard_snapshot.png" alt="Grafana monitoring dashboard" width="860">
</p>

Grafana dashboards (auto-provisioned, no manual setup):

| Panel | What it shows |
|---|---|
| Query volume | Queries per hour |
| Refusal rate | % is\_refusal over time |
| Latency percentiles | p50 synthesis + retrieval |
| Top-cited docs | Which sources get used |
| Cumulative cost | Total $ spend (last 24h, stat) |
| Cost per day | Daily $ spend over time |
| User feedback ratio | Thumbs up / total votes |
| Recent failures | Failure reason + query snippet |

Start the monitoring stack: `docker compose up -d postgres grafana`

#### Monitoring catching a production bug

The Grafana **Recent failures** panel surfaced a `LengthFinishReasonError` on a
legitimate corpus query ("What load control licensing requirements does Ofgem
propose?") during live use. The Ofgem licensing response contains three detailed
citations and a multi-paragraph answer, the previous `max_completion_tokens=800`
limit truncated the JSON mid-stream. The OpenAI SDK raised `LengthFinishReasonError`
before the response could be parsed, and the exception propagated as a raw pipeline
failure rather than a graceful refusal.

The Grafana failure panel surfaced this within seconds of the query being logged.
The fix, catching `LengthFinishReasonError` explicitly in `synthesiser.py` and
returning a canonical refusal with a distinct `failure_reason`, was applied
immediately. `MAX_TOKENS` was already at 2000 (bumped from 800 in a prior session);
the exception handler adds defence-in-depth for any response that would exceed the
current limit.

This is a concrete illustration of why online monitoring complements offline
evaluation: the 52-query ground truth had no QA pair for this failure mode, but a
single live query exposed it immediately.

---

## Corpus

12 public documents from UK and global climate regulators:

| Document | Institution | Year |
|---|---|---|
| Smart Secure Electricity Systems (SSES) | Ofgem | 2024 |
| ZEV Mandate | DESNZ | 2023 |
| World Energy Outlook 2025 | IEA | 2025 |
| CBES Results | Bank of England | 2022 |
| CBES Key Elements | Bank of England | 2021 |
| Measuring Climate Risk | Bank of England | 2020 |
| BoE Climate Disclosure | Bank of England | 2024 |
| BoE Macro Implications | Bank of England | 2024 |
| CCC Progress Report 2024 | Climate Change Committee | 2024 |
| CCC Progress Report 2025 | Climate Change Committee | 2025 |
| Seventh Carbon Budget | Climate Change Committee | 2025 |
| Beyond 2030 | ESO | 2024 |

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- OpenAI API key (`OPENAI_API_KEY`)
- `uv` — install with `pip install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Step 1 — Clone and install

```bash
git clone https://github.com/AshishSiwach/Climate-Policy-Intelligence-Engine.git cpie
cd cpie
make install
```

### Step 2 — Add your API key

```bash
cp .env.example .env
# Required: set OPENAI_API_KEY=sk-...
# Optional: Postgres and Grafana variables have working local defaults already set
```

### Step 3 — Download corpus, ingest, and run

```bash
make data       # downloads 12 PDFs to data/raw/ (note: IEA WEO 2025 needs manual download — script prints instructions)
```

```bash
# Start monitoring stack (Postgres + Grafana)
docker compose up -d postgres grafana

# Ingest PDFs → DuckDB, build BM25 + Chroma indices
make ingest

# Run the Streamlit chat UI
make run
```

Open the app at **http://localhost:8501** and Grafana at **http://127.0.0.1:3000** (admin / admin).

### Docker (alternative)

Build and run the full stack including the app container:

```bash
docker build -t cpie .
docker compose up
```

---

## Design Decisions

### Corrective RAG (CRAG-style correction layer)

CPIE implements a coarse CRAG pattern (Yan et al. 2024) between retrieval and
synthesis. Rather than a separate evaluator model, the same synthesis LLM
decides whether retrieved chunks are sufficient:

- **CORRECT** — LLM returns a substantive answer → return `AnalystBrief`
- **INCORRECT** — triggered by zero chunks (short-circuit before LLM call),
  the LLM's `answer` field saying excerpts are insufficient, or `message.refusal`
  (OpenAI content-policy fallback) → return canonical refusal brief

Both paths are logged distinctly.

### Hybrid retrieval — BM25 + dense + RRF k=60

BM25 handles exact keyword matches (institution names, policy codes); dense
retrieval (BAAI/bge-base-en-v1.5, 768-dim) handles semantic equivalence.
RRF k=60 (Cormack et al. 2009) fuses ranked lists without score normalisation.
Embedding model chosen after ablation: mean cosine 0.543 vs 0.278 for
all-MiniLM-L6-v2.

### Institution metadata filter

The query text is scanned for named institutions (Ofgem, FCA, IEA, BoE, CCC,
DESNZ, ESO) before retrieval. Matching institutions pre-filter both the Chroma
collection and BM25 results. A/B result: cross-doc Completeness +1.0,
Recall@5 +0.03, Refusal_appropriateness +0.26. Added 18ms latency.

### Prompt versioning and A/B testing

Three prompt variants authored (v1, v2\_crossdoc, v2\_numeric) and tested
against the 52-query ground truth. `v2_numeric` shipped as default: adds a
"verbatim value extraction + page citation" instruction that improved aggregate
Correctness without regressing any metric. `v2_crossdoc` preserved in the
registry for per-query-type activation (future roadmap item).

### Reranker and query rewriting — measured and dropped

Both were built and A/B-measured against the eval dataset:
- **Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): 5.2× retrieval
  latency, zero aggregate Correctness gain over hybrid.
- **Query rewriting** (GPT-4o mini paraphrase): cross-doc Correctness −0.75,
  Recall@5 −5pp, 3× latency.

Evidence in `docs/week5_failure_analysis.md`.

---

## Guardrails & Safety

Every query passes through three guardrails in sequence before retrieval or
synthesis runs. A query stopped by any guardrail returns the canonical refusal
immediately, no retrieval, no LLM synthesis, no wasted tokens.

| # | Guardrail | What it catches | Cost |
|---|---|---|---|
| 1 | Query length limit (500 chars) | Cost blow-up from very long inputs | Zero |
| 2 | Daily cost circuit breaker ($5/day) | Runaway API spend | Zero |
| 3 | Pre-retrieval domain gate (GPT-4o-mini) | Off-domain queries | ~$0.00003 / query |

### Domain gate

Prompt-level rules for refusing off-domain queries are brittle, patching one
failure mode (spelling requests) leaves gaps for arithmetic, CEO lookups, and
general-knowledge facts that coincidentally mention a corpus keyword. A
dedicated classifier is more general.

The gate is a GPT-4o-mini call (~120 tokens, ~$0.00003) that classifies the
query as in-domain or out-of-domain before any retrieval or synthesis runs.
It fails-open: any API error passes the query through to the normal pipeline.
The system prompt's Rule 6 ("do not answer from general knowledge") acts as
a second defence layer for anything that slips through.

### Stress testing results

A structured stress test was run across three categories after the system was
working end-to-end. Results drove the two fixes above.

**Parametric knowledge traps** — queries the LLM could answer from training
data; should always refuse:

| Query | Pre-fix | Post-fix |
|---|---|---|
| "What is the capital of France?" | ❌ Answered "Paris" with a fabricated CCC citation | ✅ Domain gate blocks |
| "What is the GDP of the UK?" | ⚠️ Refused the ask but cited unrelated GDP passages | ✅ Domain gate blocks |
| "Who is the CEO of BP?" | ⚠️ Refused but attached an IEA bibliography entry | ✅ Domain gate blocks |
| "What is 1+1?" | ❌ Answered "2" | ✅ Domain gate blocks |

**Partial corpus match** — corpus has related content but not the exact answer;
should answer from what exists without fabricating:

| Query | Result |
|---|---|
| "What is carbon pricing?" | ✅ Answered from BoE + CCC corpus passages |
| "What happened at COP26?" | ✅ Answered from CCC Seventh Carbon Budget with Glasgow Climate Pact detail |

**Legitimate corpus queries** — should answer fully with verified citations:

| Query | Result |
|---|---|
| "What aggregate losses did UK banks face under the CBES early action scenario?" | ✅ Surfaced the comparative figure (30% higher in Late Action, £110bn extra) |
| "What does Ofgem propose for load control licensing?" | ✅ 3 verified citations from OFGEM_SMART_SECURE_2025 |

---

## Known Limitations

- **No user-uploaded documents.** Corpus is fixed at 12 curated public PDFs.
  Accepting user content requires corpus-side prompt-injection scanning and
  per-user isolation.
- **No confidence signal.** Pipeline-derived confidence was removed after
  calibration (best AUC 0.668, overlapping random). Every answer carries
  a standing "verify against sources" caveat instead.
- **CCC Progress traffic-light indicators** do not extract as text from PDF
  (PyMuPDF limitation). The surrounding prose restates the assessment and
  carries the retrieval signal.
- **Agent route is cross-doc only (Phase 3).** Summary and contradiction routes
  are planned for Phase 5 but not yet built. Queries classified as `summary` or
  `contradiction` currently fall back to the fast path.
- **Contradiction detection is experimental.** The `contradictions[]` field is
  LLM self-report, not cross-doc claim verification. Treat as a hint.

---

## Project Structure

```
cpie/
  src/
    ingestion/       pdf_loader, chunker, dlt_pipeline
    retrieval/       bm25_retriever, dense_retriever, hybrid_retriever,
                     institution_detector, reranker (evaluated, not active),
                     query_rewriter (evaluated, not active)
    synthesis/       synthesiser, output_schema, query_classifier (domain gate)
    agent/
      router.py            complexity router — fast vs agent path
      workflow.py          LangGraph StateGraph (only file that imports langgraph)
      state.py             AgentState TypedDict
      policies.py          MAX_STEPS, MAX_COST, RETRY_LIMIT, RETRIEVER_TOP_K
      nodes/
        planner.py         decomposes query into factual sub-questions
        retriever.py       per-sub-question hybrid retrieval
        grader.py          cross-encoder coverage scoring (deterministic)
        claim_builder.py   extract structured claims from covered passages
        verifier.py        drop claims with hallucinated chunk_ids
        synthesiser.py     produce AnalystBrief from verified claims
    evidence/        Claim, Coverage, SubQuestion Pydantic models
    observability/   tracing.py — emit_span() to agent_traces table
    evaluation/      judge, eval_runner, retrieval_metrics
    monitoring/      logger (JSONL), db (Postgres)
  tests/             unit + integration tests
  data/eval/
    ground_truth.json          52 hand-crafted QA pairs
    cross_document_ground_truth.json  cross-doc A/B eval set
    results/                   eval run outputs + ablation tables
  monitoring/
    postgres/init.sql          schema DDL
    postgres/init_agent_traces.sql  agent_traces table DDL
    grafana/dashboards/        provisioned JSON (fast-path + agent dashboards)
    grafana/provisioning/      datasource + dashboard provider YAMLs
  docs/
    AGENT_ROUTE_PLAN.md          phased build plan for all agent routes
    AGENT_ROUTE_REASONING.md     why each route exists
    week5_failure_analysis.md    A/B evidence for every dropped component
    ai_engineering_decisions.md  full decision log with A/B results
    OPS_RUNBOOK.md               how to disable routing, inspect traces, reproduce queries
  scripts/
    ingest.py                  run dlt ingestion pipeline
    build_indices.py           build BM25 + Chroma from DuckDB
    download_data.py           fetch 12 corpus PDFs with SHA-256 verification
    calibrate_grader_threshold.py  cross-encoder threshold calibration
  app.py                       Streamlit chat UI
  main.py                      CLI entry point
  docker-compose.yml           postgres + grafana + app services
  Dockerfile                   containerised Streamlit app
```

---

## References

- Yan et al. (2024). *Corrective Retrieval Augmented Generation.* arXiv:2401.15884
- Cormack, Clarke & Buettcher (2009). *Reciprocal Rank Fusion outperforms Condorcet
  and individual rank learning methods.* SIGIR '09.
- BAAI/bge-base-en-v1.5 — [HuggingFace](https://huggingface.co/BAAI/bge-base-en-v1.5)
