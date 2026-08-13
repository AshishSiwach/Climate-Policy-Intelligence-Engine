# CPIE — Climate Policy Intelligence Engine

Domain-aware RAG system that reads 12 UK and global climate policy PDFs and
returns structured analyst briefs with verified citations, so policy researchers
can act on regulatory signals without reading hundreds of pages themselves.

> **Disclaimer:** CPIE is an assistant, not a source of truth. Verify every
> citation against the source document before relying on it. CPIE does not
> provide investment, legal, or regulatory advice.

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

## Architecture

```
User Query
    │
    ▼
[Institution Detector] ─── detects Ofgem / FCA / IEA / BoE / CCC in query
    │                       pre-filters retrieval to named institutions
    │
    ├────────────────────────┐
    ▼                        ▼
[BM25 Retriever]    [Dense Retriever]
(rank-bm25)         BAAI/bge-base-en-v1.5 → Chroma
    │                        │
    └────────────────────────┘
                 │
                 ▼
         [RRF Fusion k=60]   ← Hybrid: keyword + semantic
                 │
                 ▼
           Top-5 Chunks
                 │
                 ▼
    [Synthesiser — GPT-5.4-mini]
    Structured Outputs (Pydantic)
    CRAG correction layer:
      • LLM does not refuse → return AnalystBrief
      • LLM refuses or empty retrieval → canonical refusal
                 │
                 ▼
    [Citation Verifier]  ← every cited passage checked against retrieved chunks
                 │
                 ▼
         AnalystBrief
    {answer, citations[], contradictions[]}
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 [JSONL Logger]    [Postgres Writer]
 (fallback sink)   (Grafana dashboards
                    + Streamlit feedback)
```

**Ingestion (one-time, local):**

```
data/raw/ PDFs → PyMuPDF → Chunker (400t / 80t overlap) → dlt → DuckDB
                                                                    │
                                                                    ▼
                                                        build_indices.py
                                                        BM25 (.pkl) + Chroma
```

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
git clone https://github.com/ashishsiwach2789/cpie.git
cd cpie
make install
```

### Step 2 — Add your API key

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
```

### Step 3 — Ingest PDFs and run

Copy the 12 corpus PDFs into `data/raw/`, then:

```bash
# Start monitoring stack (Postgres + Grafana)
docker compose up -d postgres grafana

# Ingest PDFs → DuckDB, build BM25 + Chroma indices
uv run python scripts/ingest.py
uv run python scripts/build_indices.py

# Run the Streamlit chat UI
make run
```

Open the app at **http://localhost:8501** and Grafana at **http://localhost:3000** (admin / admin).

### Docker (alternative)

Build and run the full stack including the app container:

```bash
docker build -t cpie .
docker compose up
```

---

## Evaluation Results

Evaluated on 52 hand-crafted QA pairs across all 12 documents (29 factual,
8 numeric, 4 cross-document, 9 out-of-corpus negatives, 2 summarisation).
LLM-as-judge: GPT-5.4-mini, 4-dimensional rubric (1–5 scale).

### Retrieval (hybrid BM25 + dense + RRF, with institution metadata filter)

| Metric | Score |
|---|---|
| Recall@5 | **0.907** |
| MRR@5 | 0.884 |
| nDCG@5 | 0.894 |
| Hit@5 | 0.953 |
| Precision@5 | 0.722 |

### LLM-as-judge (52 queries, shipped config: v2\_numeric prompt + gpt-5.4-mini)

| Metric | Overall | Factual | Numeric | Cross-doc | Negative |
|---|---|---|---|---|---|
| Correctness | **4.13** | 4.14 | 4.25 | 3.50 | — |
| Faithfulness | **4.37** | 4.52 | 4.62 | 3.50 | — |
| Completeness | **3.33** | 3.07 | 3.75 | 2.75 | — |
| Refusal appropriateness | **4.85** | — | — | — | 4.11 |

Out-of-corpus negatives correctly handled: **77.8%** (7/9).

---

## Design Decisions

### Corrective RAG (CRAG-style correction layer)

CPIE implements a coarse CRAG pattern (Yan et al. 2024) between retrieval and
synthesis:

- **CORRECT** — LLM does not refuse → synthesise and return `AnalystBrief`
- **INCORRECT** — LLM emits `message.refusal`, or retrieval returns zero chunks
  → return canonical refusal brief (`"The corpus does not contain sufficient
  information…"`)

Both paths are logged distinctly. Deferred to v2: retriever-agreement gate
(replaces the deleted RRF-threshold short-circuit) and confidence layer
(removed after Week 5 calibration showed AUC 0.668 — too weak to promise to
users).

### Hybrid retrieval — BM25 + dense + RRF k=60

BM25 handles exact keyword matches (institution names, policy codes); dense
retrieval (BAAI/bge-base-en-v1.5, 768-dim) handles semantic equivalence.
RRF k=60 (Cormack et al. 2009) fuses ranked lists without score normalisation.
Embedding model chosen after Week 2 ablation: mean cosine 0.543 vs 0.278 for
all-MiniLM-L6-v2.

### Institution metadata filter (promoted from v2)

The query text is scanned for named institutions (Ofgem, FCA, IEA, BoE, CCC,
DESNZ, ESO) before retrieval. Matching institutions pre-filter both the Chroma
collection and BM25 results. Week 5 A/B: cross-doc Completeness +1.0,
Recall@5 +0.03, Refusal_appropriateness +0.26. Added 18ms latency.

### Prompt versioning and A/B testing

Three prompt variants authored (v1, v2\_crossdoc, v2\_numeric) and tested
against the 52-query ground truth. `v2_numeric` shipped as default: adds a
"verbatim value extraction + page citation" instruction that improved aggregate
Correctness without regressing any metric. `v2_crossdoc` preserved in the
registry for per-query-type activation (v2 roadmap).

### Reranker and query rewriting — measured and dropped

Both were built and A/B-measured against the eval dataset:
- **Reranker** (`cross-encoder/ms-marco-MiniLM-L-6-v2`): 5.2× retrieval
  latency, zero aggregate Correctness gain over hybrid. Per-query-type
  activation on numeric queries is a v2 candidate.
- **Query rewriting** (GPT-4o mini paraphrase): cross-doc Correctness −0.75,
  Recall@5 −5pp, 3× latency. Semantically-preserving paraphrases concentrated
  RRF votes on the same chunks, hurting diversity. v2 re-introduction requires
  HyDE-style rewrites.

Evidence in `docs/week5_failure_analysis.md`.

---

## Known Limitations

- **No user-uploaded documents.** Corpus is fixed at 12 curated public PDFs.
  Accepting user content requires corpus-side prompt-injection scanning and
  per-user isolation (v3 scope).
- **No confidence signal in v1.** Pipeline-derived confidence was removed after
  Week 5 calibration (best AUC 0.668, overlapping random). Every answer carries
  a standing "verify against sources" caveat instead.
- **CCC Progress traffic-light indicators** do not extract as text from PDF
  (PyMuPDF limitation). The surrounding prose restates the assessment and
  carries the retrieval signal.
- **Single-turn RAG.** v1 is one retrieval pass + one LLM call. Query
  decomposition, agentic loops, and iterative retrieval are v2.
- **Contradiction detection is experimental.** The `contradictions[]` field is
  LLM self-report, not cross-doc claim verification. Treat as a hint.
- **Streamlit app is not authenticated.** Anonymous single-user demo. Multi-tenant
  session tracking is a v2 concern.

---

## Monitoring

Every query is dual-written to `logs/queries.jsonl` (primary fallback) and
`cpie.query_logs` (Postgres, feeds Grafana). The Streamlit UI writes thumbs
feedback to `cpie.user_feedback`.

Grafana dashboards (auto-provisioned, no manual setup):

| Panel | What it shows |
|---|---|
| Query volume | Queries per hour |
| Refusal rate | % is\_refusal over time |
| Latency percentiles | p50 synthesis + retrieval |
| Top-cited docs | Which sources get used |
| Cost per day | Cumulative $ spend |
| User feedback ratio | Thumbs up / total votes |
| Recent failures | Failure reason + query snippet |

Start the monitoring stack: `docker compose up -d postgres grafana`

---

## Project Structure

```
cpie/
  src/
    ingestion/       pdf_loader, chunker, dlt_pipeline
    retrieval/       bm25_retriever, dense_retriever, hybrid_retriever,
                     institution_detector, reranker (preserved, not active)
    synthesis/       synthesiser, output_schema
    evaluation/      judge, eval_runner, retrieval_metrics
    monitoring/      logger (JSONL), db (Postgres)
  tests/             unit + integration tests
  data/eval/
    ground_truth.json          52 hand-crafted QA pairs
    results/                   eval run outputs + ablation tables
  monitoring/
    postgres/init.sql          schema DDL
    grafana/dashboards/        provisioned JSON
    grafana/provisioning/      datasource + dashboard provider YAMLs
  docs/
    week5_failure_analysis.md  A/B evidence for every dropped component
  scripts/
    ingest.py                  run dlt ingestion pipeline
    build_indices.py           build BM25 + Chroma from DuckDB
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
