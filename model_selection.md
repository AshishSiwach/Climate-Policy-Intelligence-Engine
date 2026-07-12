# Model Selection — CPIE Synthesis Layer

**Decision date:** 2026-07-10
**Winner:** GPT-4o mini (`gpt-4o-mini`)
**Prior winner:** GPT-4o mini (won original Haiku 3.5 vs GPT-4o mini comparison, Week 2 Step 6)

---

## Evaluation Setup

- **Corpus:** 3 pilot PDFs — Ofgem SSES 2025, CBES Results 2022, IEA WEO 2025
- **Retrieval:** Hybrid RRF k=60 (BM25 + BAAI/bge-base-en-v1.5) → cross-encoder reranker → top-5 chunks
- **Queries:** 10 total — 6 factual (2 per doc), 2 cross-document synthesis, 2 out-of-corpus negatives
- **Scoring:** Manual 1–5 quality rating (answer accuracy + citation faithfulness + schema compliance)
- **Candidate:** GPT-5.4 mini (`gpt-5.4-mini`, released Mar 2026), reasoning_effort='low'
- **Baseline:** GPT-4o mini (`gpt-4o-mini`) — current locked synthesis model

---

## Results

| Metric | GPT-5.4 mini | GPT-4o mini |
|---|---|---|
| Avg quality score (1–5) | 3.6 | 4.1 |
| Avg quality — factual only (Q01-Q06) | 2.83 | 4.33 |
| Avg quality — synthesis only (Q07-Q08) | 4.50 | 2.50 |
| JSON schema compliance | 10/10 | 10/10 |
| Avg latency | 2507ms | 2898ms |
| Est. cost per 1,000 queries | $2.19 | $0.24 |

---

## Decision Rationale

GPT-4o mini remains the locked synthesis model. GPT-5.4 mini (reasoning_effort='low') scored lower overall (3.6 vs 4.1 avg) and substantially lower on factual extraction (2.83 vs 4.33 avg on Q01-Q06): on Q02, Q04, and Q06 it retrieved the correct supporting passage into its own citations list but then refused to answer, setting confidence below 0.2 despite direct textual support — a systematic under-confidence at low reasoning effort. It did show one genuine advantage: on Q07 (cross-document synthesis) it correctly declined when the retrieved context lacked an IEA excerpt, while GPT-4o mini fabricated an unsupported claim of alignment with "the IEA World Energy Outlook's goals" backed by zero IEA citations — a real citation-faithfulness miss worth tracking. But this one win does not offset GPT-5.4 mini's weaker factual accuracy, and GPT-5.4 mini is also roughly 9x more expensive per query (see cost row above). Latency was comparable between the two and noisy run-to-run given the short prompts — not a deciding factor either way. GPT-4o mini wins on the primary criterion (answer quality) and by a wide margin on cost. Follow-up: consider raising reasoning_effort to 'medium' for GPT-5.4 mini in a future retest, and separately tighten the synthesis prompt to instruct GPT-4o mini not to draw conclusions about a document absent from the retrieved excerpts.

---

## Locked Settings

```yaml
synthesis:
  model: gpt-4o-mini
  max_tokens: 512
  temperature: 0.0
```

---

*Raw results: `data/eval/results/model_selection_raw.json`*
*Notebook: `notebooks/model_selection.ipynb`*
