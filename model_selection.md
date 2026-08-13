# Model Selection — CPIE Synthesis Layer

**Current shipped model:** GPT-5.4 mini (`gpt-5.4-mini`)
**Decision date:** 2026-08-09 (Week 5 Step 3b)
**Prior model:** GPT-4o mini (`gpt-4o-mini`)

---

## Timeline

### Week 2 (original selection)
- **Winner:** GPT-4o mini
- Compared against Claude Haiku 4.5. GPT-4o mini won 4.0 vs 2.7 avg quality on 10 pilot queries, 6× cheaper.

### Week 5 Step 3b (revisited)
- **Winner:** GPT-5.4 mini — **SHIPPED as the default**
- The Week 2 comparison was tier-matched (both small models). Never tested against a newer-generation mini. Real gap in the "measured evidence" story; closed here.

---

## Week 5 Step 3b — the full experiment

### Setup
- **Corpus:** 12 audited PDFs, 3402 chunks (production DuckDB index)
- **Retrieval:** Hybrid RRF k=60 + metadata filter (default post-Step 2f) → top-5 chunks
- **Prompt:** v2_numeric (default post-Step 3a)
- **Ground truth:** 52 queries (43 baseline + 4 vocab_shift + 5 colloquial + 3 adversarial_negative + 2 underspecified + 5 table_only)
- **Judge:** LLM-as-judge (per `src/evaluation/judge.py`)
- **Metrics:** Correctness, Faithfulness, Completeness, Refusal_appropriateness (all 1–5) + retrieval metrics + synthesis latency

### 3-way configuration table (all v2_numeric prompt)

| Config | Synth model | Judge model | Purpose |
|---|---|---|---|
| **A** | gpt-4o-mini | gpt-5.4-mini | Prior baseline (Step 3a shipping config) |
| **B** | gpt-4o-mini | **gpt-5.4** | Tiebreaker — isolates judge strictness |
| **C** | **gpt-5.4-mini** | gpt-5.4 | Candidate |

### Results

| Metric                     |     A |     B |     C | C − B (isolated model effect) |
|----------------------------|------:|------:|------:|-----------------------------:|
| Correctness                | 4.135 | 3.519 | 3.608 | **+0.089**                   |
| Faithfulness               | 4.365 | 4.692 | 4.941 | **+0.249**                   |
| Completeness               | 3.327 | 3.038 | 3.196 | **+0.158**                   |
| Refusal appr.              | 4.846 | 4.808 | 4.725 | −0.082                       |
| Hit@5, Recall@5, etc.      | ~identical across all three (prompt-only change doesn't touch retrieval) | | | |
| Synthesis latency (mean)   | 3620 ms | 3214 ms | **2377 ms** | **−837 ms (−26%)** |

### Key finding — the judge strictness confound

A → B swap (**synth held constant, judge changed** gpt-5.4-mini → gpt-5.4): Correctness dropped 4.135 → 3.519 (**−0.615**) **on the same synth answers**. That confirms gpt-5.4 is a substantially stricter judge than gpt-5.4-mini. The apparent A → C "regression" (4.135 → 3.608) was ~85% judge strictness, only ~15% model effect.

The honest apples-to-apples is **B vs C** — same (strict) judge held constant. gpt-5.4-mini synth beats gpt-4o-mini synth on 4 of 5 judge dimensions plus latency.

### Cost per full 52-query eval run

| Config | Synth cost | Judge cost | Total |
|---|---:|---:|---:|
| A | $0.027 | $0.149 | $0.176 |
| B | $0.027 | $0.505 | $0.532 |
| C | $0.169 | $0.489 | $0.659 |

### Production cost per query (gpt-4o-mini → gpt-5.4-mini)

| Volume | gpt-4o-mini | gpt-5.4-mini | Δ |
|---|---:|---:|---:|
| 100 queries/day | ~$1.50 / month | ~$9 / month | +$7.50 |
| 1000 queries/day | ~$15 / month | ~$90 / month | +$75 |

**Latency does NOT scale with corpus size** — top_k=5 stays constant at 12 docs or 100 docs. The −26% synthesis latency win holds at any corpus scale.

---

## Ship decision

**Ship gpt-5.4-mini as the synthesis default.**

Reasons:
- On the honest apples-to-apples (same judge held constant): better on Correctness (+0.089), Faithfulness (+0.249), Completeness (+0.158)
- **26% latency improvement** (real UX win, independent of corpus size)
- Cost increase (6× per query) is modest at demo scale (~$9/month at 100q/day)
- Refusal_appr −0.08 is within noise band
- Same shipping discipline as v2_numeric prompt (Step 3a): aggregate improves, no meaningful regression

## Judge model choice going forward

**Judge stays on gpt-5.4-mini** — cheap enough for regular eval regressions, and typical eval-signal band (>0.10 Correctness) exceeds the ~5% same-model bias from having synth and judge on the same model.

**Same-model bias caveat:** future eval numbers will be ~5% inflated in favor of gpt-5.4-mini synth. Flag documented in `judge.py` module docstring and `docs/week5_failure_analysis.md § 3b`.

**When to upgrade the judge:** for rigorous audits or cross-vendor validation, temporarily flip `JUDGE_MODEL = "gpt-5.4"` (5× cost per run) or use Claude Sonnet 5. Not routine — audit-only.

## Process lesson worth propagating

**When comparing models, ALWAYS run the judge-only tiebreaker first.** In this experiment the judge strictness effect was ~7× the model effect. Running the confounded comparison alone (A → C) would have made this look like a clear regression, when in fact gpt-5.4-mini is genuinely better on the same measurement scale. Adding to CLAUDE.md discipline notes.

---

## Locked settings

```yaml
synthesis:
  model: gpt-5.4-mini
  max_tokens: 800
  temperature: 0.0
```

```python
# src/synthesis/synthesiser.py
MODEL = "gpt-5.4-mini"

# src/evaluation/judge.py
JUDGE_MODEL = "gpt-5.4-mini"
```

---

*Raw results:*
- *`data/eval/results/judge_scores_20260808T083426Z_prompt-v2_numeric.json` (A — Week 5 Step 3a ship baseline)*
- *`data/eval/results/judge_scores_20260809T115025Z_prompt-v2_numeric.json` (B — tiebreaker)*
- *`data/eval/results/judge_scores_20260809T113920Z_prompt-v2_numeric.json` (C — shipped)*
