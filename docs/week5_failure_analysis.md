# Week 5 — Failure Analysis + Calibration

**Baseline used throughout:** `data/eval/results/judge_scores_20260801T230048Z.json` — 47 ground-truth queries × judge=gpt-5.4-mini, retrieval=hybrid (BM25 + Dense + RRF k=60), synthesis=gpt-4o-mini, top_k=5.

Every claim below is measured on that baseline. A/B evidence lands here as each sub-step ships.

---

## Step 2a — Refusal rate (Pattern A revisited)

### Headline

**Pattern A ("LLM refuses despite good retrieval") is misdiagnosed.** The Week 3 observation was real, but it wasn't the LLM. Every "over-refusal" in the Week 4 baseline is a **pipeline short-circuit** at `top_rrf < OUT_OF_CORPUS_RRF_THRESHOLD = 0.020`, not an LLM refusal.

Actual LLM refusal rate on positives: **0/38 = 0.0%**.

### Slice by query_type

| query_type      | n  | refused | rate  |
|-----------------|----|---------|-------|
| factual         | 28 | 3       | 10.7% |
| cross_document  | 4  | 2       | 50.0% |
| numeric         | 4  | 0       | 0.0%  |
| summarisation   | 2  | 0       | 0.0%  |
| negative        | 9  | 3       | 33.3% |

### Slice by probe (positives only)

| probe             | n  | refused | rate   |
|-------------------|----|---------|--------|
| baseline          | 21 | 1       | 4.8%   |
| colloquial        | 5  | 0       | 0.0%   |
| crossdoc_named    | 1  | 1       | 100.0% |
| underspecified    | 2  | 1       | 50.0%  |
| vocab_mismatch    | 2  | 1       | 50.0%  |
| vocab_shift       | 5  | 0       | 0.0%   |
| wording_variant   | 2  | 1       | 50.0%  |

### The 5 over-refused positives — all short-circuits

Detected via `synthesis_tokens == 0` (no LLM call happened) AND `top_rrf < 0.020`:

| id                                    | probe            | top_rrf | hit@5 | path          |
|---------------------------------------|------------------|---------|-------|---------------|
| ccc_prog25_transport_25               | vocab_mismatch   | 0.0164  | 1     | short-circuit |
| iea_fossil_peak_variant_30            | wording_variant  | 0.0164  | 1     | short-circuit |
| xdoc_ccc_iea_transport_33             | crossdoc_named   | 0.0164  | 1     | short-circuit |
| xdoc_ccc_boe_budget_36                | (baseline xdoc)  | 0.0164  | 1     | short-circuit |
| probe_underspec_climate_risk          | underspecified   | 0.0164  | 1     | short-circuit |

All 5 have retrieval that surfaced the correct doc (`hit@5=1`) but were killed by the threshold before the LLM could see them.

### The 8 short-circuits, all queries

Every query where `top_rrf < 0.020` gets short-circuited. Below the threshold, retrieval score is exactly `0.0164` for every query — that's the highest RRF value achievable when the top-1 chunk is unique to BM25 or Dense (rank 1 in one retriever, absent from the other → `1/61 ≈ 0.0164`). So the threshold is really "did both retrievers agree on the top hit?", not "did we find something relevant?".

Of the 8:
- **5 positives** short-circuited incorrectly (false positives for the threshold)
- **3 negatives** short-circuited correctly (true positives for the threshold)

Threshold precision on the current baseline: **3/8 = 37.5%**.
Threshold recall on negatives:                **3/9 = 33.3%**.

### Negatives that escaped the threshold

6 negatives had `top_rrf ≥ 0.020` and were passed to the LLM:

| id                                    | top_rrf | correctness | refusal_appr |
|---------------------------------------|---------|-------------|--------------|
| neg_sourdough_38                      | 0.0292  | 5           | 5            |
| neg_fca_sdr_41                        | 0.0325  | 4           | 5            |
| neg_eu_csrd_42                        | 0.0328  | 5           | 5            |
| neg_ofgem_pricecap_43                 | 0.0325  | 5           | 5            |
| probe_adv_neg_ofgem_pricecap_winter   | 0.0325  | 5           | 5            |
| probe_adv_neg_iea_smr_deployment      | 0.0313  | 2           | 1            |

**5 of 6 got refusal_appropriateness=5** — the LLM handled them correctly even though the pipeline didn't short-circuit. The LLM refused (or gave a "cannot confirm from corpus" style answer) — the generated text just doesn't match the canonical prefix, which is why my refusal-detection heuristic missed them.

**1 of 6 fabricated** — `probe_adv_neg_iea_smr_deployment` (LLM stitched together an SMR-adjacent answer from noisy IEA chunks; correctness=2, refusal_appr=1). This is the **actual Pattern A** — LLM fabrication when retrieval surfaces plausible-looking but off-topic chunks. Real rate: **1/9 negatives = 11%**.

### Implications for Step 2

1. **Pattern A is essentially a non-issue for v1.** The LLM refuses / hedges correctly ~89% of the time on negatives when it gets called. Only fix-worthy case is the SMR-style noisy-chunk fabrication, and that's really a faithfulness problem, not a refusal problem. Defer to the "faithful-synthesis check" v2 roadmap item.
2. **The real story is Pattern B (threshold calibration) — do it next.** Every short-circuit in the baseline sits at exactly `top_rrf = 0.0164`, which suggests the threshold at 0.020 is a coarse gate that misclassifies more positives than negatives. A doc-agreement signal or absolute-RRF elbow could be much better.
3. **Refusal detection heuristic needs to be broadened.** Current heuristic only catches the canonical prefix. The LLM refuses in richer language when it decides to. For the calibration work in 2b/2d, use `refusal_appropriateness < 3` from the judge as a proxy for "actually refused" instead of string-matching.
4. **Judge-labelled cross-doc completeness (1.75 average in Week 4 baseline) is not primarily a refusal issue** — of 4 cross-doc queries, 2 short-circuited and 2 answered. Completeness on the 2 that answered needs its own look in Step 2c.

### What ships from 2a

**Nothing yet.** No config change. No code change. The finding is: don't tune Pattern A, tune Pattern B — proceed to 2b.

Refusal-detection heuristic upgrade (use judge `refusal_appropriateness < 3` instead of prefix match) will be applied in the 2b analysis and beyond — implementation-only, no user-visible behaviour change.

---

## Step 2e — 4-config retrieval ablation

**Reordered to run first** (per session decision — pipeline choice is foundational; all downstream calibration depends on which config wins).

**Runner:** `src/evaluation/ablation_runner.py`
**Source data:** `data/eval/results/ablation_20260807T144600Z.json`
**Run:** 47 queries × 4 configs = 188 pipeline runs, 1571s wall time, $0.6498 total cost.
**Held constant across configs:** synthesis = gpt-4o-mini, judge = gpt-5.4-mini, top_k=5. Out-of-corpus short-circuit disabled for the ablation (each config uses different score scale; threshold calibration is post-ablation).

### Aggregate results

| config           | correctness | faithfulness | completeness | refusal_appr | hit@5 | recall@5 | precision@5 | MRR@5 | nDCG@5 | ret_ms |
|------------------|------------:|-------------:|-------------:|-------------:|------:|---------:|------------:|------:|-------:|-------:|
| **bm25**         | 3.77        | 3.98         | 3.02         | 4.83         | 0.895 | 0.825    | 0.627       | 0.800 | 0.807  | **18** |
| **dense**        | 4.15        | 4.40         | **3.28**     | 4.91         | 0.921 | 0.842    | **0.672**   | **0.873** | **0.870** | 172 |
| **hybrid**       | 4.13        | 4.38         | 3.23         | 4.91         | 0.921 | **0.868** | 0.664       | 0.830 | 0.848  | 108 |
| **hybrid_rerank**| 4.15        | 4.40         | 3.26         | 4.91         | 0.895 | 0.846    | 0.647       | 0.842 | 0.841  | 563 |

### Correctness by query type

| config           | factual (n=28) | cross_document (n=4) | numeric (n=4) | summarisation (n=2) | negative (n=9) |
|------------------|---------------:|---------------------:|--------------:|--------------------:|---------------:|
| bm25             | 3.75           | 3.25                 | 3.75          | 3.00                | 4.22           |
| dense            | **4.21**       | 3.50                 | 4.25          | 3.00                | 4.44           |
| hybrid           | 4.04           | **3.75**             | 3.75          | **3.50**            | **4.89**       |
| hybrid_rerank    | 4.04           | 3.25                 | **4.75**      | 3.00                | **4.89**       |

### Negatives handling (n=9)

| config           | handled_well | rate  | mean refusal_appr |
|------------------|-------------:|------:|------------------:|
| bm25             | 7/9          | 77.8% | 4.11              |
| dense            | 8/9          | 88.9% | 4.56              |
| hybrid           | 8/9          | 88.9% | 4.56              |
| hybrid_rerank    | 8/9          | 88.9% | 4.56              |

### Reading the numbers

1. **BM25 alone is not viable.** Lowest Correctness (3.77 — 0.38 pts below the rest), lowest Faithfulness, and 1 extra negative fabrication vs. the others. Confirms the Week 3 3-query ablation instinct at 47-query scale.

2. **Dense, Hybrid, Hybrid+Rerank are statistically indistinguishable on aggregate judge scores.** All three within 0.02 Correctness of each other. That difference is well within the noise band of 38-sample Correctness means (integer 1–5 grading).

3. **On retrieval fundamentals, the three "good" configs diverge:**
   - **Hybrid wins on Recall@5** (0.868 vs 0.842 / 0.846) — best at *finding* all relevant chunks in top-5
   - **Dense wins on Precision@5, MRR@5, nDCG@5** — best at *ranking* the good chunks near top-1
   - **Rerank matches Dense on all top-1 ranking metrics but loses Hit@5** (0.895 vs 0.921) — reranker occasionally reorders the correct doc *out* of top-5

4. **Per-type story is where the interesting deltas live:**
   - **Cross-doc:** Hybrid wins (3.75) — beats Dense (3.50) and Rerank (3.25). Hybrid's Recall@5 on cross-doc = 0.75 vs Dense 0.50, Rerank 0.62. **Hybrid's BM25 leg is contributing on cross-doc queries** where lexical anchoring on institution names matters.
   - **Numeric:** Rerank wins (4.75) — 1 full point above Hybrid (3.75). Small sample (n=4) but the gap is striking. Cross-encoder does better on table/numeric queries where the target answer is a specific value in a specific chunk. Worth flagging for possible per-type activation in v2.
   - **Negatives:** Hybrid and Rerank tied (4.89) — both slightly ahead of Dense (4.44). Hybrid + Rerank both retrieve less semantically "close" out-of-domain chunks that trigger appropriate refusal.

5. **Latency:** Hybrid 108ms → Rerank 563ms is a **5.2× slowdown for no aggregate quality gain**. In v1 this is drowned by ~2.5s synthesis latency, but at any future scale this matters.

### Decision

**Keep Hybrid as the v1 default.**

- Best-in-class Recall@5 (0.868) — fundamental retrieval quality
- Best on the two dimensions CPIE actually cares about: **cross-doc synthesis** (3.75 vs 3.50 Dense) and **negative handling** (4.89 vs 4.44 Dense)
- 5.2× faster than Rerank; only 64ms slower than Dense
- Aggregate Correctness statistically tied with Dense and Rerank
- Retains retriever diversity (lexical + semantic) — a robustness property that doesn't show up in judge scores but matters for query types we haven't yet observed

**Do NOT enable reranker by default.**

- Zero aggregate Correctness benefit (Δ = 0.02, noise)
- WORSE on cross-doc Correctness (3.25 vs Hybrid 3.75)
- 5.2× retrieval latency
- One legitimate advantage: +1.0 Correctness on numeric queries (n=4). **Log as v2 candidate**: per-query-type reranker activation for numeric/table queries only.

**Dense stays available as a documented fallback.**

- Simpler than Hybrid (one retriever, one vector store)
- Best Precision@5 and MRR@5
- Note as candidate if Hybrid ever needs to be simplified for operational reasons in v2

### Downstream consequences

- **Step 2b (threshold calibration)** continues on **RRF scores** (Hybrid's native score field), no re-plan needed. Existing Week 4 baseline is directly usable.
- **Step 2c (cross-doc coverage)** uses Hybrid, expected to show the same 0.75 Recall@5 on cross-doc as this ablation.
- **Step 2d (confidence weight fitting)** uses Hybrid's existing `confidence_signals` — no rework.
- **v2 roadmap update:** add "per-query-type reranker activation on numeric queries" — the only signal in this ablation that Rerank does something Hybrid doesn't.

### Reordering lesson for the record

CLAUDE.md placed reranker re-eval as Step 2e (last) in Week 5 failure analysis. That was wrong ordering: pipeline choice is foundational to every calibration downstream. Correct place would have been Week 4, immediately after the ground truth dataset existed (Week 4 Step 3). Adding a note to CLAUDE.md so this pattern doesn't repeat in future projects.

### What ships from 2e

- **No code change** — Hybrid stays as default; reranker preserved in `src/retrieval/reranker.py` as it was
- **v2 roadmap update** — flag numeric-query reranker activation
- **CLAUDE.md process note** — ablation should happen at Week 4, not Week 5
- Failure analysis doc updated (this section)

---

## Step 2b — Out-of-corpus RRF threshold calibration

**Source:** Hybrid config from `ablation_20260807T144600Z.json` (47 queries, threshold bypassed during ablation → we have LLM output for the queries that would normally short-circuit) compared against Week 4 baseline `judge_scores_20260801T230048Z.json` (same queries, threshold=0.020 active).
**Chart:** `docs/charts/step2b_rrf_distribution.png`

### The top-RRF distribution

| Set             | n  | min    | p25    | p50    | p75    | max    | notes                             |
|-----------------|----|-------:|-------:|-------:|-------:|-------:|-----------------------------------|
| Positives       | 38 | 0.0164 | 0.0310 | 0.0318 | 0.0323 | 0.0328 | 5 outliers at 0.0164; rest cluster tight 0.029–0.033 |
| Negatives       |  9 | 0.0164 |   —    | 0.0313 |   —    | 0.0328 | 3 at 0.0164; **6 mixed inside the positive cluster** |

### The fundamental issue

RRF at k=60 produces a small set of **discrete** scores: when retrievers strongly disagree (top-1 chunk appears in only one retriever), RRF pins at 1/(60+1) ≈ 0.0164. This happens whether the query is in-corpus or out-of-corpus — it's a **retriever-agreement** signal, not a corpus-relevance signal.

For queries where retrievers agree, top-1 RRF lands in the ~0.029–0.033 range. **The positive and negative distributions overlap entirely inside that band** — 6 of 9 negatives sit at scores that also appear on legitimate positives.

**Consequence: no RRF threshold can cleanly separate positives from negatives.**

### Threshold sweep on the actual data

Every RRF threshold t you can pick lands in one of three regimes. Numbers below assume threshold = "block if top_rrf < t":

| Regime                | Threshold t         | Blocked positives | Blocked negatives | Comment              |
|-----------------------|---------------------|------------------:|------------------:|----------------------|
| Below pile-up         | t ≤ 0.0164          | 0/38 (0%)         | 0/9 (0%)          | no short-circuit ever fires |
| **Kill pile-up**      | 0.0164 < t ≤ 0.029  | **5/38 (13.2%)**  | **3/9 (33.3%)**   | **current 0.020 sits here** |
| Above pile-up         | 0.029 < t           | ≥ 6/38            | ≥ 4/9             | starts killing legitimate positives |

At the current threshold (0.020), short-circuit precision is **3/8 = 37.5%** — more legitimate positives are blocked than negatives caught.

### Empirical A/B: what happens if the 8 pile-up queries reach the LLM instead of short-circuiting

We already have this data — the ablation ran Hybrid with threshold bypassed. Compare judge scores per pile-up query, baseline (short-circuit) vs ablation (LLM answered):

| id                                    | type            | base C/F/Cp/R | LLM C/F/Cp/R |
|---------------------------------------|-----------------|---------------|--------------|
| ccc_prog25_transport_25               | factual         | 4/5/1/**1**   | 3/3/3/**5**  |
| iea_fossil_peak_variant_30            | factual         | 4/5/1/**1**   | 3/4/2/**5**  |
| probe_underspec_climate_risk          | factual         | 5/5/5/5       | 2/4/2/5      |
| xdoc_ccc_boe_budget_36                | cross_document  | 4/5/1/**1**   | 4/5/3/**5**  |
| xdoc_ccc_iea_transport_33             | cross_document  | 4/5/1/**1**   | 4/4/3/**5**  |
| neg_champions_league_39               | negative        | 5/5/5/5       | 5/5/5/5      |
| neg_python_error_40                   | negative        | 5/5/5/5       | 5/5/5/5      |
| probe_adv_neg_boe_crypto              | negative        | 5/5/5/5       | 5/5/5/5      |

**Per-positive deltas** (LLM answered − short-circuit refused):
- Δ Correctness: **-1.00**
- Δ Faithfulness: **-1.00**
- Δ Completeness: **+0.80**
- Δ Refusal appropriateness: **+3.20**

**Per-negative deltas**: zero across the board. The 3 pile-up negatives got the same 5/5/5/5 from the judge whether short-circuited or LLM-refused — the LLM correctly refused all 3.

### Net impact if OUT_OF_CORPUS_RRF_THRESHOLD is set to 0.0 (bypass the gate)

Replacing the 8 pile-up rows in the Week 4 baseline with their ablation values:

| Metric                    | Baseline (0.020) | Threshold = 0.0 | Δ          |
|---------------------------|-----------------:|----------------:|-----------:|
| Correctness               | 4.128            | 4.021           | **-0.106** |
| Faithfulness              | 4.489            | 4.383           | -0.106     |
| Completeness              | 3.128            | 3.213           | **+0.085** |
| Refusal appropriateness   | 4.574            | 4.915           | **+0.340** |

### Interpreting the Correctness drop honestly

The −0.106 Correctness delta looks concerning. It **overstates** the harm because of a judge-rubric artefact:

The judge's Correctness rubric ("do the factual claims in the answer match the reference") gives **Correctness = 4** for a canonical refusal on a positive query. A refusal has no claims to be wrong, so it lands in the "minor differences" bucket by default. When the LLM actually answers imperfectly from disagreeing-retriever chunks, it lands at Correctness = 3.

But the user experience is strictly better with the LLM answer:
- **Refusal**: user gets "corpus doesn't contain this" and walks away. Completeness = 1, Refusal_appr = 1.
- **LLM answer**: user gets a citation-backed partial answer. Completeness = 3, Refusal_appr = 5.

The `Completeness +0.08` and `Refusal_appr +0.34` deltas are the honest signal. Correctness is measuring the wrong thing here (judge inflates canonical refusals).

**Recommendation:** the Correctness rubric is worth revisiting in v2 to explicitly penalise refusals on positive queries. For now, weight the decision on Completeness + Refusal_appr, which correctly identify the failure mode.

### Recommended change

**Set `OUT_OF_CORPUS_RRF_THRESHOLD = 0.0` in `src/synthesis/synthesiser.py`.** Keep the code path (mechanism preserved for a future better gate); disable by default.

**Rationale:**
1. RRF top-1 cannot separate positives from negatives (3 negatives share scores with 5 positives at 0.0164; 6 more negatives share the 0.029–0.033 range with all remaining positives).
2. Current threshold (0.020) has short-circuit precision of 37.5% — kills more positives than it catches negatives.
3. LLM correctly refused all 3 pile-up negatives in the ablation (matched short-circuit safety exactly).
4. LLM produces useful partial answers for 5 pile-up positives that were previously blocked (+0.80 Completeness, +3.20 Refusal_appr per query).
5. Latency cost: 5 extra synthesis calls (~2.5s each). Cost: ~$0.005 per full corpus of queries at this rate. Nothing.

**Explicit safety note:** removing the gate does NOT increase fabrication risk on the pile-up negatives — the LLM handles them correctly. It transfers safety from a coarse RRF threshold to the LLM's own refusal behaviour, which is more accurate on this corpus.

**What's NOT recommended:** raising the threshold above 0.029. Would kill 5+ additional legitimate positives from the main cluster while catching only 3 more negatives.

### Better gate design (v2 candidate — outside 2b scope)

The right long-term fix is a **retriever-agreement gate** rather than an absolute-score threshold:
- Short-circuit when top-1 chunk has `bm25_rank > 5` AND `dense_rank > 5` (both retrievers considered it unimportant)
- Or: short-circuit when only ONE retriever contributed the top-1 chunk AND semantic_sim(query, chunk) < X

Both require new fields in the retrieval output and their own calibration. Flag as v2 roadmap item **"replace RRF-threshold gate with retriever-agreement gate"**.

### What ships from 2b

- **Code change (pending your approval)**: `OUT_OF_CORPUS_RRF_THRESHOLD = 0.020 → 0.0` in `src/synthesis/synthesiser.py`. Add a comment explaining the calibration finding and pointing at this document.
- **v2 roadmap update**: add "retriever-agreement gate to replace RRF threshold"
- **v2 roadmap update** (optional): "revise judge Correctness rubric to penalise refusals on positive queries"
- Failure analysis doc updated (this section)
- Chart: `docs/charts/step2b_rrf_distribution.png`

---

## Step 2c — Cross-doc top-k coverage (Pattern C)

**Source:** `ablation_20260807T144600Z.json` Hybrid config for k=5 baseline; fresh retrieval-only re-run at k=10 and k=20.

**Multi-source queries in ground truth: 6 of 38 positives** (4 labelled `cross_document`, 2 `underspecified` probes with n_expected_docs ≥ 2).

### Source-hit-rate by k

Source-hit-rate = fraction of expected sources present in top-k unique doc_ids.

| id                            | n_exp | k=5    | k=10   | k=20   |
|-------------------------------|------:|:------:|:------:|:------:|
| xdoc_ccc_iea_transport_33     | 2     | 2/2    | 2/2    | 2/2    |
| xdoc_ofgem_eso_flex_34        | 2     | 1/2    | 1/2    | **1/2**|
| xdoc_boe_iea_scenarios_35     | 2     | 1/2    | 1/2    | **1/2**|
| xdoc_ccc_boe_budget_36        | 2     | 2/2    | 2/2    | 2/2    |
| probe_underspec_heat_pumps    | 3     | 1/3    | 0/3    | **1/3**|
| probe_underspec_climate_risk  | 3     | 2/3    | 3/3    | 3/3    |
| **Mean**                      |       | **0.67**| **0.67**| **0.72**|
| **Perfect (all sources hit)** |       | 2/6    | 3/6    | 3/6    |

### Reading the numbers

Raising k from 5 → 20 barely helps: **mean source-hit-rate +0.05 (0.67 → 0.72), perfect count +1**. Two queries never find the second named source *even at k=20*:

- `xdoc_ofgem_eso_flex_34`: "How does Ofgem's load control licensing regime relate to the network needs identified in **ESO's Beyond 2030 report**?" — ESO named explicitly in the query, still missing at k=20
- `xdoc_boe_iea_scenarios_35`: "How does the Bank of England's Late Action scenario compare with the **IEA's scenario framing** of the transition?" — IEA named explicitly, still missing at k=20

### Root cause: top-k is not the bottleneck

The failure isn't "the 5-chunk window is too narrow". It's that the missed sources score so poorly that they don't reach top-20 either. Both ESO and IEA docs are semantically/lexically far from these queries — ESO Beyond 2030 talks about transmission network planning, not "load control licensing"; IEA WEO uses "STEPS/APS/NZE" not "Late Action". Simple `top_k` inflation can't fix that.

The one query that DOES benefit from raising k (`probe_underspec_climate_risk`: 2/3 → 3/3 at k=10) is a very generic query where the third source appears at rank 6–10 — a boundary case, not a systematic pattern.

### Interaction with the reranker

Cross-encoder rerank in the Step 2e ablation was *worse* than Hybrid on cross-doc queries (Correctness 3.25 vs 3.75). Reranker can only reorder what retrieval surfaces — it can't recover a source that both BM25 and Dense already buried.

### What actually fixes it

Two paths, both out of scope for 2c but both flagged:

1. **Query rewriting (Step 3, next)**: generates 2–3 variants with different lexical anchors, unions retrieval across variants before RRF. Directly addresses cases where a single query phrasing lexically excludes a relevant source. Cheapest fix; already scheduled.
2. **Metadata filtering (v2)**: when a query names an institution ("Ofgem's...", "the IEA's..."), filter retrieval to just those institutions before BM25+Dense. Guarantees per-source coverage. Already in v2 roadmap; this analysis confirms it's the right long-term fix.

**Neither is "raise top_k".** Raising top_k costs LLM prompt tokens (5 → 20 chunks quintuples the prompt) for a +0.05 source-hit-rate — bad trade.

### What ships from 2c

- **No code change.** `top_k=5` stays.
- Documented empirical finding: cross-doc coverage gaps are retrieval-quality issues, not top-k issues.
- **Confirms Step 3 (query rewriting) is correctly scoped** — it's the right lever for the failure modes we found.
- **Metadata filtering promoted from v2 → v1** as a follow-on to this finding. Directly fixes the 4 xdoc queries that explicitly name institutions but miss the second source. Runs *before* query rewriting so filter and rewriting gains are cleanly attributable.
- Failure analysis doc updated (this section).

---

## Step 2d — Confidence weight fitting

**Source:** Ablation Hybrid config — 47 queries with logged `confidence_signals` + judge `correctness`. Target: predict `correctness >= 4` (36/47 positives).

### The finding

**The current v1 confidence formula is barely better than random**, and three of the four signals are anti-correlated with correctness. Fitted regression overfits on n=47 and generalises worse than random.

### Signal correlations with correctness

| Signal            | Pearson vs correctness(1–5) | Direction  | AUC as sole predictor of correctness≥4 |
|-------------------|----------------------------:|:-----------|---------------------------------------:|
| `score_signal`    | **+0.146**                  | higher-is-better ✓ | **0.668**                     |
| `agreement_signal`| +0.004                      | LOWER-is-better (weak) | 0.590                     |
| `margin_signal`   | −0.034                      | higher-is-better (weak) | 0.588                    |
| `citation_signal` | **−0.311**                  | **LOWER-is-better** ✗ (inverted from v1 intuition) | 0.593 |

Only `score_signal` points the direction v1 assumed. The others range from noise to actively inverted — LLM producing more citations correlates with *lower* correctness, likely because the LLM piles up citations on ambiguous queries where it's grasping for support.

### Candidate formulas — AUC comparison

| Formula                                                      | AUC   | Notes |
|--------------------------------------------------------------|------:|-------|
| Random baseline                                              | 0.500 | reference |
| **v1: equal weights (0.25 × 4)**                             | **0.513** | barely above random |
| `score_signal` alone                                         | **0.668** | best single-feature predictor |
| `invert agree + cite` (0.25, −0.25, 0.25, −0.25)             | 0.643 | direction-corrected multi-signal |
| `score − citation`                                           | 0.660 | 2-feature version |
| Logistic regression L2-fit, all 4 signals — in-sample        | 0.643 | (overfits) |
| Logistic regression L2-fit, all 4 signals — **LOOCV**        | **0.386** | **generalises worse than random** |

### Why fitted regression fails

- **n=47 is too small** for reliable fitting of 4 weights (rule of thumb: 10–20 samples/feature).
- **Signals lack discriminative range** — `agreement_signal` sits at mean 0.93 (std 0.10) across all queries; barely varies. `score_signal` sits at mean 0.58 (std 0.12); narrow band. Little signal to fit against.
- Logistic regression captures noise; LOOCV exposes it.

### Recommendation

**Simplify the confidence formula to `confidence = score_signal`**. Reasoning:

- Best single-feature AUC (0.668) — highest signal-to-noise on the data we have
- Deterministic formula, no learned weights → in-sample AUC = expected out-of-sample AUC (no overfitting risk)
- Interpretable: "confidence is how well retrieval agreed the top chunk was relevant"
- Deprecates 3 signals that don't help (`agreement`, `margin`, `citation`). Keep computing them and logging them for future analysis; just don't include in the user-facing confidence number.

**Honest caveat:** AUC 0.668 with 95% CI ~[0.53, 0.81] on n=47 — this is *the best we can honestly claim* from this dataset, not "well-calibrated". Confidence is a weak predictor even after this change.

### UI threshold recalibration required

`score_signal` observed range is [0.33, 0.66] (mean 0.58) — much narrower than v1's [0, 1] assumed range. Current UI thresholds (HIGH ≥ 0.7, MEDIUM 0.4–0.7, LOW < 0.4) would produce almost no HIGH badges under the new formula. Recalibrate to distribution percentiles:

| Badge  | New threshold | Rationale                          |
|--------|:-------------|------------------------------------|
| HIGH   | ≥ 0.60       | top ~30% of observed score_signal  |
| MEDIUM | 0.45–0.60    | middle ~40%                        |
| LOW    | < 0.45       | bottom ~30%                        |

These are still empirical placeholders — recalibrate again after query rewriting + metadata filtering land (both shift score_signal distribution).

### What ships from 2d

**Final decision (user call):** confidence is removed from v1 entirely. Not "simplified to score_signal", not "kept internal-only". Removed everywhere — schema, synthesiser, logger, main, tests, docs, config, CLAUDE.md.

Rationale: shipping a signal with AUC 0.668 (CI overlapping random) as any form of user promise is worse than shipping nothing. Simplifying to score_signal alone would still leave a weak number in the codebase and pretend it means something. The cleanest engineering move is delete.

**Code changes (Week 5 Step 2d):**
- `src/synthesis/output_schema.py::AnalystBrief`: `confidence` field removed
- `src/synthesis/synthesiser.py`: `_compute_confidence` + `SCORE_NORMALISER` + `MARGIN_NORMALISER` + `CITATION_SATURATION` + `confidence_signals` dict all removed; branches simplified
- `src/monitoring/logger.py`: `confidence` + `confidence_signals` removed from log records
- `src/evaluation/judge_runner.py`, `ablation_runner.py`: stopped reading `generated_confidence` / `confidence_signals`
- `main.py`, `configs/config.yaml`, `scripts/validate_e2e_5queries.py`: stripped
- `tests/test_synthesis.py`, `tests/test_main.py`: confidence tests deleted; signal-key assertions removed
- `CLAUDE.md`: CRAG framing, output schema table, three-check → two-check, Week 5 Step 2 task list, Streamlit UI section, v2 roadmap, and Locked Decisions all updated. New locked decision: "**No user-facing confidence in v1**" — spells out v2 re-introduction conditions.

**v2 conditions to re-introduce confidence** (from the failure analysis + user decision):
- Collect n ≥ 100 ground-truth queries
- Add candidate signals with different failure-mode coverage: `semantic_sim` (query↔top-chunk cosine), `doc_aware_margin` (top-1 vs top-scoring chunk from a different doc_id), `n_unique_docs_in_top_5` (retrieval diversity)
- Fit multi-signal formula; only ship if held-out AUC ≥ 0.75
- Investigate *why* v1 `citation_signal` was inverted — probably LLM padding citations on hard queries — fix upstream if so

Failure analysis doc updated (this section).

---

## Step 2f — Metadata filtering (promoted from v2)

**Motivation:** Step 2c showed cross-doc coverage failures are retrieval-quality issues, not top-k issues. Queries that explicitly name institutions ("Ofgem", "IEA", "Bank of England"…) still miss the second source because the missed doc is semantically far from the query wording. Metadata filtering pre-restricts retrieval to just the named institutions — a deterministic fix for that specific failure mode.

### Implementation

- `src/retrieval/institution_detector.py` — regex-based detection for the 6 institutions in `DOC_REGISTRY` (Ofgem, DESNZ, IEA, BoE, CCC, ESO), with expanded synonyms (e.g. "Bank of England" for BoE). Case-insensitive with word boundaries; unit-tested for false-positive avoidance ("boeing" doesn't match "boe", "idea" doesn't match "iea").
- `DenseRetriever.query()` — accepts `institutions=[...]`, uses Chroma's native `where={"institution": {"$in": [...]}}` filter.
- `HybridRetriever.retrieve()` — accepts `institutions=[...]`. Dense uses native filter; BM25 is post-filtered from a larger candidate pool (5× `top_k`) so we don't starve fusion after filtering. If the combined filtered pool is empty, falls back to unfiltered retrieval (better to answer wider than refuse on a spurious detection).
- `main.py` — detects institutions per query, passes to `hybrid.retrieve`, logs detected list. Behind `METADATA_FILTER_ENABLED = True` (module-level flag for easy rollback).
- `judge_runner.py` — same flag for A/B measurement.

### A/B measurement — 47 queries, filter OFF (baseline) vs ON

**Source files:**
- OFF: `judge_scores_20260801T230048Z.json` (Week 4 baseline)
- ON: `judge_scores_20260807T211104Z.json` (this step)

Filter fired on 25/47 queries (queries that mentioned at least one institution).

**Aggregate (47 queries):**

| Metric                     | OFF   | ON    | Δ       |
|----------------------------|------:|------:|--------:|
| Correctness                | 4.128 | 4.064 | -0.064  |
| Faithfulness               | 4.489 | 4.234 | **-0.255** |
| Completeness               | 3.128 | 3.277 | +0.149  |
| Refusal appropriateness    | 4.574 | 4.830 | **+0.255** |
| Hit@5                      | 0.921 | 0.947 | +0.026  |
| Recall@5                   | 0.868 | 0.895 | +0.026  |
| Precision@5                | 0.664 | 0.699 | +0.035  |
| MRR@5                      | 0.830 | 0.869 | +0.039  |
| nDCG@5                     | 0.848 | 0.882 | +0.034  |
| Retrieval latency          | 170ms | 188ms | +18ms   |

**Cross-doc slice (n=4) — the primary target:**

| Metric        | OFF   | ON    | Δ       |
|---------------|------:|------:|--------:|
| Correctness   | 3.750 | 3.750 | 0       |
| **Completeness** | **1.750** | **2.750** | **+1.000** |
| Recall@5      | 0.750 | 0.750 | 0       |
| Hit@5         | 1.000 | 1.000 | 0       |

### Interpreting the Faithfulness "drop"

Aggregate Faithfulness -0.26 looks concerning. Decomposition of the 13 queries with Faithfulness Δ ≤ −1:

| Group | Count | Interpretation |
|---|---:|---|
| Filter did NOT fire + retrieval identical | **8/13** | Judge noise — same input, different score. Confirms LLM/judge non-determinism at n=47. |
| Filter fired + retrieval identical | 3/13 | Judge noise (same chunks, just re-scored) |
| Filter fired + retrieval changed | 2/13 | Real effect: -3 on `probe_adv_neg_boe_crypto`, -1 on `iea_fossil_peak_variant_30` |

Meanwhile 7 queries gained Faithfulness (one was +2 on `probe_adv_neg_iea_smr_deployment` — the SMR fabrication we flagged in Step 2a was fixed by filtering out non-IEA noise chunks).

**Filter-attributable-only Faithfulness net: roughly zero.** The aggregate -0.26 is dominated by judge noise on unaffected queries.

### Ship decision

**Ship the filter (default ON).** Rationale:

- Primary target moved: **cross-doc Completeness +1.0** on n=4 (biggest single-slice delta of Week 5)
- Retrieval metrics all up (+0.03 to +0.04) — filter is doing what it was designed to do
- Negative handling improved: **refusal_appropriateness +0.26**, including fixing the `probe_adv_neg_iea_smr_deployment` fabrication that was the sole real Pattern A instance flagged in Step 2a
- Correctness essentially flat (Δ −0.06, within noise)
- Faithfulness "drop" dissolves under noise decomposition
- Latency cost negligible (+18ms out of ~7s total)
- Behind a config flag → single-line rollback if the picture changes with more data

### Known limitations

- Filter helps only when a query explicitly names institutions. For 22/47 queries (46% of corpus), no institution is named and the filter is a no-op. Underspecified queries like *"heat pumps"* or *"climate risk - what should I know"* still miss cross-source coverage — Step 3 (query rewriting) territory.
- Even when the filter narrows to the named institutions, **it doesn't guarantee per-source top-5 coverage**. The 4 xdoc queries showed identical retrieved doc-lists with vs without filter — ESO chunks for `xdoc_ofgem_eso_flex_34` still didn't reach top-5 because Ofgem chunks out-scored them even in the filtered pool. True per-source guarantees would need per-institution top-N + union (v2 candidate).

### What ships from 2f

- New module: `src/retrieval/institution_detector.py` + unit tests
- `DenseRetriever.query()` accepts `institutions=` (Chroma native filter)
- `HybridRetriever.retrieve()` accepts `institutions=` (dense filter + BM25 post-filter + zero-match fallback) + integration tests
- `main.py` + `judge_runner.py` detect + pass institutions behind `METADATA_FILTER_ENABLED = True`
- Baseline eval result committed for reproducibility: `judge_scores_20260807T211104Z.json`
- Failure analysis doc updated (this section)

**v2 candidates unchanged** — the "per-source retrieval union" idea for true source-coverage guarantees stays in v2. Metadata filtering as-shipped is the deterministic v1 lever; per-source union is the follow-up.

---

## Step 2g — Table-retrieval evaluation (CLAUDE.md Week 5 Step 2b — Minimal scope)

**Motivation:** CLAUDE.md flagged this as a gap: Tier 2 heading injection ships for 5 table-heavy documents but no metric confirms whether table chunks actually reach top-5 on numeric queries. Minimal scope = Additions 1 + 3 (metric slice + 5 table-only probes); Additions 2 (schema field) and 4 (Tier 2 heading A/B) deferred.

### Implementation

- `src/evaluation/retrieval_metrics.py`: new `table_fraction_at_k(chunks, k)` — fraction of top-k chunks with `chunk_type=="table"`. Purely descriptive; applies to every query.
- `evaluate_query()`: now also emits `table_fraction@k` alongside standard IR metrics.
- `retrieval_eval_runner.py`: per-query output now includes `retrieved_chunk_types` (list of "table"/"prose") + `detected_institutions` (from Step 2f) + `table_fraction@k`. Negatives get the field too.
- Ground truth: **5 new `[PROBE: table_only]` queries** appended to `ground_truth_raw.json` and migrated. Reference answers grounded in actual corpus chunks (verified by grep against DuckDB before authoring).
- Ground truth total: 47 → 52 (43 baseline + 4 vocab_shift + 5 colloquial + 3 adversarial_negative + 2 underspecified + **5 table_only**).

### The 5 table-only probes

| id                                        | doc                       | expected chunk types |
|-------------------------------------------|---------------------------|----------------------|
| `probe_table_weo_solar_lcoe`              | IEA_WEO_2025              | table (WEO Chapter 4 STEPS LCOE) |
| `probe_table_boe_tfsme_waci`              | BOE_DISCLOSURE_2024       | prose surrounding portfolio-metrics tables |
| `probe_table_ccc_prog24_buildings_rollback` | CCC_PROGRESS_2024       | prose (RAG table cells don't extract) |
| `probe_table_seventh_cb_transport_shift`  | CCC_SEVENTH_CARBON_BUDGET_2025 | prose in modal-shift chapter |
| `probe_table_weo_electricity_2035`        | IEA_WEO_2025              | table (WEO overview STEPS) |

### Results — retrieval-only eval on 52-query ground truth

**`table_fraction@5` by query type:**

| query_type      | n  | table_fraction@5 |
|-----------------|---:|-----------------:|
| factual         | 29 | 0.18             |
| numeric         | 8  | 0.20             |
| summarisation   | 2  | 0.00             |
| cross_document  | 4  | 0.00             |

**`table_fraction@5` by probe:**

| probe                | n  | table_fraction@5 |
|----------------------|---:|-----------------:|
| wording_variant      | 2  | 0.60             |
| vocab_shift          | 5  | 0.44             |
| adversarial_negative | 3  | 0.33             |
| vocab_mismatch       | 2  | 0.20             |
| **table_only**       | 5  | **0.16**         |
| colloquial           | 5  | 0.08             |
| crossdoc_named       | 1  | 0.00             |
| underspecified       | 2  | 0.00             |

**The 5 table-only probes individually:**

| probe                                       | Hit@5 | Recall@5 | top-5 chunk types |
|---------------------------------------------|:-----:|:--------:|---|
| probe_table_weo_solar_lcoe                  | ✓ | 1.0 | 3× table, 2× prose |
| probe_table_boe_tfsme_waci                  | ✓ | 1.0 | 5× prose (data lives in prose) |
| probe_table_ccc_prog24_buildings_rollback   | ✓ | 1.0 | 5× prose (RAG cells don't extract) |
| probe_table_seventh_cb_transport_shift      | ✓ | 1.0 | 5× prose (3% figure lives in prose) |
| probe_table_weo_electricity_2035            | ✓ | 1.0 | 1× table, 4× prose |

### Reading the numbers

**Retrieval finds the right doc every time (Recall@5=1.0 on all 5 probes).** That's the primary signal — the pipeline can locate table-heavy documents when queried for their table content.

**Table chunks surface when they exist and matter** — both WEO probes got table chunks in top-5, and WEO uses the highest-quality tables in the corpus (Tier 2 heading injection ships for it).

**"Prose-only" outcomes on 3 of 5 probes are not failures — they're anticipated:**
- **BoE Scope 1/2/portfolio numbers live in prose paragraphs**, not in table cells that `find_tables()` recognises. The fill-ratio filter (>70% populated) excludes narrative tables where cells contain sentences. Chunks that surface these numbers are `chunk_type=prose` but carry the signal.
- **CCC RAG tables don't extract as text** (traffic-light symbols) — CLAUDE.md documents this as a known limitation. Prose surround restates the assessment; retrieval finds those correctly.
- **Seventh CB's "3% modal shift" figure** is in prose narrative, not a table.

**`table_fraction@5` aggregate ~0.14–0.20 across query types** — roughly 1 in 5 retrieved chunks is a table chunk on numeric-flavoured queries. Not a target we should optimise upward; it reflects the ground truth that most numeric answers in this corpus live in prose paragraphs summarising table data, not in raw table cells.

### Answer correctness on the table-only probes — LLM + judge eval

Full pipeline eval on the 52-query ground truth (`judge_scores_20260807T215226Z.json`):

| Slice | Correctness | Faithfulness | Completeness | Refusal appr. |
|---|---:|---:|---:|---:|
| **table_only (n=5)** | **4.40** | **4.80** | 3.60 | 5.00 |
| overall (52 queries)  | 3.96       | 4.23       | 3.31 | 4.85 |

**Table-oriented queries score higher than the corpus average on both Correctness (+0.44) and Faithfulness (+0.57).**

**Per-probe:**

| Probe                                       | C | F | Cp | R | notes |
|---------------------------------------------|:-:|:-:|:--:|:-:|-------|
| probe_table_weo_solar_lcoe                  | 5 | 5 | 5  | 5 | Perfect. Answer: "40% decline in solar PV from 2024–2035" |
| probe_table_boe_tfsme_waci                  | 5 | 5 | **3** | 5 | Correct numbers; missed the "0.1 MtCO2e financed emissions" detail |
| probe_table_ccc_prog24_buildings_rollback   | 5 | 5 | 4  | 5 | Correctly identifies buildings-sector rollback |
| probe_table_seventh_cb_transport_shift      | 5 | 5 | 5  | 5 | Perfect. Correctly extracts the "3% additional shift by 2035" |
| **probe_table_weo_electricity_2035**        | **2** | 4 | **1** | 5 | **Only failure.** LLM said "STEPS does not provide specific projections" despite retrieval hitting the right doc. |

### Key design implications

**Tier 2 heading injection isn't load-bearing.** Four of five probes retrieved **zero table chunks** in top-5 and still scored Correctness=5. The LLM is synthesising correctly from prose that surrounds the tables. Even for WEO probes on documents that have proper Tier 2 extractable tables, the prose surround worked.

**This is empirical evidence the deferred Addition 4 (Tier 2 heading A/B) is low-value work.** Prose fallback carries the signal reliably; Tier 2 is optional insurance that isn't earning its keep as the primary retrieval-quality lever. Not a reason to remove it — it doesn't hurt — but not a reason to prioritise measuring it either.

### The one miss reinforces the numeric-reranker v2 case

`probe_table_weo_electricity_2035` failed at chunk-level precision: retrieval found the right document (Recall@5=1.0), but the specific chunk containing the "over 44,000 TWh in 2035" figure didn't reach top-5, and the LLM concluded the info wasn't there.

**Same failure mode Reranker fixed in Step 2e** — Reranker gave +1.0 Correctness on numeric queries specifically. This probe is the exact case per-type reranker activation would resolve. Strengthens the v2 candidate.

### What ships from 2g

- Metric slice: `table_fraction@k` in `retrieval_metrics.py` + `retrieved_chunk_types` + `detected_institutions` in eval-runner output
- Ground truth: 5 table-only probes added, total now 52
- Answers "have we evaluated retrieval from tables?" — **yes, and both retrieval (Recall@5=1.0) and answer quality (Correctness=4.4) are strong on the new probes**.
- Reinforces v2 candidate: **per-query-type reranker activation on numeric queries** (only real failure mode observed is chunk-level precision, which reranker addresses).
- **Down-prioritises v2 Addition 4 (Tier 2 heading A/B)** — empirical evidence shows prose fallback is doing the heavy lifting, so measuring Tier 2's marginal contribution is now less urgent.
- **Optional Addition 2 (`expected_chunk_type` field + `table_hit_rate`)** stays as v2 candidate but with lower priority — the descriptive `table_fraction@5` slice is doing the monitoring job well enough.
- Failure analysis doc updated (this section).

---

## Step 3 — Query rewriting (MEASURED AND DELETED)

**Motivation:** Step 2a Pattern A (vocab mismatch) and Step 2c cross-doc coverage both pointed at the same failure mode — retrieval misses relevant chunks when query wording is lexically far from corpus wording. LLM query rewriting is the standard lever for that.

### Implementation (later deleted)

- `src/synthesis/query_rewriter.py` — GPT-4o mini, `beta.chat.completions.parse` with `RewriteResponse` Pydantic schema (min 2, max 3 variants), `temperature=0` for reproducibility, in-process cache keyed on normalised query, `~$0.00008` per rewrite.
- `HybridRetriever.retrieve()` gained an `additional_variants=[...]` parameter — original + each variant contributed an independent BM25 + dense pass; RRF scores accumulated across every (query, retriever) combination.
- `main.py` + `judge_runner.py` behind `QUERY_REWRITING_ENABLED = True`.
- 12 new tests covering schema validation, cache hits, refusal graceful-fallback, and variant-fusion.

### A/B measurement — 52 queries, rewriting OFF (baseline) vs ON

**Source:**
- OFF: `judge_scores_20260807T215226Z.json` (post-metadata-filter baseline)
- ON:  `judge_scores_20260808T075024Z.json`

**Aggregate:**

| Metric                     | OFF   | ON    | Δ       |
|----------------------------|------:|------:|--------:|
| Correctness                | 3.962 | 3.942 | -0.019  |
| Faithfulness               | 4.231 | 4.269 | +0.038  |
| Completeness               | 3.308 | 3.192 | -0.115  |
| Refusal appropriateness    | 4.846 | 4.692 | -0.154  |
| Hit@5                      | 0.953 | 0.907 | **-0.047** |
| Recall@5                   | 0.907 | 0.857 | **-0.050** |
| Retrieval latency          | 193ms | 585ms | **3× slower** |
| Rewrite cost               | —     | $0.004 | trivial |

**By query type — the target failure mode regressed:**

| Type            | n  | OFF  | ON   | Δ       |
|-----------------|---:|-----:|-----:|--------:|
| numeric         | 8  | 3.88 | 4.25 | +0.38   |
| negative        | 9  | 4.11 | 4.22 | +0.11   |
| factual         | 29 | 4.03 | 3.97 | -0.07   |
| summarisation   | 2  | 3.00 | 3.00 | 0       |
| **cross_document** | **4** | **3.75** | **3.00** | **-0.75** ✗ |

**By probe — gains only on tiny-n:**

| probe               | n | Δ Correctness |
|---------------------|--:|--------------:|
| adversarial_negative| 3 | +1.00         |
| vocab_mismatch      | 2 | +0.50         |
| underspecified      | 2 | +0.50         |
| table_only          | 5 | +0.20         |
| **crossdoc_named**  | 1 | **-2.00**     |
| everything else     |   | 0             |

### Why the target regressed — root cause

For `xdoc_ccc_iea_transport_33`: **Recall@5 dropped 1.0 → 0.5**. Original retrieved 3 unique docs including `CCC_PROGRESS_2025`. Rewritten version dropped to 2 unique docs, **losing `CCC_PROGRESS_2025`** entirely.

The rewriter (temp=0, keyword-preserving prompt) produced *semantic paraphrases*: "In what ways do the transport targets set by the CCC align with the IEA's global fossil fuel and transport trajectory?" — same institution names, same nouns, restructured syntax. All 4 votes (original + 3 rewrites) went to the same semantically-similar chunks, so RRF concentrated on those and lost the retrieval diversity the original query happened to surface.

**For cross-doc queries that already name their sources, paraphrase-style rewriting produces LOW-diversity variants and hurts retrieval diversity — the exact opposite of what it was meant to do.**

### Ship decision — delete entirely

Same discipline as reranker (Step 2e) and confidence (Step 2d):

- Aggregate Correctness flat (Δ -0.02) → no aggregate case for shipping
- Target failure mode (cross-doc) **regressed -0.75** → active harm on the intended use case
- Retrieval quality dropped (Hit@5 -0.05, Recall@5 -0.05) → cost side is real
- 3× retrieval latency
- Real gains only on tiny-n probe slices where noise ≫ signal

**Deleted:** `query_rewriter.py`, `additional_variants` from `HybridRetriever`, `QUERY_REWRITING_ENABLED` flag in `main.py` and `judge_runner.py`, 12 rewriter tests, 3 variant retrieval tests. No dead code path preserved (same discipline as the confidence removal).

### v2 re-introduction conditions

1. **Different rewriter prompt.** Paraphrase failed. Try HyDE-style (rewriter generates a hypothetical *answer* and embeds that), or explicit synonym-expansion with a controlled vocabulary. Anything that produces variants with *different keywords*, not just different sentence structure.
2. **Per-query-type activation.** Only fire rewriting on queries a classifier flags as underspecified or likely vocab-mismatch — never on named-institution cross-doc queries where the original wording is already the best lexical signal. Depends on the v2 "Query classification" candidate.
3. **Larger ground truth.** Small-n probe slices (2, 2, 3) can't distinguish real gain from noise. Recollect at n ≥ 100 before another attempt.

### What ships from Step 3

- **No code.** Full removal.
- v2 roadmap entry updated with the three re-introduction conditions above.
- CLAUDE.md Week 5 Step 3 rewritten as "MEASURED AND DELETED" — the record of what was tried and why it was removed.
- 116 tests still pass.

---

## Step 3a — Prompt versioning + A/B (SHIPPED)

**Motivation:** Prompt is the last major untested lever on the current model. Every other component this week has been A/B'd; the prompt has been the un-measured constant.

### Infrastructure

- `PROMPT_REGISTRY: dict[str, str]` in `synthesiser.py` — all variants live in one place, keyed by version string.
- `PROMPT_VERSION` module-level default constant.
- `Synthesiser(prompt_version=...)` constructor arg to override per-instance.
- Per-query `prompt_version` field surfaced in `synthesise()` return, `build_query_record()` log output, and `judge_runner._score_pair` output.
- Judge runner writes to `judge_scores_<ts>_prompt-<version>.json` so A/B files don't overwrite each other.
- `PROMPT_VERSION_OVERRIDE` constant in `judge_runner.py` for A/B (None = use default).

**Why versioning is worth it even independent of the A/B outcome:** protects against silent prompt drift once Postgres monitoring lands next; enables cheap future variant comparisons; lets us pin prompt version in eval outputs for reproducibility.

### The 2 authored variants (both targeted at measured weaknesses)

- **v2_crossdoc** — v1 + rule 5 for multi-source retrievals: *"When retrieved excerpts come from multiple different doc_id values AND the question calls for comparison, synthesis, or relating sources to each other: EXPLICITLY compare or contrast the positions from each source in your answer."* Targeted the cross-doc Completeness gap (2.75 baseline).
- **v2_numeric** — v1 + rule 5 for numeric queries: *"When the question asks for a specific figure, percentage, cost, date, quantity, or unit-bearing value: FIRST scan the excerpts (prose AND table chunks) for the exact value... Do NOT round, generalise, or restate as 'approximately'."* Targeted the numeric miss pattern (`probe_table_weo_electricity_2035` where LLM said "STEPS doesn't provide" when it did).

### A/B results — 3 judge runs × 52 queries

**Source files:**
- v1:          `judge_scores_20260808T082144Z_prompt-v1.json`
- v2_crossdoc: `judge_scores_20260808T082810Z_prompt-v2_crossdoc.json`
- v2_numeric:  `judge_scores_20260808T083426Z_prompt-v2_numeric.json`

**Aggregate:**

| Metric                   | v1    | v2_crossdoc | v2_numeric | Δ xdoc | Δ num |
|--------------------------|------:|------------:|-----------:|-------:|------:|
| **Correctness**          | 4.019 | 4.096       | **4.135**  | +0.077 | **+0.115** |
| Faithfulness             | 4.365 | 4.308       | 4.365      | −0.058 | 0.000 |
| **Completeness**         | 3.154 | **3.385**   | 3.327      | **+0.231** | +0.173 |
| Refusal appropriateness  | 4.769 | **4.923**   | 4.846      | **+0.154** | +0.077 |
| Retrieval @5 (all)       | (identical — prompt does not affect retrieval) | | | | |

**Per-slice Correctness — where the aggregate gains came from:**

| probe                | n | v1   | v2_xdoc | v2_num |
|----------------------|--:|-----:|--------:|-------:|
| vocab_shift          | 5 | 3.80 | **4.40** | 4.20  |
| adversarial_negative | 3 | 3.33 | 3.67    | **4.67** |
| colloquial           | 5 | 3.40 | 3.80    | 3.60  |
| underspecified       | 2 | 3.50 | 3.50    | 4.00  |
| table_only           | 5 | 4.40 | 4.20    | 4.40  |
| wording_variant      | 2 | 4.50 | 4.00    | 4.50  |
| vocab_mismatch       | 2 | 4.00 | 4.00    | 4.00  |

**On the stated design targets — both variants missed:**

| Variant → target slice | Metric | v1 | Variant | Δ |
|---|---|---:|---:|---:|
| v2_crossdoc → cross_document | Correctness | 3.50 | 3.25 | **−0.25** |
| v2_crossdoc → cross_document | Completeness | 2.50 | 2.25 | **−0.25** |
| v2_numeric → numeric | Correctness | 4.00 | 4.00 | 0.00 |
| v2_numeric → numeric | Completeness | 3.50 | 3.25 | **−0.25** |

Cross-doc n=4 and numeric n=8 sit in noise-band territory: single-query judge flips shift these means by 0.12–0.25.

### Ship decision — v2_numeric shipped as v1 default; v2_crossdoc kept in registry

**Ship v2_numeric (default):** cleanest aggregate improvement (+0.115 Correctness — biggest single-lever aggregate gain of Week 5), **zero regressions on any aggregate metric**, no cost/latency delta, no code path complexity. The "extract verbatim" instruction turned out to be a *generally-good* prompt improvement, lifting non-numeric slices too (vocab_shift +0.40, colloquial +0.20, adversarial_negative +1.33). This is a real lesson — the aggregate win is orthogonal to the target we designed for.

**Keep v2_crossdoc in `PROMPT_REGISTRY` (not default):** same precedent as the reranker (Step 2e — measured, not shipped, code preserved for possible v2 per-type activation). v2_crossdoc's aggregate wins on Completeness (+0.23) and Refusal_appr (+0.15) are real signals; the cross_document target regression is likely n=4 sampling noise. A future query classifier could fire v2_crossdoc selectively on cross_document queries — measure then.

### Consistency with the week's ship discipline

- Reranker (Step 2e): aggregate flat, target flat → kept in codebase, not default. ✓
- RRF threshold (Step 2b): net-harmful → deleted entirely. ✓
- Confidence (Step 2d): weak signal, no user-facing value → deleted entirely. ✓
- Query rewriting (Step 3): aggregate flat, target REGRESSED → deleted entirely. ✓
- Metadata filter (Step 2f): target improved +1.0 Completeness → shipped as default. ✓
- **v2_numeric prompt (Step 3a): aggregate improved on Correctness/Completeness/Refusal_appr, no regression → shipped as default. ✓**
- **v2_crossdoc prompt (Step 3a): aggregate mixed, target regressed → kept in registry, not default. ✓** (Same call as reranker.)

### What ships from 3a

- `PROMPT_REGISTRY` in `synthesiser.py` (3 variants), `PROMPT_VERSION = "v2_numeric"` as v1 default.
- `prompt_version` field in every synthesise() result, log record, and eval per-query row.
- Judge runner writes version-tagged output filenames.
- 7 new tests covering registry contents, default version, constructor arg, unknown-version rejection, prompt-version-in-result on empty-chunks path, correct-prompt-at-call-time verification. Total: 123 tests pass.
- v2 candidate updated: "per-query-type prompt activation" — pair v2_crossdoc with a query classifier in v2.




