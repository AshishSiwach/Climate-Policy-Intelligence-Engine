# Week 3 — E2E Validation Observations

**Date:** 2026-07-13
**Author:** Ashish Siwach
**Scope:** Full v1 pipeline validation before Week 3 submission
**Pipeline:** `query → BM25 + Dense → RRF (k=60) → top-5 → GPT-4o mini → JSON brief`

---

## Purpose of this document

These are **observations**, not bugs to fix in Week 3. The 5-query test was an E2E plumbing check, not a tuning signal — 5 designer-picked queries have no statistical meaning.

The failure patterns below become **inputs to Week 4 ground truth writing** and **Week 5 failure analysis**. Any prompt change, threshold change, or architectural change must be validated against 35–50 ground truth queries with LLM-as-judge scoring — not proposed from 5 anecdotes.

## System framing (informs how we interpret failures)

CPIE v1 is a **single-turn domain RAG system**, not a deep research system. It aspires to deep-research-quality answers (verified citations, pipeline confidence, contradiction detection) through a simpler architecture. This framing matters:

- Failures are single-turn-RAG failure modes (prompt calibration, threshold calibration, top-k coverage)
- We do NOT propose deep-research fixes (agentic loops, query decomposition) — those are v2/v3 roadmap items

---

## Pipeline plumbing — all 5 checks pass

| Check | Result |
|---|---|
| All modules wire together end-to-end | Pass |
| Retrieval feeds synthesis correctly | Pass |
| Citation verification runs and drops unverified passages | Pass |
| Out-of-corpus short-circuit fires (Q5: sourdough) | Pass |
| JSONL log captures all required fields per query | Pass |

The v1 pipeline is complete. Ship for Week 3 submission.

---

## Observed failure patterns (inputs to Week 4/5, not v1 fixes)

### Pattern A — LLM refuses despite good retrieval

**Observed on:** Q2 (CBES aggregate losses), Q4 (CCC vs IEA cross-doc)

**Data:**
| Query | Top RRF | Top-5 docs | Confidence | Signals |
|---|---|---|---|---|
| Q2 CBES losses | 0.032 | 5× BOE_CBES_RESULTS_2021 | 0.614 | score=0.64, agreement=0.98, cite=0.00 |
| Q4 CCC vs IEA | 0.029 | CCC×3 + DESNZ×1 (no IEA) | 0.506 | score=0.59, agreement=0.71, cite=0.00 |

**Trace:**
- LLM was called (~2100 prompt tokens each)
- Completion was 32 tokens — exact length of "The corpus does not contain sufficient information to answer this query."
- LLM chose the canonical refusal even though retrieval surfaced on-topic chunks
- Q2: retrieval was on the right document but chunks may not have used the exact phrases "aggregate losses" or "early action scenario"
- Q4: retrieval only surfaced CCC, no IEA — genuinely can't compare with what's on the desk

**Two sub-patterns to disentangle in Week 5:**
1. **Vocabulary mismatch** (Q2): LLM too literal — refuses when chunk phrasing differs from query terminology. Prompt Rule #4 (system prompt) may be too easy to trigger.
2. **Genuine coverage gap** (Q4): cross-doc query where top-5 doesn't include all required sources. Arguably correct refusal behaviour, but retrieval strategy is limiting.

**What Week 4/5 answers:** Is Pattern A systematic (>10% of factual queries refuse) or query-specific? Only ground truth reveals.

**What to write into ground truth to probe this:**
- Multiple CBES-style queries where chunk phrasing differs from query terminology
- Multiple cross-doc queries with expected sources named explicitly
- Vocabulary-shift variants: same intent, different words

---

### Pattern B — out-of-corpus threshold marginal on legitimate hits

**Observed on:** Q3 (IEA peak fossil fuel demand — with "global" modifier)

**Data:**
| Query variant | Top RRF | Result |
|---|---|---|
| "peak fossil fuel demand" (Step 3) | 0.031 | Passed to LLM, correct answer |
| "peak of *global* fossil fuel demand" (Q3) | 0.016 | Short-circuited, "corpus does not contain" |

**Trace:**
- Same target document (IEA_WEO_2025) was in top-5 for both variants
- Adding "global" moved score just below the 0.020 threshold
- Threshold triggered even though retrieval was correct

**Root observation:** The 0.020 threshold was picked from ablation on 3 queries — it works on average but has no calibration against a real distribution of query rewordings.

**What Week 4/5 answers:** What is the actual score distribution for real hits vs out-of-corpus queries across 35–50 examples? Threshold should be set at the elbow, not by 3-query gut-feel.

**What to write into ground truth to probe this:**
- Wording variants of the same underlying question (adds/removes modifiers like "global", "aggregate", "recent")
- Queries in question form vs statement form
- Queries with and without institution names

---

### Pattern C — cross-document queries constrained by fixed top-5

**Observed on:** Q4 (CCC vs IEA cross-doc)

**Data:**
- Top-5 all from CCC + DESNZ, no IEA
- Query explicitly names both CCC and IEA, but retrieval couldn't surface both
- LLM refused (correct behaviour given missing source)

**Root observation:** Cross-doc queries need at least one hit per named source. Fixed top-5 doesn't guarantee coverage across multiple sources.

**What Week 4/5 answers:** How common are cross-doc queries in the ground truth? If >5 of 50, is this a systematic weakness worth addressing (e.g., higher top-k, or per-institution retrieval mode)?

**What to write into ground truth to probe this:**
- CCC vs IEA comparison queries (multiple)
- Ofgem vs DESNZ policy alignment queries
- Cross-institutional queries with 3+ named sources

---

## What NOT to do

Per this session's core principle — measure before you change:

- **Do NOT tune the system prompt** based on Q2 refusal
- **Do NOT lower the 0.020 threshold** based on Q3
- **Do NOT expand top-k from 5 to 10** based on Q4
- **Do NOT add an agent loop** or query decomposition — v2/v3 territory

Each of these might turn out to be the right fix. None is justified by 5 queries.

---

## Handoff to Week 4

When writing the 35–50 ground truth QA dataset, include queries specifically designed to reveal:
- Whether Pattern A (vocabulary-mismatch refusal) is common
- Where the 0.020 threshold sits relative to the true hit distribution
- How much of the corpus a typical cross-doc question needs

## Handoff to Week 5

Failure analysis prompts to look for:
- Refusal rate by query type (factual / cross-doc / negative)
- Threshold precision/recall on out-of-corpus classification
- Top-k coverage for cross-doc queries

Any proposed fix ships with A/B evidence: "we changed X, error rate went from Y to Z on the same ground truth queries."

---

## Known v1 limitation — narrative hallucination (not caught by citation verification)

**What's caught today:** `_verify_citations` compares each cited `passage` against retrieved chunks. If the LLM invents a quote that doesn't exist, we drop the citation.

**What's NOT caught:** the LLM's *narrative* (the un-quoted answer text) can contradict the cited passage. Concrete failure mode:

- LLM retrieves chunk containing: *"Ofgem intends to begin accepting licence applications from the end of 2026."*
- LLM answer writes: *"Ofgem will begin accepting applications from December 2025."* — wrong date
- LLM cites the chunk verbatim → citation verification PASSES
- Wrong answer text ships to user with a valid-looking citation

Citation verification checks quoted material, not answer faithfulness. This is the classic "faithful synthesis" problem for RAG.

**Why not fixed in v1:** requires either (a) a second LLM call to verify answer↔chunk faithfulness (doubles cost + latency), or (b) an NLI (natural language inference) model checking each answer sentence against retrieved chunks. Both are architectural additions worth measuring before adopting.

**Week 5 mitigation:** LLM-as-judge on ground truth QA will detect this at eval time. Set up the judge rubric to specifically score "answer faithfulness to retrieved chunks" separately from "answer quality vs reference." If eval shows > 5% narrative-hallucination rate, add a faithfulness check to v1.1.

**Do NOT catch this at inference time in v1.** Log it as a v2 candidate in the roadmap.

---

## Log record for this validation

All 5 queries logged to `logs/queries.jsonl` on 2026-07-13. Fields per record:
`timestamp, query, retrieved_doc_ids, retrieved_pages, rrf_scores, retrieval_latency_ms, synthesis_latency_ms, confidence, confidence_signals, model_used, prompt_tokens, completion_tokens, cost_usd, citation_count, contradiction_count, failure_reason`
