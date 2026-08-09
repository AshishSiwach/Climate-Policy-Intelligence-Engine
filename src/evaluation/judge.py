"""
LLM-as-judge harness.

Scores a generated AnalystBrief against a ground-truth reference on four
dimensions, each 1-5 with a short rationale:

  correctness           — factual claims match reference
  faithfulness          — answer supported by retrieved chunks (not fabricated)
  completeness          — covers all key points from reference
  refusal_appropriateness — refused when it should have, answered when it should have

Uses OpenAI Structured Outputs (client.beta.chat.completions.parse) with a
Pydantic schema so the judge's output cannot deviate from the rubric.

Model is a parameter (default gpt-5.4-mini); if the model doesn't exist or
doesn't support Structured Outputs, swap via constructor arg.

**Same-model bias caveat (Week 5 Step 3b):** synthesis was upgraded to
gpt-5.4-mini as v1 default; the judge is also gpt-5.4-mini. Same-model judge
introduces ~5% inflation in favor of the same-model answers per published
studies. Accepted for regular eval runs because judge cost is 5× cheaper on
gpt-5.4-mini than gpt-5.4, and typical eval-signal band (>0.10 Correctness)
exceeds the bias band. For rigorous audits or cross-vendor validation,
temporarily flip to `JUDGE_MODEL = "gpt-5.4"` (~5× cost per run) or use a
Claude judge — pattern documented in docs/week5_failure_analysis.md § 3b.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Placeholder defaults — override via constructor.
JUDGE_MODEL = "gpt-5.4-mini"
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1200
JUDGE_TIMEOUT_SEC = 30.0
JUDGE_MAX_RETRIES = 2

# Per-1M-token prices as (input, output). Cached-input pricing exists but the
# chat-completions Usage object doesn't return cached-token breakdowns in v1;
# we count all input at the uncached rate — treat cost as an upper bound.
_JUDGE_PRICING = {
    "gpt-4o-mini":   (0.15, 0.60),
    "gpt-4o":        (2.50, 10.00),
    "gpt-5.4-mini":  (0.75, 4.50),   # cached input: $0.075/M (not tracked in v1)
    "gpt-5.4":       (2.50, 15.00),  # cached input: $0.25/M (not tracked in v1)
}


# ---------------------------------------------------------------------------
# Judge output schema
# ---------------------------------------------------------------------------

class JudgeScore(BaseModel):
    """Structured Outputs schema — what the judge model must return."""

    correctness: int = Field(..., ge=1, le=5, description="1-5 factual accuracy vs reference")
    correctness_rationale: str = Field(..., description="One-sentence justification")

    faithfulness: int = Field(..., ge=1, le=5, description="1-5 grounding in retrieved chunks")
    faithfulness_rationale: str = Field(..., description="One-sentence justification")

    completeness: int = Field(..., ge=1, le=5, description="1-5 coverage of reference key points")
    completeness_rationale: str = Field(..., description="One-sentence justification")

    refusal_appropriateness: int = Field(
        ..., ge=1, le=5,
        description="1-5 — did the system refuse when it should have, or answer when it should have"
    )
    refusal_appropriateness_rationale: str = Field(..., description="One-sentence justification")


# ---------------------------------------------------------------------------
# Judge system prompt (rubric)
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """You are an evaluator scoring a RAG system's answer against a ground-truth reference.

You will receive:
  QUESTION            — what was asked
  QUERY_TYPE          — factual / cross_document / summarisation / numeric / negative
  REFERENCE_ANSWER    — the human-authored ground truth
  RETRIEVED_CHUNKS    — what the retrieval layer surfaced (may be empty for short-circuited queries)
  GENERATED_ANSWER    — what the RAG system produced

Score four dimensions, each 1-5, with a one-sentence rationale for each.

1. CORRECTNESS — do the factual claims in GENERATED_ANSWER match REFERENCE_ANSWER?
   5 = all claims match
   4 = minor factual differences or missing minor details
   3 = some factual differences or key details missing
   2 = multiple factual errors or major details missing
   1 = largely wrong or contradicts the reference

2. FAITHFULNESS — is GENERATED_ANSWER grounded in RETRIEVED_CHUNKS (not fabricated)?
   5 = every claim traceable to a retrieved chunk
   4 = almost every claim traceable; minor extrapolation
   3 = some claims traceable, some plausible but unsupported
   2 = multiple unsupported claims
   1 = largely fabricated / not supported
   If GENERATED_ANSWER is the canonical refusal ("corpus does not contain..."), score 5.

3. COMPLETENESS — does GENERATED_ANSWER cover all key points in REFERENCE_ANSWER?
   5 = all key points covered
   4 = nearly all points covered, one minor missing
   3 = some key points covered, some missing
   2 = multiple key points missing
   1 = misses most of what the reference covers
   For negative queries where REFERENCE_ANSWER is a canonical refusal, score 5 if GENERATED_ANSWER is also a refusal.

4. REFUSAL_APPROPRIATENESS — did the system's refusal or non-refusal match what it should have done?
   For QUERY_TYPE == "negative":
     5 = system correctly refused ("corpus does not contain...")
     1 = system fabricated an answer instead of refusing
   For QUERY_TYPE == positive (factual/cross_document/summarisation/numeric):
     5 = system answered (no inappropriate refusal)
     3 = system refused but the corpus DID contain the information
     1 = system refused a query it should have answered
   For underspecified probes (marked in notes):
     5 = system hedged or returned low-confidence multi-source answer
     3 = system answered but with appropriate qualification
     1 = system answered confidently without qualification

Be strict but fair. Provide one sentence per rationale — no essays."""


# ---------------------------------------------------------------------------
# The judge
# ---------------------------------------------------------------------------

class LLMJudge:
    """Wraps a model call + rubric prompt into a scoring function."""

    def __init__(
        self,
        model: str = JUDGE_MODEL,
        temperature: float = JUDGE_TEMPERATURE,
        max_tokens: int = JUDGE_MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=JUDGE_TIMEOUT_SEC,
            max_retries=JUDGE_MAX_RETRIES,
        )

    def score(
        self,
        *,
        question: str,
        query_type: str,
        expected_answer: str,
        retrieved_chunks: list[dict],
        generated_answer: str,
    ) -> dict[str, Any]:
        """
        Score one RAG output. Returns dict with:
          scores           — JudgeScore (Pydantic)
          latency_ms       — LLM call latency
          prompt_tokens    — usage
          completion_tokens
          cost_usd         — 0 if pricing unknown (logs a warning once)
        """
        user_msg = _build_judge_message(
            question=question,
            query_type=query_type,
            expected_answer=expected_answer,
            retrieved_chunks=retrieved_chunks,
            generated_answer=generated_answer,
        )

        t0 = time.time()
        response = self._client.beta.chat.completions.parse(
            model=self.model,
            temperature=self.temperature,
            # `max_completion_tokens` is the current parameter name (newer models like
            # gpt-5.x require it; older models accept it as alias for max_tokens).
            max_completion_tokens=self.max_tokens,
            response_format=JudgeScore,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        latency_ms = (time.time() - t0) * 1000

        message = response.choices[0].message
        usage = response.usage

        if message.refusal:
            # Judge refused to score — treat as invalid, log for investigation.
            logger.warning("Judge refused to score: %s", message.refusal)
            return {
                "scores": None,
                "latency_ms": latency_ms,
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "cost_usd": _estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens),
                "refusal_reason": message.refusal,
            }

        scores: JudgeScore = message.parsed

        return {
            "scores": scores,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": _estimate_cost(self.model, usage.prompt_tokens, usage.completion_tokens),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_judge_message(
    *,
    question: str,
    query_type: str,
    expected_answer: str,
    retrieved_chunks: list[dict],
    generated_answer: str,
) -> str:
    """Render the judge's user-side message. Keep it compact — judge cost scales with prompt tokens."""
    chunk_block = "\n\n".join(
        f"[Chunk {i}] doc_id={c['doc_id']} page={c['page_number']}\n{c['text']}"
        for i, c in enumerate(retrieved_chunks, 1)
    ) or "(no chunks — pipeline short-circuited)"

    return (
        f"QUESTION:\n{question}\n\n"
        f"QUERY_TYPE: {query_type}\n\n"
        f"REFERENCE_ANSWER:\n{expected_answer}\n\n"
        f"RETRIEVED_CHUNKS:\n{chunk_block}\n\n"
        f"GENERATED_ANSWER:\n{generated_answer}"
    )


_pricing_warned: set[str] = set()


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate cost from _JUDGE_PRICING. Warn once per unknown model, return 0."""
    if model not in _JUDGE_PRICING:
        if model not in _pricing_warned:
            logger.warning(
                "No pricing for judge model %r — logged cost will be 0. "
                "Add to _JUDGE_PRICING when known.", model,
            )
            _pricing_warned.add(model)
        return 0.0
    inp, out = _JUDGE_PRICING[model]
    return round(
        (prompt_tokens / 1_000_000) * inp + (completion_tokens / 1_000_000) * out,
        6,
    )
