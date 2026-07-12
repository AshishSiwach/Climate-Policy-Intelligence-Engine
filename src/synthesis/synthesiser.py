"""
Synthesis layer — GPT-4o mini structured JSON output.

Input:  query + top-k retrieved chunks (from HybridRetriever)
Output: AnalystBrief (validated Pydantic model)

Key design decisions (see CLAUDE.md):
  - Confidence is pipeline-derived (RRF scores + spread + citation count),
    NOT self-assessed by the LLM.
  - Citations are verified against retrieved chunks after synthesis;
    fabricated ones are dropped.
  - Table chunks are flagged in the prompt so the LLM extracts values
    rather than paraphrasing.
  - Out-of-corpus queries: if top RRF score below threshold, return
    confidence=0.0 without calling the LLM.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from synthesis.output_schema import AnalystBrief, Citation, LLMResponse

logger = logging.getLogger(__name__)

MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_TOKENS = 800
OUT_OF_CORPUS_RRF_THRESHOLD = 0.020   # empirical: real hits score ~0.03+, out-of-corpus <0.02

OUT_OF_CORPUS_ANSWER = (
    "The corpus does not contain sufficient information to answer this query."
)

SYSTEM_PROMPT = """You are a climate policy research analyst. You answer questions using ONLY the retrieved excerpts provided.

Rules:
1. Every factual claim in your answer MUST be supported by a citation. Never invent citations.
2. Quote verbatim from the excerpts — do not paraphrase quoted material inside a citation's `passage` field.
3. Chunks marked `[chunk_type: table]` contain tabular data. Extract specific values and units; do not paraphrase.
4. If the excerpts do not answer the question, respond with the answer field set to exactly: "The corpus does not contain sufficient information to answer this query." — and leave `citations` empty.
5. Contradictions between excerpts: only report if two excerpts make directly opposing factual claims. Otherwise leave `contradictions` empty. This is experimental — err on the side of not flagging.

Return JSON matching this exact schema:
{
  "answer": "<direct response>",
  "citations": [
    {"doc_id": "<source doc_id>", "passage": "<verbatim quote>", "page": <int>}
  ],
  "contradictions": [
    {"doc_a": "<doc_id>", "doc_b": "<doc_id>", "summary": "<one-line description>"}
  ]
}"""


class Synthesiser:
    """Turns retrieved chunks into a validated AnalystBrief."""

    def __init__(
        self,
        model: str = MODEL,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def synthesise(self, query: str, chunks: list[dict]) -> dict[str, Any]:
        """
        Run synthesis over the top-k retrieved chunks.

        Returns a dict with:
          brief          — AnalystBrief (validated)
          latency_ms     — LLM call latency
          prompt_tokens  — usage.prompt_tokens
          completion_tokens — usage.completion_tokens
          cost_usd       — estimated cost for this call
        """
        # Out-of-corpus short-circuit — don't spend an LLM call
        if not chunks or max(c.get("rrf_score", 0.0) for c in chunks) < OUT_OF_CORPUS_RRF_THRESHOLD:
            logger.info("Out-of-corpus query — top RRF below threshold, skipping LLM call")
            return {
                "brief": AnalystBrief(
                    answer=OUT_OF_CORPUS_ANSWER,
                    citations=[],
                    confidence=0.0,
                    contradictions=[],
                ),
                "latency_ms": 0.0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cost_usd": 0.0,
                "confidence_signals": {
                    "score_signal": 0.0,
                    "spread_signal": 0.0,
                    "citation_signal": 0.0,
                    "out_of_corpus": True,
                },
            }

        context = _format_context(chunks)
        user_msg = f"Question: {query}\n\nRetrieved excerpts:\n{context}"

        t0 = time.time()
        response = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        latency_ms = (time.time() - t0) * 1000

        raw_json = response.choices[0].message.content or "{}"
        llm_response = LLMResponse.model_validate_json(raw_json)

        # Verify citations against retrieved chunks — drop fabricated ones
        verified_citations = _verify_citations(llm_response.citations, chunks)

        # Pipeline-derived confidence (with individual signals for Week 5 calibration)
        confidence, signals = _compute_confidence(chunks, verified_citations)

        brief = AnalystBrief(
            answer=llm_response.answer,
            citations=verified_citations,
            confidence=confidence,
            contradictions=llm_response.contradictions,
        )

        usage = response.usage
        cost_usd = _estimate_cost(
            self.model, usage.prompt_tokens, usage.completion_tokens
        )

        logger.info(
            "Synthesis: %.0fms, %d prompt + %d completion tokens, $%.5f",
            latency_ms, usage.prompt_tokens, usage.completion_tokens, cost_usd,
        )

        return {
            "brief": brief,
            "latency_ms": latency_ms,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": cost_usd,
            "confidence_signals": signals,   # log for Week 5 calibration
        }


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a compact, LLM-friendly context block."""
    lines = []
    for i, c in enumerate(chunks, 1):
        header = (
            f"[Excerpt {i}] doc_id={c['doc_id']}  page={c['page_number']}  "
            f"chunk_type={c.get('chunk_type', 'prose')}"
        )
        lines.append(header)
        lines.append(c["text"].strip())
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Citation verification
# ---------------------------------------------------------------------------

def _verify_citations(citations: list[Citation], chunks: list[dict]) -> list[Citation]:
    """
    Drop citations whose passage does not appear in any retrieved chunk.

    Comparison is case-insensitive with whitespace normalised. Fabricated
    citations (LLM invented a passage) get dropped; genuine ones survive.
    """
    verified = []
    normalised_chunks = [_normalise(c["text"]) for c in chunks]

    for cit in citations:
        target = _normalise(cit.passage)
        if not target:
            continue

        # Anchor on the first 60 chars of the normalised passage — long enough
        # to be distinctive, short enough that minor extraction noise doesn't
        # break the match.
        anchor = target[:60]
        if any(anchor in chunk_text for chunk_text in normalised_chunks):
            verified.append(cit)
        else:
            logger.info(
                "Dropped unverified citation: doc_id=%s page=%d passage=%r",
                cit.doc_id, cit.page, cit.passage[:80],
            )

    return verified


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Pipeline-derived confidence
# ---------------------------------------------------------------------------

def _compute_confidence(
    chunks: list[dict], citations: list[Citation]
) -> tuple[float, dict[str, float]]:
    """
    Combine three signals into a [0, 1] confidence score AND return them
    individually for downstream logging.

      score_signal    — top RRF score, mapped to [0, 1]
      spread_signal   — how clustered top scores are (higher = agree, lower = scattered)
      citation_signal — how many chunks the answer actually leans on

    Weights (0.5 / 0.3 / 0.2) are v1 placeholders — gut-feel, not empirically
    calibrated. Individual signals are logged so Week 5 can fit real weights
    against ground truth LLM-as-judge scores via logistic regression.
    """
    if not chunks:
        return 0.0, {
            "score_signal": 0.0,
            "spread_signal": 0.0,
            "citation_signal": 0.0,
            "out_of_corpus": False,
        }

    rrf_scores = [c.get("rrf_score", 0.0) for c in chunks]
    top = rrf_scores[0]

    # score_signal — top RRF above the out-of-corpus floor is a good sign.
    # Real hits typically 0.03; strong hits 0.05+. Clamp above 0.05.
    score_signal = min(top / 0.05, 1.0)

    # spread_signal — mean of top-3 relative to top-1. If top-3 are tightly
    # clustered, spread ≈ 1 (retrievers agree). If top-1 is a lone outlier,
    # spread drops.
    top3 = rrf_scores[:3]
    spread_signal = (sum(top3) / len(top3)) / top if top > 0 else 0.0

    # citation_signal — count of verified citations, capped at 3 for full credit.
    citation_signal = min(len(citations) / 3.0, 1.0)

    confidence = 0.5 * score_signal + 0.3 * spread_signal + 0.2 * citation_signal
    confidence = round(max(0.0, min(1.0, confidence)), 3)

    signals = {
        "score_signal": round(score_signal, 3),
        "spread_signal": round(spread_signal, 3),
        "citation_signal": round(citation_signal, 3),
        "out_of_corpus": False,
    }
    return confidence, signals


# ---------------------------------------------------------------------------
# Cost estimation
# ---------------------------------------------------------------------------

# GPT-4o mini pricing per 1M tokens (as of 2026)
_PRICING = {
    "gpt-4o-mini": (0.15, 0.60),   # (input, output) per 1M tokens
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    inp, out = _PRICING.get(model, (0.0, 0.0))
    return round(
        (prompt_tokens / 1_000_000) * inp + (completion_tokens / 1_000_000) * out,
        6,
    )
