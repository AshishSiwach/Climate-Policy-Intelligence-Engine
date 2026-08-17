"""
Query rewriter — GPT-4o-mini paraphrase variants for retrieval augmentation.

Built and A/B-measured against the 52-query eval set. Results vs plain query:
  - Cross-doc Correctness  : −0.75
  - Recall@5               : −5pp
  - Retrieval latency       : +3×

Not active in the live pipeline — plain queries outperform rewrites on this
corpus because policy text uses the exact terminology analysts already know.
See docs/week5_failure_analysis.md for full ablation data.
"""

from __future__ import annotations

import logging
import os

from openai import OpenAI

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SYSTEM = (
    "You rewrite user queries to improve retrieval from a climate-policy document corpus. "
    "Return exactly one improved query on a single line. "
    "Keep the same intent. Do not add information not in the original query. "
    "Do not include explanation or commentary."
)


def rewrite_query(query: str, api_key: str | None = None) -> str:
    """Return a rewritten version of *query* for better retrieval coverage.

    Fails open: returns the original query unchanged on any error so the
    pipeline is never blocked by a rewriter failure.

    Args:
        query: The raw user query.
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.

    Returns:
        Rewritten query string, or the original if rewriting fails.
    """
    try:
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=10.0,
            max_retries=1,
        )
        response = client.chat.completions.create(
            model=_MODEL,
            temperature=0.3,
            max_tokens=120,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": query},
            ],
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten:
            logger.debug("Query rewritten: %r → %r", query, rewritten)
            return rewritten
    except Exception as exc:
        logger.warning("Query rewriter failed (%s) — using original query", exc)
    return query


def rewrite_variants(query: str, n: int = 3, api_key: str | None = None) -> list[str]:
    """Return *n* paraphrase variants of *query* plus the original.

    Used during A/B evaluation to test retrieval breadth across multiple
    phrasings. In production the extra LLM calls add ~3× latency with no
    measurable Recall gain on this corpus.

    Args:
        query: The raw user query.
        n: Number of variants to generate (excluding the original).
        api_key: OpenAI API key. Falls back to ``OPENAI_API_KEY`` env var.

    Returns:
        List starting with the original query, followed by up to *n* variants.
        Deduplication preserves order. On error, returns ``[query]``.
    """
    try:
        client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=15.0,
            max_retries=1,
        )
        system = (
            f"You rewrite user queries to improve retrieval from a climate-policy corpus. "
            f"Return exactly {n} alternative phrasings, one per line, no numbering. "
            f"Preserve intent. Do not add information not present in the original."
        )
        response = client.chat.completions.create(
            model=_MODEL,
            temperature=0.7,
            max_tokens=300,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
        )
        lines = [
            line.strip()
            for line in response.choices[0].message.content.splitlines()
            if line.strip()
        ]
        variants = [query] + lines[:n]
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return unique
    except Exception as exc:
        logger.warning("Query rewrite_variants failed (%s) — returning original", exc)
        return [query]
