"""
Agent synthesiser node — Phase 2 Cross-Document Route.

DISTINCT from the fast-path src/synthesis/synthesiser.py.
Do NOT modify that file.

Composes the final AnalystBrief from verified claims.
One LLM call (GPT-4o-mini).
No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.
"""

from __future__ import annotations

import json
import logging
import os

from src.evidence.claims import Claim, Coverage
from src.synthesis.output_schema import AnalystBrief, Citation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a climate policy research analyst composing a final comparative analysis.

You will be given:
1. The original research question.
2. A list of verified factual claims extracted from retrieved documents, each with their source.

Your task: write a coherent, comparative analytical answer to the question using ONLY
the provided claims.
- Reference specific sources by their doc_id when drawing on them.
- If claims from different sources agree or disagree, say so explicitly.
- Do NOT introduce information not present in the claims.
- Be concise but complete: cover what the claims cover, and note any gaps.

Return a JSON object with this exact structure:
  {"answer": "...", "coverage_gaps": ["description of gap 1", ...]}

coverage_gaps should list sub-questions or topics that had only partial or no coverage.
Return ONLY the JSON object.
"""


def run_synthesiser(state: dict) -> dict:
    """Agent synthesiser node: compose AnalystBrief from verified claims.

    Input: state["verified_claims"], state["coverage"], state["query"]
    Output: {"result": AnalystBrief.model_dump(), "steps_used": state["steps_used"] + 1,
             "cost_used_usd": updated, "termination_reason": "complete"}
    """
    verified_claims: list = state.get("verified_claims", [])
    coverage: dict = state.get("coverage", {})
    query: str = state.get("query", "")
    retrievals: dict = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)
    cost_used_usd = state.get("cost_used_usd", 0.0)

    # Flatten all chunks for evidence_id → Citation resolution
    all_chunks: dict[str, dict] = {}
    for chunk_list in retrievals.values():
        for chunk in chunk_list:
            cid = chunk.get("chunk_id")
            if cid:
                all_chunks[cid] = chunk

    # Identify coverage gaps from grader output
    coverage_gaps: list[str] = [
        (cov.gap_reason or sq_id)
        for sq_id, cov in coverage.items()
        if isinstance(cov, Coverage) and cov.status in ("partial", "not_covered")
    ]

    # Build citations from verified claims' evidence_ids
    citations = _build_citations(verified_claims, all_chunks)

    # LLM call to compose the answer
    answer, llm_gaps, call_cost = _call_synthesiser(query, verified_claims, coverage_gaps)

    # Merge grader-detected gaps with LLM-identified gaps (deduplicate, preserve order)
    seen: set[str] = set(coverage_gaps)
    all_gaps = list(coverage_gaps)
    for g in llm_gaps:
        if g not in seen:
            seen.add(g)
            all_gaps.append(g)

    brief = AnalystBrief(
        answer=answer,
        citations=citations,
        coverage_gaps=all_gaps,
        contradictions=[],
        truncated=False,
        termination_reason="complete",
    )

    return {
        "result": brief.model_dump(),
        "steps_used": steps_used + 1,
        "cost_used_usd": cost_used_usd + call_cost,
        "termination_reason": "complete",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _call_synthesiser(
    query: str,
    claims: list,
    coverage_gaps: list[str],
) -> tuple[str, list[str], float]:
    """Call LLM to compose answer. Returns (answer_text, llm_coverage_gaps, cost_usd)."""
    try:
        from openai import OpenAI  # deferred import

        key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key)

        user_content = _build_prompt(query, claims, coverage_gaps)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.0,
        )

        raw = response.choices[0].message.content or "{}"
        usage = response.usage

        prompt_tokens = usage.prompt_tokens if usage else max(len(user_content) // 4, 1)
        completion_tokens = usage.completion_tokens if usage else max(len(raw) // 4, 1)
        cost = _estimate_cost(prompt_tokens, completion_tokens)

        parsed = json.loads(raw)
        answer = str(parsed.get("answer", "")).strip()
        llm_gaps = [str(g) for g in parsed.get("coverage_gaps", []) if g]

        if not answer:
            answer = "Insufficient verified evidence to compose an answer."

        return answer, llm_gaps, cost

    except Exception as exc:
        logger.warning("Agent synthesiser LLM call failed: %s", exc)
        return "Synthesis failed due to an internal error.", [], 0.0


def _build_prompt(query: str, claims: list, coverage_gaps: list[str]) -> str:
    lines = [f"Question: {query}\n", "Verified claims:"]
    for i, claim in enumerate(claims):
        if isinstance(claim, Claim):
            text = claim.text
            doc = claim.source_doc_id
            eids = ", ".join(claim.evidence_ids)
        else:
            text = claim.get("text", "")
            doc = claim.get("source_doc_id", "")
            eids = ", ".join(claim.get("evidence_ids", []))
        lines.append(f"  [{i}] (source={doc} | evidence_ids={eids}) {text}")

    if coverage_gaps:
        lines.append(f"\nKnown coverage gaps (partial/not_covered): {coverage_gaps}")

    return "\n".join(lines)


def _build_citations(claims: list, all_chunks: dict[str, dict]) -> list[Citation]:
    """Convert verified claims' evidence_ids to Citation objects."""
    seen: set[str] = set()
    citations: list[Citation] = []

    for claim in claims:
        evidence_ids = claim.evidence_ids if isinstance(claim, Claim) else claim.get("evidence_ids", [])

        for evidence_id in evidence_ids:
            if evidence_id in seen:
                continue
            seen.add(evidence_id)

            chunk = all_chunks.get(evidence_id)
            if chunk is None:
                continue

            passage = chunk.get("text") or chunk.get("passage") or ""
            citations.append(
                Citation(
                    doc_id=chunk["doc_id"],
                    passage=passage[:500],
                    page=chunk["page_number"],
                    publication_date=chunk.get("publication_date"),
                )
            )

    return citations


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output."""
    return round(
        (prompt_tokens / 1_000_000) * 0.15 + (completion_tokens / 1_000_000) * 0.60,
        6,
    )
