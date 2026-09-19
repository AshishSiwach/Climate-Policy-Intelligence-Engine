"""
Claim Builder node — Phase 2 Cross-Document Route.

Makes a single batched LLM call (GPT-4o-mini) to extract structured claims
from retrieved passages for covered/partial sub-questions.

No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.

Rules:
- Only process sub-questions where coverage.status != "not_covered".
- One LLM call total (batched over all covered sub-questions).
- Each claim must have: id, text, evidence_ids, source_doc_id.
- No claim without at least one evidence_id.
- On malformed LLM output: return empty claims list (don't crash).
- Max claims: 12 total across all sub-questions.
"""

from __future__ import annotations

import json
import logging
import os

from src.evidence.claims import Claim, SubQuestion

logger = logging.getLogger(__name__)

_MAX_CLAIMS = 12

_SYSTEM_PROMPT = """\
You are a claim extractor for a climate policy analysis system.

Given sub-questions and retrieved passages for each, extract factual claims
that are directly supported by the passages.

Rules:
- Only extract claims for sub-questions shown below.
- Each claim must be supported by at least one retrieved passage chunk.
- For each claim, list the chunk_ids (shown in [chunk_id=...] headers) that support it.
- Do NOT fabricate chunk_ids. Only use chunk_ids visible in the passages below.
- Assign each claim a unique id: "c_0", "c_1", etc. (sequential, no duplicates).
- Maximum 12 claims total.
- Skip claims with no supporting chunk_id.

Return a JSON object with this exact structure:
{"claims": [{"id": "c_0", "text": "...", "evidence_ids": ["chunk_id_1"], "source_doc_id": "doc_id"}]}

Return ONLY the JSON object.
"""


def run_claim_builder(state: dict) -> dict:
    """Claim builder node: extract claims from covered/partial sub-questions.

    Input: state["sub_questions"], state["retrievals"], state["coverage"]
    Output: {"claims": [Claim, ...], "steps_used": state["steps_used"] + 1,
             "cost_used_usd": updated}

    On LLM/parse failure, returns empty claims list (never crashes).
    """
    sub_questions: list = state.get("sub_questions", [])
    retrievals: dict = state.get("retrievals", {})
    coverage: dict = state.get("coverage", {})
    steps_used = state.get("steps_used", 0)
    cost_used_usd = state.get("cost_used_usd", 0.0)

    # Ablation bypass: skip LLM call and return empty claims (eval flag only)
    if state.get("_ablate_no_claim_builder"):
        logger.info("ClaimBuilder: ablation bypass — returning empty claims")
        return {"claims": [], "steps_used": steps_used + 1, "cost_used_usd": cost_used_usd}

    # Filter to covered/partial only — skip not_covered
    eligible_sqs = [
        sq for sq in sub_questions if _get_sq_id(sq) in coverage and coverage[_get_sq_id(sq)].status != "not_covered"
    ]

    if not eligible_sqs:
        logger.info("ClaimBuilder: no eligible sub-questions (all not_covered or empty)")
        return {"claims": [], "steps_used": steps_used + 1, "cost_used_usd": cost_used_usd}

    claims, call_cost = _call_claim_builder(eligible_sqs, retrievals, coverage)
    claims = _assign_sub_question_ids(claims, retrievals)

    return {
        "claims": claims,
        "steps_used": steps_used + 1,
        "cost_used_usd": cost_used_usd + call_cost,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_sq_id(sq) -> str:
    return sq.id if isinstance(sq, SubQuestion) else sq.get("id", "sq_?")


def _call_claim_builder(
    sub_questions: list,
    retrievals: dict,
    coverage: dict,
) -> tuple[list[Claim], float]:
    """Call LLM and parse claims. Returns (claims, cost_usd)."""
    try:
        from openai import OpenAI  # deferred import

        key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key)

        user_content = _build_prompt(sub_questions, retrievals, coverage)

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

        claims = _parse_claims(raw)
        return claims, cost

    except Exception as exc:
        logger.warning("ClaimBuilder LLM call failed: %s", exc)
        return [], 0.0


def _build_prompt(sub_questions: list, retrievals: dict, coverage: dict) -> str:
    """Serialise eligible sub-questions + retrieved passages for the claim extractor."""
    lines: list[str] = []
    for sq in sub_questions:
        sq_id = _get_sq_id(sq)
        question = sq.question if isinstance(sq, SubQuestion) else sq.get("question", "")
        cov = coverage.get(sq_id)
        status = cov.status if cov else "covered"

        lines.append(f"\n### Sub-question {sq_id} (coverage={status}): {question}")
        chunks = retrievals.get(sq_id, [])
        for chunk in chunks[:4]:
            chunk_id = chunk.get("chunk_id", "")
            doc_id = chunk.get("doc_id", "")
            text = (chunk.get("text") or chunk.get("passage") or "")[:500]
            lines.append(f"[chunk_id={chunk_id}] [doc_id={doc_id}]\n{text}")

    return "\n".join(lines)


def _parse_claims(raw: str) -> list[Claim]:
    """Parse LLM JSON into Claim objects. Returns empty list on any failure."""
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return []

        raw_claims = parsed.get("claims", [])
        if not isinstance(raw_claims, list):
            return []

        claims: list[Claim] = []
        seen_ids: set[str] = set()

        for i, item in enumerate(raw_claims[:_MAX_CLAIMS]):
            if not isinstance(item, dict):
                continue

            claim_id = str(item.get("id", f"c_{i}"))
            text = str(item.get("text", "")).strip()
            evidence_ids = item.get("evidence_ids", [])
            source_doc_id = str(item.get("source_doc_id", ""))

            # Drop invalid claims
            if not text:
                logger.info("ClaimBuilder: dropping claim %r — empty text", claim_id)
                continue
            if not isinstance(evidence_ids, list) or not evidence_ids:
                logger.info("ClaimBuilder: dropping claim %r — no evidence_ids", claim_id)
                continue
            if claim_id in seen_ids:
                logger.info("ClaimBuilder: dropping duplicate claim id %r", claim_id)
                continue

            seen_ids.add(claim_id)
            claims.append(
                Claim(
                    id=claim_id,
                    text=text,
                    evidence_ids=[str(e) for e in evidence_ids if e],
                    source_doc_id=source_doc_id,
                )
            )

        return claims

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("ClaimBuilder parse failed: %s", exc)
        return []


def _assign_sub_question_ids(claims: list[Claim], retrievals: dict) -> list[Claim]:
    """Annotate each claim with the sub_question_id that owns most of its evidence.

    Builds a reverse map chunk_id → sq_id from retrievals, then for each claim
    takes a plurality vote over its evidence_ids.  Claims whose evidence_ids are
    all hallucinated (no matching chunk anywhere) keep sub_question_id=None.
    """
    chunk_to_sq: dict[str, str] = {}
    for sq_id, chunks in retrievals.items():
        for chunk in chunks:
            cid = chunk.get("chunk_id")
            if cid:
                chunk_to_sq[cid] = sq_id

    annotated: list[Claim] = []
    for claim in claims:
        votes: dict[str, int] = {}
        for eid in claim.evidence_ids:
            sq = chunk_to_sq.get(eid)
            if sq:
                votes[sq] = votes.get(sq, 0) + 1

        winner = max(votes, key=votes.__getitem__) if votes else None
        annotated.append(Claim(
            id=claim.id,
            text=claim.text,
            sub_question_id=winner,
            evidence_ids=claim.evidence_ids,
            source_doc_id=claim.source_doc_id,
        ))

    return annotated


def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
    """GPT-4o-mini pricing: $0.15/1M input, $0.60/1M output."""
    return round(
        (prompt_tokens / 1_000_000) * 0.15 + (completion_tokens / 1_000_000) * 0.60,
        6,
    )
