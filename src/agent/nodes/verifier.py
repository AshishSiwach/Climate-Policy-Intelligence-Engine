"""
Verifier node — Phase 2 Cross-Document Route.

Filters claims to those whose evidence_ids refer to chunk_ids that actually
exist in state["retrievals"].  No LLM call, no semantic matching — just a
set-membership check to catch any chunk_ids the claim_builder LLM hallucinated.

No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.
"""

from __future__ import annotations

import logging

from src.evidence.claims import Claim

logger = logging.getLogger(__name__)


def run_verifier(state: dict) -> dict:
    """Verifier node: drop claims whose evidence_ids are not in retrievals.

    For each claim produced by claim_builder:
      - Keep evidence_ids that exist as chunk_id in state["retrievals"]
      - Drop claims where zero evidence_ids survive (fully hallucinated)

    Input:  state["claims"], state["retrievals"]
    Output: {"verified_claims": [Claim, ...], "steps_used": state["steps_used"] + 1}
    """
    claims: list = state.get("claims", [])
    retrievals: dict = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)

    # Build per-sub-question chunk sets for scoped evidence checking
    sq_chunk_sets: dict[str, set[str]] = {
        sq_id: {chunk.get("chunk_id") for chunk in chunks if chunk.get("chunk_id")}
        for sq_id, chunks in retrievals.items()
    }

    verified: list[Claim] = []

    for raw_claim in claims:
        claim = _coerce_claim(raw_claim)
        if claim is None:
            continue

        # Strict scoped check: claims with no sub_question_id or an unrecognised one
        # are dropped — they can't be verified against a specific retrieval bucket.
        sq_id = claim.sub_question_id
        if not sq_id:
            logger.info("Verifier: dropped claim %r — no sub_question_id assigned", claim.id)
            continue
        allowed = sq_chunk_sets.get(sq_id, set())

        valid_eids = [e for e in claim.evidence_ids if e in allowed]

        if not valid_eids:
            logger.info(
                "Verifier: dropped claim %r (sq=%s) — none of %s found in retrievals",
                claim.id,
                sq_id,
                claim.evidence_ids,
            )
            continue

        verified.append(Claim(
            id=claim.id,
            text=claim.text,
            sub_question_id=claim.sub_question_id,
            evidence_ids=valid_eids,
            source_doc_id=claim.source_doc_id,
        ))

    logger.info("Verifier: %d/%d claims passed", len(verified), len(claims))
    return {"verified_claims": verified, "steps_used": steps_used + 1}


def _coerce_claim(raw) -> Claim | None:
    if isinstance(raw, Claim):
        return raw
    if isinstance(raw, dict):
        try:
            return Claim(**raw)
        except Exception as exc:
            logger.warning("Verifier: could not coerce claim dict: %s", exc)
            return None
    logger.warning("Verifier: unexpected claim type %s", type(raw))
    return None
