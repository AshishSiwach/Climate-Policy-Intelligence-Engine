"""
Verifier node — Phase 2 Cross-Document Route.

Runs the hardened citation verifier on every claim's evidence_ids.
No LLM calls — purely deterministic.
No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.

This is the only place where model-hallucinated evidence gets caught before synthesis.
It must not be skipped.
"""

from __future__ import annotations

import logging

from src.evidence.claims import Claim

logger = logging.getLogger(__name__)


def run_verifier(state: dict) -> dict:
    """Verifier node: drop any claim whose evidence can't be verified.

    Input: state["claims"], state["retrievals"]
    Output: {"verified_claims": [Claim, ...], "steps_used": state["steps_used"] + 1}

    For each claim:
    - Collect the chunks referenced by claim.evidence_ids from state["retrievals"]
    - Build an LLMCitation with chunk_id=evidence_id and passage=the chunk's text
    - Call verify_citations(citations, all_chunks) from src/evidence/citations.py
    - If at least one citation survives: keep the claim with only the verified evidence_ids
    - If zero citations survive: drop the claim entirely

    No LLM call — purely deterministic verification.
    """
    claims: list = state.get("claims", [])
    retrievals: dict = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)

    # Flatten all chunks from all sub-questions into a chunk_id → chunk dict
    all_chunks: dict[str, dict] = {}
    for chunk_list in retrievals.values():
        for chunk in chunk_list:
            cid = chunk.get("chunk_id")
            if cid:
                all_chunks[cid] = chunk

    flat_chunks = list(all_chunks.values())

    verified: list[Claim] = []

    for raw_claim in claims:
        claim = _coerce_claim(raw_claim)
        if claim is None:
            continue

        surviving_ids = _verify_claim_evidence(claim, all_chunks, flat_chunks)

        if not surviving_ids:
            logger.info("Verifier: dropped claim %r — zero surviving evidence_ids", claim.id)
            continue

        # Correct source_doc_id to match the first verified chunk's actual doc_id
        first_chunk = all_chunks.get(surviving_ids[0])
        corrected_doc_id = first_chunk["doc_id"] if first_chunk else claim.source_doc_id

        verified.append(
            Claim(
                id=claim.id,
                text=claim.text,
                evidence_ids=surviving_ids,
                source_doc_id=corrected_doc_id,
            )
        )

    return {"verified_claims": verified, "steps_used": steps_used + 1}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _coerce_claim(raw) -> Claim | None:
    """Accept Claim objects or dicts; return None on failure."""
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


def _verify_claim_evidence(
    claim: Claim,
    all_chunks: dict[str, dict],
    flat_chunks: list[dict],
) -> list[str]:
    """Return the subset of evidence_ids that survive hardened verification."""
    from src.evidence.citations import verify_citations
    from src.synthesis.output_schema import LLMCitation

    surviving: list[str] = []

    for evidence_id in claim.evidence_ids:
        chunk = all_chunks.get(evidence_id)
        if chunk is None:
            logger.info(
                "Verifier: evidence_id %r not found in retrievals for claim %r",
                evidence_id,
                claim.id,
            )
            continue

        passage = (chunk.get("text") or chunk.get("passage") or "").strip()
        if not passage:
            logger.info(
                "Verifier: chunk %r has empty text — dropping evidence_id for claim %r",
                evidence_id,
                claim.id,
            )
            continue

        cit = LLMCitation(
            doc_id=chunk.get("doc_id", ""),
            passage=passage,
            page=chunk.get("page_number", 1),
            chunk_id=evidence_id,
        )

        # Fail-closed: verify_citations drops the citation on any rule failure
        result = verify_citations([cit], flat_chunks)
        if result:
            surviving.append(evidence_id)
        else:
            logger.info(
                "Verifier: evidence_id %r failed verification for claim %r",
                evidence_id,
                claim.id,
            )

    return surviving
