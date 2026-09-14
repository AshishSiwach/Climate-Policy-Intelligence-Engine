"""
Document resolver node — Phase 5a Summary Route.

For summary queries only. Resolves the user's target document to an exact
corpus doc_id BEFORE the planner runs, so retrieval is scoped to the right
document and the pipeline can give an explicit corpus-gap error rather than
silently summarising the wrong report.

Resolution strategy
-------------------
1. LLM extracts {institution, report_keyword, year} from the query.
2. Filter _corpus_doc_ids by institution prefix (case-insensitive).
3. Score remaining candidates by how well they match the keyword + year.
4. If exactly one candidate scores above threshold → resolved_doc_id.
5. If multiple candidates tie → LLM disambiguation.
6. If zero candidates → termination_reason = "corpus_gap".

When _corpus_doc_ids is not injected in state, falls back to a retrieval
probe: retrieve top-k chunks from the target institution and inspect their
doc_ids. This covers callers that don't pre-load the doc_id list.

Non-summary queries: no-op — returns {} immediately.
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM_PROMPT = """\
Extract the target document the user wants summarised. Return a JSON object with:
  "institution": string — the organisation name (e.g. "IEA", "BoE", "Ofgem", "CCC", "ESO", "DESNZ")
  "report_keyword": string — the key title words (e.g. "World Energy Outlook", "Financial Stability Report")
  "year": string or null — the publication year if mentioned (e.g. "2023", "2024") otherwise null

Return ONLY: {"institution": "...", "report_keyword": "...", "year": ...}
"""


def run_document_resolver(state: dict) -> dict:
    """Document resolver node: resolve target doc_id for summary queries.

    Returns {"resolved_doc_id": str} on success.
    Returns {"termination_reason": "corpus_gap"} if the requested document is
    not in the corpus.
    Returns {} (no-op) for non-summary task types.
    """
    task_type = state.get("task_type", "")
    if task_type != "summary":
        return {}

    query = state.get("query", "")
    corpus_doc_ids: list[str] = state.get("_corpus_doc_ids") or []

    # Step 1: extract target document metadata from query
    extracted = _extract_target_document(query)
    if extracted is None:
        logger.warning("document_resolver: LLM extraction failed — skipping resolution")
        return {}  # degrade gracefully; planner uses required_source instead

    institution = extracted.get("institution", "")
    keyword = extracted.get("report_keyword", "")
    year = extracted.get("year")
    logger.info(
        "document_resolver: target institution=%r keyword=%r year=%r",
        institution,
        keyword,
        year,
    )

    # Step 2: get candidate doc_ids — from injected list or retrieval probe
    if corpus_doc_ids:
        candidates = corpus_doc_ids
    else:
        candidates = _probe_via_retriever(state, institution)

    if not candidates:
        logger.info("document_resolver: no corpus doc_ids available — skipping resolution")
        return {}

    # Step 3: score and select
    resolved = _select_best_match(candidates, institution, keyword, year)

    if resolved is None:
        logger.info(
            "document_resolver: no corpus match for institution=%r keyword=%r year=%r",
            institution,
            keyword,
            year,
        )
        return {"termination_reason": "corpus_gap"}

    logger.info("document_resolver: resolved to doc_id=%r", resolved)
    return {"resolved_doc_id": resolved}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_target_document(query: str) -> dict | None:
    """Call LLM to extract {institution, report_keyword, year} from the query."""
    import json

    try:
        from openai import OpenAI

        key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _EXTRACT_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ],
            response_format={"type": "json_object"},
            max_tokens=128,
            temperature=0.0,
        )

        raw = response.choices[0].message.content or "{}"
        return json.loads(raw)

    except Exception as exc:
        logger.warning("document_resolver: LLM extraction failed: %s", exc)
        return None


def _probe_via_retriever(state: dict, institution: str) -> list[str]:
    """Retrieve a small sample from the institution to discover its doc_ids."""
    retriever = state.get("_retriever")
    query = state.get("query", "")

    if retriever is None:
        return []

    try:
        kwargs: dict = {"top_k": 10}
        if institution:
            kwargs["institutions"] = [institution]
        chunks = retriever.retrieve(query, **kwargs)
        doc_ids = list({c["doc_id"] for c in chunks if c.get("doc_id")})
        logger.debug("document_resolver: probe found doc_ids=%s", doc_ids)
        return doc_ids
    except Exception as exc:
        logger.warning("document_resolver: retrieval probe failed: %s", exc)
        return []


def _select_best_match(
    candidates: list[str],
    institution: str,
    keyword: str,
    year: str | None,
) -> str | None:
    """Score candidates against institution + keyword + year. Return best match or None."""
    # Normalise to lowercase tokens for matching
    inst_tok = institution.lower().strip()
    kw_toks = set(re.split(r"[\s_\-]+", keyword.lower().strip())) - {"", "the", "a", "an", "of"}
    year_tok = (year or "").strip()

    scored: list[tuple[float, str]] = []

    for doc_id in candidates:
        doc_lower = doc_id.lower()
        score = 0.0

        # Institution prefix match
        if inst_tok and doc_lower.startswith(inst_tok):
            score += 3.0
        elif inst_tok and inst_tok in doc_lower:
            score += 1.5

        # Keyword token matches
        for tok in kw_toks:
            if tok in doc_lower:
                score += 1.0

        # Year match
        if year_tok and year_tok in doc_id:
            score += 2.0

        if score > 0:
            scored.append((score, doc_id))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)

    # Require minimum score: institution match + at least one keyword token
    best_score, best_id = scored[0]
    if best_score < 2.0:
        return None

    # If top two candidates are tied, prefer the one with the closer year
    if len(scored) >= 2 and abs(scored[0][0] - scored[1][0]) < 0.5 and year_tok:
        # pick the one with the matching year
        for score, doc_id in scored:
            if year_tok in doc_id:
                return doc_id

    return best_id
