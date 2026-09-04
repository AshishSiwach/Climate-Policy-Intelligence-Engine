"""
Grader node — Phase 1 Foundations.

Makes a single LLM call (GPT-4o-mini) to evaluate coverage across ALL
sub-questions and their retrieved passages in one batch.

No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.

Fail-permissive: if the grader LLM returns unparseable output, all
sub-questions are treated as "covered" so the workflow can continue.
"""

from __future__ import annotations

import json
import logging
import os

from src.evidence.claims import Coverage, SubQuestion

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a coverage grader for a climate policy retrieval system.

Given a list of sub-questions and retrieved passages for each, classify
the coverage of each sub-question.

Coverage levels:
  - "covered": the passages clearly answer the sub-question
  - "partial": the passages partially address it but have gaps
  - "not_covered": the passages do not address the sub-question

Return a JSON object keyed by sub-question id. For each:
  - "status": one of "covered", "partial", "not_covered"
  - "gap_reason": string explaining the gap (required for "partial" and "not_covered"; null for "covered")

Example:
{
  "sq_0": {"status": "covered", "gap_reason": null},
  "sq_1": {"status": "partial", "gap_reason": "No data on post-2020 emissions targets"}
}

Return ONLY the JSON object.
"""


def run_grader(state: dict) -> dict:
    """Grader node: assess coverage for all sub-questions given retrieved chunks.

    Returns partial dict with "coverage" and incremented "steps_used".
    On LLM/parse failure, defaults all sub-questions to "covered" (fail permissive).
    """
    sub_questions: list = state.get("sub_questions", [])
    retrievals: dict = state.get("retrievals", {})
    steps_used = state.get("steps_used", 0)

    if not sub_questions:
        return {"coverage": {}, "steps_used": steps_used + 1}

    # Build coverage — attempt LLM grading, fall back on any failure
    coverage = _call_grader(sub_questions, retrievals)

    if coverage is None:
        logger.warning("Grader: LLM failed or returned unparseable output — defaulting all to 'covered'")
        coverage = _default_coverage(sub_questions)

    return {"coverage": coverage, "steps_used": steps_used + 1}


def _call_grader(sub_questions: list, retrievals: dict) -> dict[str, Coverage] | None:
    """Call LLM and parse coverage dict. Returns None on any failure."""
    try:
        from openai import OpenAI  # deferred import

        key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key)

        user_content = _build_grader_prompt(sub_questions, retrievals)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            max_tokens=512,
            temperature=0.0,
        )

        raw = response.choices[0].message.content or "{}"
        return _parse_coverage(raw, sub_questions)

    except Exception as exc:
        logger.warning("Grader LLM call failed: %s", exc)
        return None


def _build_grader_prompt(sub_questions: list, retrievals: dict) -> str:
    """Serialise sub-questions + retrieved passages for the grader prompt."""
    lines: list[str] = []
    for sq in sub_questions:
        if isinstance(sq, SubQuestion):
            sq_id, question = sq.id, sq.question
        else:
            sq_id = sq.get("id", "sq_?")
            question = sq.get("question", "")

        chunks = retrievals.get(sq_id, [])
        passages = " | ".join((c.get("text") or c.get("passage") or "")[:300] for c in chunks[:5])
        lines.append(f"{sq_id}: Q={question!r}  PASSAGES={passages!r}")

    return "\n".join(lines)


def _parse_coverage(raw: str, sub_questions: list) -> dict[str, Coverage] | None:
    """Parse LLM JSON into Coverage objects. Returns None on failure."""
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            return None

        coverage: dict[str, Coverage] = {}
        for sq in sub_questions:
            sq_id = sq.id if isinstance(sq, SubQuestion) else sq.get("id", "sq_?")
            entry = parsed.get(sq_id, {})
            status = entry.get("status", "covered")
            if status not in {"covered", "partial", "not_covered"}:
                status = "covered"
            coverage[sq_id] = Coverage(
                sub_question_id=sq_id,
                status=status,
                gap_reason=entry.get("gap_reason"),
            )
        return coverage

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Grader parse failed: %s", exc)
        return None


def _default_coverage(sub_questions: list) -> dict[str, Coverage]:
    """Return all-covered fallback for each sub-question."""
    coverage: dict[str, Coverage] = {}
    for sq in sub_questions:
        sq_id = sq.id if isinstance(sq, SubQuestion) else sq.get("id", "sq_?")
        coverage[sq_id] = Coverage(sub_question_id=sq_id, status="covered")
    return coverage
