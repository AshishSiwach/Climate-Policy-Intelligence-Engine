"""
Planner node — Phase 1 Foundations.

Decomposes a complex query into a list of SubQuestion objects using GPT-4o-mini.
No langgraph imports. Takes a plain dict (AgentState) and returns a partial dict.

Retry logic:
  - On malformed JSON, retry once with a stricter prompt.
  - If still bad after retry, return termination_reason="fallback_to_fast".
"""

from __future__ import annotations

import json
import logging
import os

from src.evidence.claims import SubQuestion

logger = logging.getLogger(__name__)

_MAX_SUB_QUESTIONS = 6

# --- cross_doc prompts (Phase 3) ---

_CROSSDOC_SYSTEM_PROMPT = """\
You are a research planner for a climate policy analysis engine.

Given a complex query, decompose it into at most {max_sq} focused factual sub-questions
that together provide the evidence needed to answer the query. Each sub-question must be
directly answerable from a retrieved passage — do NOT generate synthesis or comparison
sub-questions ("How do X and Y compare?"). The synthesis step handles comparisons
automatically once factual evidence is retrieved.

Return a JSON array where each element has these exact keys:
  - "id": string like "sq_0", "sq_1", ... (sequential)
  - "question": string — the factual sub-question text
  - "required_source": string or null — institution name if a specific source is needed (e.g. "BoE", "Ofgem", "IPCC")
  - "task_type": "factual"

Return ONLY the JSON array, no other text.
"""

_CROSSDOC_STRICT_SYSTEM_PROMPT = """\
You are a research planner. Return a valid JSON array and NOTHING ELSE.

Decompose the query into at most {max_sq} factual sub-questions (no comparison/synthesis questions). Each must have:
  "id" (sq_0, sq_1...), "question" (string), "required_source" (string or null),
  "task_type": "factual".

Example: [{"id":"sq_0","question":"What is X?","required_source":null,"task_type":"factual"}]

Return ONLY the JSON array.
"""

# --- summary prompts (Phase 5a) ---

_SUMMARY_SYSTEM_PROMPT = """\
You are a research planner for a climate policy analysis engine.

Given a request to summarise a document or report, decompose it into at most {max_sq}
focused factual sub-questions that together cover the report's main sections. Each
sub-question must be directly answerable from a retrieved passage.

Generate sub-questions covering these standard report sections (adapt to the query's subject):
  1. Overall objectives and scope — what does this report set out to do?
  2. Key findings and headline metrics — what are the main numbers or conclusions?
  3. Sectoral or thematic analysis — what does it say about specific sectors, technologies, or regions?
  4. Policy recommendations — what actions or policies does it propose?
  5. Identified risks, barriers, or evidence gaps

If the query names a specific institution, set "required_source" to that institution for
all sub-questions so retrieval is targeted to that document set.

Return a JSON array where each element has these exact keys:
  - "id": string like "sq_0", "sq_1", ... (sequential)
  - "question": string — the factual sub-question text
  - "required_source": string or null — institution name if a specific source is needed
  - "task_type": "factual"

Return ONLY the JSON array, no other text.
"""

_SUMMARY_STRICT_SYSTEM_PROMPT = """\
You are a research planner. Return a valid JSON array and NOTHING ELSE.

Decompose the summary request into at most {max_sq} section-covering factual sub-questions.
Cover: objectives, key findings, sectoral analysis, policy recommendations, evidence gaps.
Each must have: "id" (sq_0, sq_1...), "question" (string), "required_source" (string or null),
"task_type": "factual".

Example: [{"id":"sq_0","question":"What are the main objectives?","required_source":"IEA","task_type":"factual"}]

Return ONLY the JSON array.
"""

# Backward-compatible aliases kept for any external tests that reference them
_SYSTEM_PROMPT = _CROSSDOC_SYSTEM_PROMPT
_STRICT_SYSTEM_PROMPT = _CROSSDOC_STRICT_SYSTEM_PROMPT


def run_planner(state: dict) -> dict:
    """Planner node: decomposes state["query"] into SubQuestion objects.

    Selects prompt based on state["task_type"]: summary queries get section-covering
    sub-questions; cross_doc and all others get factual evidence sub-questions.

    Returns partial dict with "sub_questions" and incremented "steps_used".
    On repeated failure returns {"termination_reason": "fallback_to_fast"}.
    """
    query = state.get("query", "")
    task_type = state.get("task_type", "cross_doc")
    steps_used = state.get("steps_used", 0)

    if task_type == "summary":
        normal_prompt = _SUMMARY_SYSTEM_PROMPT
        strict_prompt = _SUMMARY_STRICT_SYSTEM_PROMPT
    else:
        normal_prompt = _CROSSDOC_SYSTEM_PROMPT
        strict_prompt = _CROSSDOC_STRICT_SYSTEM_PROMPT

    logger.info("Planner: task_type=%s", task_type)

    result = _call_planner(query, system_prompt=normal_prompt, use_json_mode=True)
    if result is None:
        logger.warning("Planner: first attempt failed — retrying with strict prompt")
        result = _call_planner(query, system_prompt=strict_prompt, use_json_mode=False)

    if result is None:
        logger.error("Planner: both attempts failed — falling back to fast path")
        return {"termination_reason": "fallback_to_fast", "steps_used": steps_used + 1}

    return {"sub_questions": result, "steps_used": steps_used + 1}


def _call_planner(query: str, system_prompt: str, use_json_mode: bool) -> list[SubQuestion] | None:
    """Call GPT-4o-mini and parse the response. Returns None on failure."""
    try:
        from openai import OpenAI  # deferred import

        key = os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=key)

        system = system_prompt.format(max_sq=_MAX_SUB_QUESTIONS)

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"Query: {query}"},
            ],
            response_format={"type": "json_object"} if use_json_mode else None,
            max_tokens=512,
            temperature=0.0,
        )

        raw = response.choices[0].message.content or "[]"
        return _parse_sub_questions(raw)

    except Exception as exc:
        logger.warning("Planner LLM call failed: %s", exc)
        return None


def _parse_sub_questions(raw: str) -> list[SubQuestion] | None:
    """Parse JSON string into a list of SubQuestion. Returns None on parse failure."""
    try:
        # Handle both {"sub_questions": [...]} and bare [...]
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            # Extract the list from the first list-valued key
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                parsed = []

        if not isinstance(parsed, list):
            return None

        result: list[SubQuestion] = []
        for i, item in enumerate(parsed[:_MAX_SUB_QUESTIONS]):
            if not isinstance(item, dict):
                continue
            # Normalise id in case model omitted it
            item.setdefault("id", f"sq_{i}")
            result.append(SubQuestion(**item))

        return result if result else None

    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.warning("Planner parse failed: %s", exc)
        return None
