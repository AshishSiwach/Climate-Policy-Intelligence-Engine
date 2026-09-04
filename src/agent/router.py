"""
Complexity router — Phase 1 Foundations.

Classifies a query into one of six task types using GPT-4o-mini and routes it
to either the fast path or the agent path.

Routing rule:
  cross_doc | summary | contradiction  →  Path.AGENT
  factual | numeric | unsupported      →  Path.FAST

Fail-open: any error returns (Path.FAST, "factual").
"""

from __future__ import annotations

import json
import logging
import os
from enum import Enum

logger = logging.getLogger(__name__)

_VALID_TYPES = {"factual", "numeric", "cross_doc", "summary", "contradiction", "unsupported"}
_AGENT_TYPES = {"cross_doc", "summary", "contradiction"}

_SYSTEM_PROMPT = """\
You are classifying a climate policy question by query type. Respond with a JSON
object containing exactly one key "task_type" with one of these values:

- factual: single-source lookup of a specific fact or definition
- numeric: asks for a specific number, year, percentage, or quantity
- cross_doc: requires comparing or synthesising across multiple documents or institutions
- summary: asks to summarise a long report, section, or document
- contradiction: asks whether sources agree or disagree on a specific point
- unsupported: out of domain or unanswerable from a climate policy corpus

Return only: {"task_type": "<value>"}
"""


class Path(str, Enum):
    FAST = "fast"
    AGENT = "agent"


def complexity_router(query: str, api_key: str | None = None) -> tuple[Path, str]:
    """Classify query and return (Path, task_type_str).

    On any error returns (Path.FAST, "factual") — fail-open so the pipeline
    always has a route even when the router LLM is unavailable.

    Args:
        query: The user's question.
        api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.

    Returns:
        Tuple of (Path enum, task_type string).
    """
    try:
        return _call_router(query, api_key)
    except Exception as exc:
        logger.warning("complexity_router failed (fail-open to FAST): %s", exc)
        return Path.FAST, "factual"


def _call_router(query: str, api_key: str | None) -> tuple[Path, str]:
    """Inner router — raises on any error so the outer wrapper can catch it."""
    from openai import OpenAI  # deferred import — only load when needed

    key = api_key or os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        response_format={"type": "json_object"},
        max_tokens=64,
        temperature=0.0,
    )

    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    task_type = parsed.get("task_type", "factual")

    if task_type not in _VALID_TYPES:
        logger.warning("Router returned unknown task_type %r — defaulting to factual", task_type)
        task_type = "factual"

    path = Path.AGENT if task_type in _AGENT_TYPES else Path.FAST
    return path, task_type
