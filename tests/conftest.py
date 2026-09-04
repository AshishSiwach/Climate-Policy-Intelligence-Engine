"""
Shared pytest fixtures + path setup.

The editable install (`_editable_impl_cpie.pth`) adds the MAIN repo's `src/`
to sys.path.  When running in a worktree (e.g. phase-0-hardening), we must
also insert the WORKTREE's `src/` at position 0 so new modules created in the
worktree (``evidence``, ``config``) shadow the main repo's copies and are
found first.  main.py sits at the repo root, so we add that too.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# Worktree's src/ must come first so new packages (evidence, config) are found
# before the main-repo editable install on sys.path.
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    page: int = 1,
    institution: str = "Ofgem",
    publication_date: str = "2024",
    chunk_type: str = "prose",
    rrf_score: float = 0.03,
) -> dict:
    """Minimal chunk dict matching what our real pipeline produces."""
    return {
        "text": text,
        "doc_id": doc_id,
        "chunk_id": f"{doc_id}_{chunk_index}",  # matches dlt_pipeline format
        "institution": institution,
        "doc_type": "consultation",
        "jurisdiction": "UK",
        "publication_date": publication_date,
        "page_number": page,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
        "token_count": len(text.split()),
        "rrf_score": rrf_score,
    }


@pytest.fixture
def sample_chunks() -> list[dict]:
    """Three synthetic chunks from two documents — enough for retrieval + synthesis tests."""
    return [
        _make_chunk(
            "OFGEM_TEST",
            0,
            "Ofgem proposes new load control licensing requirements for 2026 applications.",
            page=1,
            rrf_score=0.032,
        ),
        _make_chunk(
            "OFGEM_TEST",
            1,
            "The consultation closes in February 2026 and responses will be published on the Ofgem website.",
            page=2,
            rrf_score=0.028,
        ),
        _make_chunk(
            "BOE_TEST",
            0,
            "UK banks faced aggregate losses of £334 billion under the CBES early action scenario.",
            page=53,
            institution="BoE",
            publication_date="2021",
            rrf_score=0.015,
        ),
    ]


@pytest.fixture
def tmp_log_path(tmp_path) -> Path:
    """Isolated log file path for guardrail tests that inspect the log."""
    return tmp_path / "test_queries.jsonl"
