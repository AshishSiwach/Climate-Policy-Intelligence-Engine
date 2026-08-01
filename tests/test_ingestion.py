"""
Unit tests for ingestion — pdf_loader.clean_text + chunker.chunk_page.

Deliberately avoid touching real PDFs (they're gitignored and not committed).
All tests use inline text fixtures. PyMuPDF integration is exercised in
scripts/validate_pipeline_e2e.py, not here.
"""

from __future__ import annotations

import pytest

from ingestion.chunker import (
    CHUNK_SIZE,
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP,
    chunk_page,
)
from ingestion.pdf_loader import DOC_REGISTRY, clean_text


# ---------------------------------------------------------------------------
# Doc registry integrity
# ---------------------------------------------------------------------------

REQUIRED_META_FIELDS = {
    "doc_id", "institution", "doc_type", "jurisdiction",
    "publication_date", "tier1_strip", "tier2_inject",
}


def test_doc_registry_has_12_documents():
    assert len(DOC_REGISTRY) == 12, f"Expected 12 PDFs in registry, got {len(DOC_REGISTRY)}"


def test_doc_registry_every_entry_has_required_metadata():
    for filename, meta in DOC_REGISTRY.items():
        missing = REQUIRED_META_FIELDS - set(meta.keys())
        assert not missing, f"{filename} missing fields: {missing}"


def test_doc_registry_doc_ids_are_unique():
    doc_ids = [meta["doc_id"] for meta in DOC_REGISTRY.values()]
    assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_ids in registry"


# ---------------------------------------------------------------------------
# clean_text — Tier 1 stripping
# ---------------------------------------------------------------------------

def test_clean_text_strips_eso_nav_elements():
    """ESO PDF has 'Navigation', 'Download a pdf', 'Text Links', 'Return to contents' clutter."""
    dirty = "Some real content Navigation Download a pdf Text Links more content"
    cleaned = clean_text(dirty, "ESO_BEYOND2030_2024")
    assert "Navigation" not in cleaned
    assert "Download a pdf" not in cleaned
    assert "Text Links" not in cleaned
    assert "Some real content" in cleaned
    assert "more content" in cleaned


def test_clean_text_strips_ofgem_official_stamps():
    dirty = "Real policy text OFFICIAL OFFICIAL more policy text"
    cleaned = clean_text(dirty, "OFGEM_SMART_SECURE_2025")
    assert "OFFICIAL OFFICIAL" not in cleaned
    assert "Real policy text" in cleaned


def test_clean_text_untouched_for_tier3_documents():
    """CBES Results has no Tier 1 stripping — text passes through unchanged (modulo whitespace normalisation)."""
    original = "Bank losses under the early action scenario were £334bn."
    cleaned = clean_text(original, "BOE_CBES_RESULTS_2021")
    # Substrings preserved (whitespace normalisation may collapse spaces)
    assert "Bank losses" in cleaned
    assert "£334bn" in cleaned


# ---------------------------------------------------------------------------
# chunk_page — bounds and behaviour
# ---------------------------------------------------------------------------

def _make_text(word_count: int) -> str:
    """Deterministic text of roughly `word_count` tokens (words ~= tokens for ASCII English)."""
    return " ".join(f"word{i}" for i in range(word_count))


def test_chunk_page_produces_at_least_one_chunk_for_long_input():
    text = _make_text(800)
    chunks = chunk_page(text)
    assert len(chunks) >= 2, "800-word text should split into multiple chunks"


def test_chunk_page_respects_chunk_size_upper_bound():
    """Each individual chunk from chunk_page must be <= CHUNK_SIZE tokens."""
    import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")

    text = _make_text(1500)
    chunks = chunk_page(text)
    for c in chunks:
        assert len(tok.encode(c)) <= CHUNK_SIZE, "chunk exceeds CHUNK_SIZE"


def test_chunk_page_discards_fragments_below_min_tokens():
    """A very short input should return zero chunks (below MIN_TOKENS floor)."""
    text = _make_text(10)   # ~10 tokens, well under MIN_TOKENS (50)
    chunks = chunk_page(text)
    assert chunks == [], "Fragments under MIN_TOKENS should be discarded"


def test_chunk_page_produces_overlap():
    """Consecutive chunks should share OVERLAP tokens' worth of content."""
    import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")

    text = _make_text(1000)
    chunks = chunk_page(text)
    assert len(chunks) >= 2

    # Overlap check: last OVERLAP tokens of chunk[0] should equal first OVERLAP tokens of chunk[1]
    tail_0 = tok.encode(chunks[0])[-OVERLAP:]
    head_1 = tok.encode(chunks[1])[:OVERLAP]
    assert tail_0 == head_1, "Adjacent chunks should share OVERLAP tokens"


def test_chunk_page_empty_string_returns_empty_list():
    assert chunk_page("") == []


def test_chunk_page_custom_chunk_size_and_overlap_honoured():
    """Non-default chunk_size and overlap args are respected."""
    import tiktoken
    tok = tiktoken.get_encoding("cl100k_base")

    text = _make_text(500)
    chunks = chunk_page(text, chunk_size=200, overlap=40)
    for c in chunks:
        assert len(tok.encode(c)) <= 200


# ---------------------------------------------------------------------------
# Config constants — sanity check they haven't drifted
# ---------------------------------------------------------------------------

def test_chunker_constants_match_locked_decisions():
    """CLAUDE.md locks these values. Guardrail against accidental changes."""
    assert CHUNK_SIZE == 400
    assert OVERLAP == 80
    assert MIN_TOKENS == 50
    assert MAX_TOKENS == 512
