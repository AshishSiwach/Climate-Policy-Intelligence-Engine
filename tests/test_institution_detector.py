"""
Unit tests for institution detection (query-time metadata filtering).

The detector must:
  - match all 6 corpus institutions by acronym + expanded name
  - be case-insensitive
  - respect word boundaries (no false positives on substrings)
  - return names in a stable, deterministic order
  - return [] on empty input or no matches
"""

from __future__ import annotations

import pytest

from retrieval.institution_detector import detect_institutions

# ---------------------------------------------------------------------------
# Positive cases — each institution matches its own name + variants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected",
    [
        ("What does Ofgem say about load control?", ["Ofgem"]),
        ("ofgem lowercase should match", ["Ofgem"]),
        ("OFGEM uppercase should match", ["Ofgem"]),
        ("DESNZ policy on ZEV mandate", ["DESNZ"]),
        ("Department for Energy Security and Net Zero response", ["DESNZ"]),
        ("What does the IEA project?", ["IEA"]),
        ("International Energy Agency scenario", ["IEA"]),
        ("BoE CBES results", ["BoE"]),
        ("Bank of England disclosure", ["BoE"]),
        ("CCC Seventh Carbon Budget", ["CCC"]),
        ("Climate Change Committee progress report", ["CCC"]),
        ("ESO Beyond 2030 report", ["ESO"]),
        ("National Grid transmission plan", ["ESO"]),
        ("Electricity System Operator forecast", ["ESO"]),
    ],
)
def test_single_institution_detected(query, expected):
    assert detect_institutions(query) == expected


# ---------------------------------------------------------------------------
# Multi-institution queries
# ---------------------------------------------------------------------------


def test_cross_doc_query_returns_both_institutions():
    result = detect_institutions("How does the Bank of England's Late Action scenario compare with the IEA's framing?")
    assert set(result) == {"BoE", "IEA"}


def test_three_institution_query_returns_all_three():
    result = detect_institutions("Compare Ofgem, DESNZ, and the CCC on decarbonisation")
    assert set(result) == {"Ofgem", "DESNZ", "CCC"}


# ---------------------------------------------------------------------------
# Order preservation — insertion order of _INSTITUTION_PATTERNS
# ---------------------------------------------------------------------------


def test_result_order_matches_pattern_definition_order():
    # Patterns defined in order: Ofgem, DESNZ, IEA, BoE, CCC, ESO
    # Query mentions IEA before Ofgem in text — result should still be Ofgem first
    result = detect_institutions("IEA and Ofgem both weigh in")
    assert result == ["Ofgem", "IEA"]


# ---------------------------------------------------------------------------
# Word boundaries — avoid false positives on substrings
# ---------------------------------------------------------------------------


def test_boeing_does_not_match_boe():
    """'boe' is a substring of 'boeing' — \\b must prevent that match."""
    assert "BoE" not in detect_institutions("Boeing 737 climate impact")


def test_idea_does_not_match_iea():
    assert "IEA" not in detect_institutions("What's your idea about climate?")


def test_no_stray_ccc_match_in_technical_string():
    """Bare 'ccc' as a substring in 'accccommodate' shouldn't match."""
    # Real word test — 'access' contains 'cc' but not 'ccc' as a bounded word
    assert detect_institutions("access to energy is a policy priority") == []


# ---------------------------------------------------------------------------
# Empty / no-match cases
# ---------------------------------------------------------------------------


def test_empty_query_returns_empty():
    assert detect_institutions("") == []


def test_none_free_generic_query_returns_empty():
    assert detect_institutions("What are the main greenhouse gases?") == []


def test_climate_policy_query_without_named_institution_returns_empty():
    assert detect_institutions("How is heat pump adoption progressing?") == []
