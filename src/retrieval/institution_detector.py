"""
Institution detection for query-time metadata filtering.

Scans the query for mentions of the 6 institutions in DOC_REGISTRY and returns
the ones detected. Used by HybridRetriever to pre-filter the corpus to just
the named institutions when the query explicitly mentions them.

Added Week 5 Step after cross-doc
top-k coverage analysis (Step 2c) showed that 4 of 6 multi-source queries
name their expected institutions explicitly, but retrieval still misses the
second source even at k=20 because the missed doc is semantically far from
the query wording. Filter directly fixes that failure mode by guaranteeing
per-source coverage.

Detection is case-insensitive with word boundaries. Fall-back behaviour
(zero-match after filter) lives in HybridRetriever, not here — this module
is pure detection.
"""

from __future__ import annotations

import re

# Institution → list of regex patterns that indicate a mention.
# Values map exactly to the `institution` metadata field written by pdf_loader:
# {"ESO", "Ofgem", "IEA", "CCC", "BoE", "DESNZ"}.
_INSTITUTION_PATTERNS: dict[str, re.Pattern] = {
    "Ofgem": re.compile(r"\bofgem\b", re.IGNORECASE),
    "DESNZ": re.compile(
        r"\bdesnz\b|\bdepartment for energy security and net zero\b",
        re.IGNORECASE,
    ),
    "IEA": re.compile(
        r"\biea\b|\binternational energy agency\b",
        re.IGNORECASE,
    ),
    "BoE": re.compile(
        r"\bboe\b|\bbank of england\b",
        re.IGNORECASE,
    ),
    "CCC": re.compile(
        r"\bccc\b|\bclimate change committee\b",
        re.IGNORECASE,
    ),
    "ESO": re.compile(
        r"\beso\b|\bnational grid\b|\belectricity system operator\b",
        re.IGNORECASE,
    ),
}


def detect_institutions(query: str) -> list[str]:
    """
    Return the list of institution names mentioned in the query.

    Order-preserving (matches _INSTITUTION_PATTERNS insertion order). Empty
    list if none are mentioned — caller (HybridRetriever) uses that as the
    signal to skip filtering entirely.
    """
    if not query:
        return []
    return [name for name, pattern in _INSTITUTION_PATTERNS.items()
            if pattern.search(query)]
