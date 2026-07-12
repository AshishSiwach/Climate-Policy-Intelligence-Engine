"""
PDF loader with document-type-aware cleaning and table handling.

Tier 1 — active stripping before chunking (ESO, Ofgem)
Tier 2 — heading injection on table pages (WEO2025, Carbon Budget, CCC Progress, BoE Disclosure)
Tier 3 — keep as-is (all others)
"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")

# ---------------------------------------------------------------------------
# Document registry — one entry per PDF in data/raw/
# ---------------------------------------------------------------------------

DOC_REGISTRY = {
    "739682_ESO_Beyond2030_Report_2024_PRINT.pdf": {
        "doc_id": "ESO_BEYOND2030_2024",
        "institution": "ESO",
        "doc_type": "report",
        "jurisdiction": "UK",
        "publication_date": "2024",
        "tier1_strip": True,
        "tier2_inject": False,
    },
    "Smart-Secure-Electricity-Systems-Implementing-the-load-control-licensing-regime-consultation.pdf": {
        "doc_id": "OFGEM_SMART_SECURE_2024",
        "institution": "Ofgem",
        "doc_type": "consultation",
        "jurisdiction": "UK",
        "publication_date": "2024",
        "tier1_strip": True,
        "tier2_inject": False,
    },
    "WorldEnergyOutlook2025.pdf": {
        "doc_id": "IEA_WEO_2025",
        "institution": "IEA",
        "doc_type": "report",
        "jurisdiction": "Global",
        "publication_date": "2025",
        "tier1_strip": False,
        "tier2_inject": True,
    },
    "The-Seventh-Carbon-Budget.pdf": {
        "doc_id": "CCC_SEVENTH_CARBON_BUDGET_2025",
        "institution": "CCC",
        "doc_type": "statutory_report",
        "jurisdiction": "UK",
        "publication_date": "2025",
        "tier1_strip": False,
        "tier2_inject": True,
    },
    "Progress-in-reducing-emissions-2024-Report-to-Parliament-Web.pdf": {
        "doc_id": "CCC_PROGRESS_2024",
        "institution": "CCC",
        "doc_type": "statutory_report",
        "jurisdiction": "UK",
        "publication_date": "2024",
        "tier1_strip": False,
        "tier2_inject": True,
    },
    "Progress-in-reducing-emissions-2025-report-to-Parliament.pdf": {
        "doc_id": "CCC_PROGRESS_2025",
        "institution": "CCC",
        "doc_type": "statutory_report",
        "jurisdiction": "UK",
        "publication_date": "2025",
        "tier1_strip": False,
        "tier2_inject": True,
    },
    "boes-climate-related-financial-disclosure-2024.pdf": {
        "doc_id": "BOE_DISCLOSURE_2024",
        "institution": "BoE",
        "doc_type": "disclosure",
        "jurisdiction": "UK",
        "publication_date": "2024",
        "tier1_strip": False,
        "tier2_inject": True,
    },
    "results-of-the-2021-climate-biennial-exploratory-scenario.pdf": {
        "doc_id": "BOE_CBES_RESULTS_2021",
        "institution": "BoE",
        "doc_type": "report",
        "jurisdiction": "UK",
        "publication_date": "2021",
        "tier1_strip": False,
        "tier2_inject": False,
    },
    "key-elements-2021-biennial-exploratory-scenario-financial-risks-climate-change.pdf": {
        "doc_id": "BOE_CBES_KEY_ELEMENTS_2021",
        "institution": "BoE",
        "doc_type": "report",
        "jurisdiction": "UK",
        "publication_date": "2021",
        "tier1_strip": False,
        "tier2_inject": False,
    },
    "measuring-climate-related-financial-risks-using-scenario-analysis.pdf": {
        "doc_id": "BOE_MEASURING_CLIMATE_RISKS",
        "institution": "BoE",
        "doc_type": "report",
        "jurisdiction": "UK",
        "publication_date": "2020",
        "tier1_strip": False,
        "tier2_inject": False,
    },
    "zev-mandate-consultation-summary-of-responses-and-joint-government-response.pdf": {
        "doc_id": "DESNZ_ZEV_MANDATE_2024",
        "institution": "DESNZ",
        "doc_type": "consultation",
        "jurisdiction": "UK",
        "publication_date": "2024",
        "tier1_strip": False,
        "tier2_inject": False,
    },
    "climate-change-possible-macroeconomic-implications.pdf": {
        "doc_id": "BOE_MACRO_IMPLICATIONS",
        "institution": "BoE",
        "doc_type": "working_paper",
        "jurisdiction": "UK",
        "publication_date": "2019",
        "tier1_strip": False,
        "tier2_inject": False,
    },
}

# ---------------------------------------------------------------------------
# Tier 1 — stripping patterns
# ---------------------------------------------------------------------------

_ESO_NAV_PATTERN = re.compile(
    r'\b(Navigation|Download a pdf|Text Links|Return to contents)\b',
    re.IGNORECASE,
)
_ESO_SOCIAL_PATTERN = re.compile(r'@\S+|linkedin\.com\S*|twitter\.com\S*', re.IGNORECASE)

_OFGEM_HEADER_PATTERN = re.compile(
    r'Consultation\s+Smart\s+Secure\s+Electricity\s+Systems[^\n]*\n?',
    re.IGNORECASE,
)
_OFGEM_OFFICIAL_PATTERN = re.compile(r'\bOFFICIAL\s+OFFICIAL\b', re.IGNORECASE)

# ---------------------------------------------------------------------------
# Tier 2 — heading detection
# ---------------------------------------------------------------------------

_HEADING_PATTERN = re.compile(
    r'(\d+[\.\d]*\s+[A-Z][^\n]{5,60}|Table\s+\d+[\.\d]*[^\n]{5,60})'
)

_FILL_RATIO_THRESHOLD = 0.70  # discard table detections where >70% of cells are empty


def _is_real_table(table) -> bool:
    """Return True if the table has enough non-empty cells to be real data."""
    cells = [cell for row in table.extract() for cell in row]
    if not cells:
        return False
    filled = sum(1 for c in cells if c and str(c).strip())
    return (filled / len(cells)) >= (1 - _FILL_RATIO_THRESHOLD)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_text(text: str, doc_id: str) -> str:
    """Apply Tier 1 stripping rules for ESO and Ofgem documents."""
    if "ESO" in doc_id:
        text = _ESO_NAV_PATTERN.sub(' ', text)
        text = _ESO_SOCIAL_PATTERN.sub(' ', text)
    if "OFGEM" in doc_id:
        text = _OFGEM_HEADER_PATTERN.sub(' ', text)
        text = _OFGEM_OFFICIAL_PATTERN.sub(' ', text)
    return re.sub(r'\s{3,}', '  ', text).strip()


def detect_table_page(page: fitz.Page) -> bool:
    """Return True if the page contains at least one real table."""
    tables = page.find_tables().tables
    return any(_is_real_table(t) for t in tables)


def inject_heading(doc: fitz.Document, page_idx: int) -> str:
    """Walk back up to 3 pages to find the nearest section heading."""
    for lookback in range(page_idx, max(page_idx - 3, -1), -1):
        text = doc[lookback].get_text("text")
        match = _HEADING_PATTERN.search(text)
        if match:
            return f"[Section: {match.group(0).strip()}]"
    return ""


def load_pdf(path: str | Path) -> list[dict]:
    """
    Load a PDF and return a list of page dicts ready for chunking.

    Each dict contains:
        text, doc_id, institution, doc_type, jurisdiction,
        publication_date, page_number, chunk_type
    """
    path = Path(path)
    filename = path.name
    meta = DOC_REGISTRY.get(filename)
    if meta is None:
        raise ValueError(f"PDF not in registry: {filename}. Add it to DOC_REGISTRY.")

    doc_id = meta["doc_id"]
    doc = fitz.open(path)
    pages = []

    for page_idx, page in enumerate(doc):
        raw_text = page.get_text("text")
        if not raw_text.strip():
            continue

        text = clean_text(raw_text, doc_id) if meta["tier1_strip"] else raw_text

        is_table = detect_table_page(page)
        chunk_type = "table" if is_table else "prose"

        # Tier 2 — heading injection on table pages
        heading_prefix = ""
        if meta["tier2_inject"] and is_table:
            heading_prefix = inject_heading(doc, page_idx)

        pages.append({
            "text": text,
            "doc_id": doc_id,
            "institution": meta["institution"],
            "doc_type": meta["doc_type"],
            "jurisdiction": meta["jurisdiction"],
            "publication_date": meta["publication_date"],
            "page_number": page_idx + 1,
            "chunk_type": chunk_type,
            "heading_prefix": heading_prefix,
        })

    doc.close()
    logger.info("Loaded %s: %d pages", doc_id, len(pages))
    return pages
