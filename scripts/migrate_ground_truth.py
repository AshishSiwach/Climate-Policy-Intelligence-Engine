"""
Migrate ground_truth_raw.json (as-authored schema) to ground_truth.json
(enriched schema with per-doc expected_sources).

As-authored schema (per pair):
  expected_source_docs: list[str]
  expected_page_range: [start, end] | null

Enriched schema (per pair):
  expected_sources: list[{doc_id: str, page_range: [start, end] | null}]

Migration rules:
  - Single doc + range: one entry {doc_id, page_range}
  - Single doc + null range: one entry {doc_id, page_range=null}
  - Multi doc + any range value: fan out to per-doc entries, page_range applied
    only to the FIRST doc (all others null) — because the as-authored schema
    has no per-doc range concept; the range only makes sense for the primary source
  - Empty docs (negatives): []

Also validates:
  - All doc_ids match DOC_REGISTRY doc_ids
  - No duplicate ids across pairs
  - Every pair has all required fields
  - query_types are within the allowed set

Run:
  uv run python scripts/migrate_ground_truth.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from ingestion.pdf_loader import DOC_REGISTRY

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate_ground_truth")

RAW_PATH = Path("data/eval/ground_truth_raw.json")
OUT_PATH = Path("data/eval/ground_truth.json")

ALLOWED_QUERY_TYPES = {"factual", "cross_document", "summarisation", "numeric", "negative"}
REQUIRED_FIELDS = {
    "id", "query_type", "question", "expected_answer",
    "expected_source_docs", "expected_page_range", "notes",
}

# Ground-truth authoring used slightly different doc_id naming than DOC_REGISTRY.
# These aliases translate at migration time so the raw ground truth file stays
# human-readable and future edits don't need to memorise our internal ids.
DOC_ID_ALIASES = {
    # Naming / date drift between authored ground truth and DOC_REGISTRY.
    # DOC_REGISTRY was corrected (see pdf_loader.py) after PDF front-matter verification;
    # aliases translate the authored ids in ground_truth_raw.json to the canonical ids.
    "ZEV_MANDATE_2023":             "DESNZ_ZEV_MANDATE_2023",
    "OFGEM_SMART_SECURE_2024":      "OFGEM_SMART_SECURE_2025",
    "BOE_MEASURING_RISKS_SCENARIO": "BOE_MEASURING_CLIMATE_RISKS",
    "BOE_CLIMATE_DISCLOSURE_2024":  "BOE_DISCLOSURE_2024",
    "CCC_SEVENTH_BUDGET_2025":      "CCC_SEVENTH_CARBON_BUDGET_2025",
}


def _resolve(doc_id: str) -> str:
    """Apply DOC_ID_ALIASES to translate authored id → canonical id."""
    return DOC_ID_ALIASES.get(doc_id, doc_id)


def _valid_doc_ids() -> set[str]:
    return {meta["doc_id"] for meta in DOC_REGISTRY.values()}


def _migrate_one(pair: dict, valid_docs: set[str]) -> dict:
    """Return the enriched pair. Raises if validation fails."""
    missing = REQUIRED_FIELDS - pair.keys()
    if missing:
        raise ValueError(f"Pair {pair.get('id', '?')!r} missing fields: {missing}")

    qt = pair["query_type"]
    if qt not in ALLOWED_QUERY_TYPES:
        raise ValueError(f"Pair {pair['id']} has unknown query_type: {qt!r}")

    docs_raw: list[str] = pair["expected_source_docs"]
    docs: list[str] = [_resolve(d) for d in docs_raw]   # apply aliases
    page_range = pair["expected_page_range"]   # [start, end] | null

    # Validate doc_ids AFTER alias resolution
    if qt != "negative":
        unknown = [d for d in docs if d not in valid_docs]
        if unknown:
            raise ValueError(
                f"Pair {pair['id']} references unknown doc_ids (after alias resolution): {unknown}. "
                f"Valid ids: {sorted(valid_docs)}"
            )

    # Build expected_sources per migration rules
    if not docs:                         # negative — empty
        expected_sources: list[dict] = []
    elif len(docs) == 1:                 # single-doc
        expected_sources = [{"doc_id": docs[0], "page_range": page_range}]
    else:                                # multi-doc (cross_document)
        # Apply page_range to first doc only (usually null anyway for cross-doc);
        # remaining docs get null. Post-hoc enrichment in Week 5 can fill in per-doc
        # ranges via inspection.
        expected_sources = [
            {"doc_id": docs[0], "page_range": page_range},
            *[{"doc_id": d, "page_range": None} for d in docs[1:]],
        ]

    return {
        "id": pair["id"],
        "query_type": qt,
        "question": pair["question"],
        "expected_answer": pair["expected_answer"],
        "expected_sources": expected_sources,
        "notes": pair["notes"],
    }


def _validate_ids_unique(pairs: list[dict]) -> None:
    ids = [p["id"] for p in pairs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"Duplicate ids found: {duplicates}")


def main() -> None:
    if not RAW_PATH.exists():
        logger.error("Raw ground truth not found at %s", RAW_PATH)
        sys.exit(1)

    with open(RAW_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        logger.error("Raw ground truth must be a JSON list. Got %s", type(raw).__name__)
        sys.exit(1)

    _validate_ids_unique(raw)

    valid_docs = _valid_doc_ids()
    migrated: list[dict] = []
    errors: list[str] = []
    for pair in raw:
        try:
            migrated.append(_migrate_one(pair, valid_docs))
        except ValueError as e:
            errors.append(str(e))

    if errors:
        logger.error("Migration failed with %d error(s):", len(errors))
        for e in errors:
            logger.error("  %s", e)
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(migrated, f, indent=2, ensure_ascii=False)

    # ── Report ────────────────────────────────────────────────────────────
    by_type: dict[str, int] = {}
    with_page_range = 0
    without_page_range = 0
    total_sources = 0
    for m in migrated:
        by_type[m["query_type"]] = by_type.get(m["query_type"], 0) + 1
        for src in m["expected_sources"]:
            total_sources += 1
            if src["page_range"] is not None:
                with_page_range += 1
            else:
                without_page_range += 1

    logger.info("Migrated %d pairs → %s", len(migrated), OUT_PATH)
    logger.info("By query_type:")
    for qt in sorted(by_type):
        logger.info("  %-15s %d", qt, by_type[qt])
    logger.info(
        "Source-level page range coverage: %d/%d populated (%.0f%%)",
        with_page_range, total_sources,
        100 * with_page_range / max(total_sources, 1),
    )
    if DOC_ID_ALIASES:
        logger.info("Doc-id aliases applied during migration:")
        for authored, canonical in DOC_ID_ALIASES.items():
            logger.info("  %-32s → %s", authored, canonical)


if __name__ == "__main__":
    main()
