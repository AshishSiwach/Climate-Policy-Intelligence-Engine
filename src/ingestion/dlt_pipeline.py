"""
dlt ingestion pipeline — writes chunk records to DuckDB.

Replaces the JSON-per-doc intermediate from Week 3. DuckDB is now the canonical
source that `build_indices.py` reads from to build BM25 + Chroma.

Layout:
    data/processed/cpie_ingestion.duckdb   (DuckDB file, gitignored)
      └── schema: cpie
          └── table:  chunks               (primary key: chunk_id)

Write disposition: merge on chunk_id — re-running the pipeline updates changed
rows and inserts new ones. Incremental ingestion works for free.

Run:
    uv run python scripts/ingest.py
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import dlt

from ingestion.chunker import chunk_document
from ingestion.index_manifest import (
    build_document_entry,
    delete_orphaned_chunks,
    sha256_file,
    write_manifest,
)
from ingestion.pdf_loader import DOC_REGISTRY

logger = logging.getLogger(__name__)

# DuckDB file lives next to the other index artefacts (all under data/processed/,
# which is gitignored).
DUCKDB_PATH = Path("data/processed/cpie_ingestion.duckdb")
DATASET_NAME = "cpie"
TABLE_NAME = "chunks"


@dlt.resource(
    name=TABLE_NAME,
    primary_key="chunk_id",
    write_disposition="merge",
)
def pdf_chunks_resource(raw_dir: str = "data/raw") -> Iterator[dict]:
    """
    Iterate DOC_REGISTRY, chunk each PDF, yield chunks with a globally-unique
    `chunk_id` composed from doc_id + chunk_index.

    Missing PDFs are logged and skipped — do not raise, so a partial corpus
    still ingests what's present.
    """
    raw_path = Path(raw_dir)

    for filename in DOC_REGISTRY:
        pdf_path = raw_path / filename
        if not pdf_path.exists():
            logger.warning("PDF not found, skipping: %s", pdf_path)
            continue

        for chunk in chunk_document(pdf_path):
            yield {
                **chunk,
                "chunk_id": f"{chunk['doc_id']}_{chunk['chunk_index']}",
            }


def build_pipeline() -> dlt.Pipeline:
    """Construct the dlt pipeline. DuckDB destination pinned to DUCKDB_PATH."""
    DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return dlt.pipeline(
        pipeline_name="cpie_ingestion",
        destination=dlt.destinations.duckdb(str(DUCKDB_PATH)),
        dataset_name=DATASET_NAME,
    )


def run_ingestion(raw_dir: str = "data/raw") -> dict:
    """Run the pipeline end-to-end. Returns dlt's LoadInfo as a dict-ish object.

    After the dlt pipeline runs:
    1. Tombstone: orphaned chunk_ids for re-ingested docs are deleted.
    2. Manifest: ``data/processed/index_manifest.json`` is written with per-doc
       content hashes, chunk counts, and ingestion timestamps.
    """
    raw_path = Path(raw_dir)

    # --- Pre-ingestion: collect the new chunks per doc_id so we know what
    # chunk_ids to expect after the merge.  We iterate the resource once here
    # (before dlt runs) just to build the metadata dict; dlt will iterate it
    # again internally.
    new_chunks_by_doc: dict[str, list[str]] = {}  # doc_id → list[chunk_id]
    doc_filenames: dict[str, str] = {}
    for filename in DOC_REGISTRY:
        pdf_path = raw_path / filename
        if not pdf_path.exists():
            continue
        meta = DOC_REGISTRY[filename]
        doc_id = meta["doc_id"]
        doc_filenames[doc_id] = filename
        # chunk_document yields chunk dicts; dlt_pipeline appends chunk_id
        doc_chunks = chunk_document(pdf_path)
        new_chunks_by_doc[doc_id] = [f"{doc_id}_{c['chunk_index']}" for c in doc_chunks]

    # --- Tombstone: before the pipeline run, identify and delete orphaned rows.
    # (dlt merge won't delete rows whose chunk_id is absent from new data.)
    for doc_id, new_ids in new_chunks_by_doc.items():
        delete_orphaned_chunks(DUCKDB_PATH, doc_id, set(new_ids))

    # --- Run the dlt pipeline (merge disposition handles inserts/updates)
    pipeline = build_pipeline()
    load_info = pipeline.run(pdf_chunks_resource(raw_dir=raw_dir))
    logger.info("Ingestion complete. LoadInfo:\n%s", load_info)

    # --- Write the index manifest
    try:
        from config.settings import settings_fingerprint

        fingerprint = settings_fingerprint()
    except Exception:
        fingerprint = "unknown"

    now = datetime.now(timezone.utc).isoformat()
    manifest_docs: dict[str, dict] = {}
    for doc_id, chunk_ids in new_chunks_by_doc.items():
        filename = doc_filenames.get(doc_id, "")
        pdf_path = raw_path / filename
        try:
            content_hash = sha256_file(pdf_path)
        except Exception:
            content_hash = "unknown"
        manifest_docs[doc_id] = build_document_entry(
            doc_id=doc_id,
            filename=filename,
            content_hash=content_hash,
            chunk_count=len(chunk_ids),
            ingested_at=now,
        )

    write_manifest(manifest_docs, pipeline_config_fingerprint=fingerprint)

    return load_info
