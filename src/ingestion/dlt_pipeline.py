"""
dlt ingestion pipeline — writes chunk records to DuckDB.

Replaces the JSON-per-doc intermediate from Week 3. DuckDB is now the canonical
source that `build_indices.py` reads from to build BM25 + Chroma.

Layout:
    data/processed/cpie_ingestion.duckdb   (DuckDB file, gitignored)
      └── schema: cpie
          └── table:  chunks               (primary key: chunk_id)

Write disposition: merge on chunk_id — re-running the pipeline updates changed
rows and inserts new ones. Free incremental ingestion (v2 roadmap item promoted
to v1 by this change).

Run:
    uv run python scripts/ingest.py
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import dlt

from ingestion.chunker import chunk_document
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
    """Run the pipeline end-to-end. Returns dlt's LoadInfo as a dict-ish object."""
    pipeline = build_pipeline()
    load_info = pipeline.run(pdf_chunks_resource(raw_dir=raw_dir))
    logger.info("Ingestion complete. LoadInfo:\n%s", load_info)
    return load_info
