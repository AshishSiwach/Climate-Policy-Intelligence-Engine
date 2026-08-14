"""
Build BM25 and Chroma dense indices over all chunked documents.

Reads: data/processed/cpie_ingestion.duckdb  (schema: cpie, table: chunks)
Writes:
  data/processed/bm25_index.pkl   (pickled BM25Okapi + chunks)
  data/processed/chroma_db/       (Chroma persistent client)

Run:
  uv run python scripts/ingest.py           # populate DuckDB first
  uv run python scripts/build_indices.py    # then build BM25 + Chroma
"""

import logging
import time
from pathlib import Path

import duckdb

from ingestion.dlt_pipeline import DATASET_NAME, DUCKDB_PATH, TABLE_NAME
from retrieval import BM25Retriever, DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_indices")

PROCESSED_DIR = Path("data/processed")
BM25_PATH = PROCESSED_DIR / "bm25_index.pkl"
CHROMA_DIR = PROCESSED_DIR / "chroma_db"


def load_all_chunks() -> list[dict]:
    """
    Load every chunk from DuckDB into a flat list ordered by (doc_id, chunk_index).

    Casts the DataFrame back to a list of plain dicts because the retrievers
    expect that shape (they iterate .items() / index by string keys).
    """
    if not DUCKDB_PATH.exists():
        raise RuntimeError(f"DuckDB file not found at {DUCKDB_PATH}. Run `uv run python scripts/ingest.py` first.")

    with duckdb.connect(str(DUCKDB_PATH), read_only=True) as conn:
        df = conn.execute(f"SELECT * FROM {DATASET_NAME}.{TABLE_NAME} ORDER BY doc_id, chunk_index").fetchdf()

    chunks = df.to_dict("records")
    logger.info("Loaded %d chunks from %s", len(chunks), DUCKDB_PATH)
    return chunks


def main() -> None:
    chunks = load_all_chunks()
    if not chunks:
        raise RuntimeError(f"No chunks found in {PROCESSED_DIR}. Run ingestion first.")

    # BM25
    t0 = time.time()
    bm25 = BM25Retriever()
    bm25.build(chunks)
    bm25.save(BM25_PATH)
    logger.info("BM25 built + saved in %.1fs", time.time() - t0)

    # Dense (Chroma)
    t0 = time.time()
    dense = DenseRetriever(persist_dir=CHROMA_DIR)
    dense.build(chunks)
    logger.info("Chroma built + persisted in %.1fs", time.time() - t0)

    logger.info("Done. BM25=%s  Chroma=%s", BM25_PATH, CHROMA_DIR)


if __name__ == "__main__":
    main()
