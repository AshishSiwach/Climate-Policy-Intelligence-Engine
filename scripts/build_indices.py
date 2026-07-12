"""
Build BM25 and Chroma dense indices over all chunked documents.

Reads: data/processed/*.json  (chunk output from ingestion pipeline)
Writes:
  data/processed/bm25_index.pkl   (pickled BM25Okapi + chunks)
  data/processed/chroma_db/       (Chroma persistent client)

Run:
  uv run python scripts/build_indices.py
"""

import json
import logging
import time
from pathlib import Path

from retrieval import BM25Retriever, DenseRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("build_indices")

PROCESSED_DIR = Path("data/processed")
BM25_PATH = PROCESSED_DIR / "bm25_index.pkl"
CHROMA_DIR = PROCESSED_DIR / "chroma_db"


def load_all_chunks() -> list[dict]:
    """Load every chunk JSON from data/processed/ into a flat list."""
    chunks = []
    for path in sorted(PROCESSED_DIR.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            chunks.extend(json.load(f))
    logger.info("Loaded %d chunks from %d files", len(chunks), len(list(PROCESSED_DIR.glob("*.json"))))
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
