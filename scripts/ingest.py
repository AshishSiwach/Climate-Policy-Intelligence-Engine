"""
Ingest all PDFs in data/raw/ into DuckDB via the dlt pipeline.

Run:
    uv run python scripts/ingest.py

Output:
    data/processed/cpie_ingestion.duckdb  (schema: cpie, table: chunks)
"""

import logging

from ingestion.dlt_pipeline import run_ingestion

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


if __name__ == "__main__":
    load_info = run_ingestion()
    print(load_info)
