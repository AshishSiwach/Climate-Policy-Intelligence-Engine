"""
Tombstone tests — Phase 0 Baseline Hardening.

Verifies that when a document is re-ingested with fewer chunks, the orphaned
old chunk rows are deleted from DuckDB.

These tests use DuckDB directly (no dlt pipeline) to keep the test fast and
free of PDF fixtures. The tombstone logic in ``index_manifest`` is exercised
in isolation.
"""

from __future__ import annotations

import duckdb

from ingestion.index_manifest import delete_orphaned_chunks, get_existing_chunk_ids

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TABLE = "main.chunks"  # Use the default 'main' schema for in-process DuckDB tests


def _create_table(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id  VARCHAR PRIMARY KEY,
            doc_id    VARCHAR NOT NULL,
            text      VARCHAR,
            page_number INTEGER
        )
        """
    )


def _insert_chunks(con: duckdb.DuckDBPyConnection, rows: list[tuple[str, str, str, int]]) -> None:
    """Insert (chunk_id, doc_id, text, page_number) rows."""
    for chunk_id, doc_id, text, page in rows:
        con.execute(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?)",
            [chunk_id, doc_id, text, page],
        )


def _count_chunks(con: duckdb.DuckDBPyConnection, doc_id: str) -> int:
    row = con.execute("SELECT COUNT(*) FROM chunks WHERE doc_id = ?", [doc_id]).fetchone()
    return row[0] if row else 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orphaned_chunk_deleted_on_re_ingestion(tmp_path):
    """Ingest doc with 3 chunks, re-ingest with 2, verify exactly 2 remain."""
    db_path = tmp_path / "test.duckdb"
    doc_id = "TEST_DOC"

    # ── Phase 1: first ingestion (3 chunks) ──────────────────────────────
    con = duckdb.connect(str(db_path))
    _create_table(con)
    _insert_chunks(
        con,
        [
            ("TEST_DOC_0", doc_id, "First chunk text.", 1),
            ("TEST_DOC_1", doc_id, "Second chunk text.", 2),
            ("TEST_DOC_2", doc_id, "Third chunk text.", 3),
        ],
    )
    assert _count_chunks(con, doc_id) == 3, "Setup: should have 3 chunks after first ingestion"
    con.close()

    # ── Phase 2: re-ingest with 2 chunks (doc got shorter) ───────────────
    # The new ingestion produces only chunk_ids 0 and 1; chunk 2 is orphaned.
    new_chunk_ids = {"TEST_DOC_0", "TEST_DOC_1"}

    deleted = delete_orphaned_chunks(db_path, doc_id, new_chunk_ids, table="main.chunks")

    assert deleted == 1, f"Expected 1 orphan deleted, got {deleted}"

    # ── Verify DuckDB state ───────────────────────────────────────────────
    con = duckdb.connect(str(db_path))
    remaining = _count_chunks(con, doc_id)
    con.close()

    assert remaining == 2, f"DuckDB should contain exactly 2 chunks after tombstone, got {remaining}"


def test_no_deletion_when_all_chunks_retained(tmp_path):
    """Re-ingest with same chunk set — no rows deleted."""
    db_path = tmp_path / "test.duckdb"
    doc_id = "STABLE_DOC"

    con = duckdb.connect(str(db_path))
    _create_table(con)
    _insert_chunks(
        con,
        [
            ("STABLE_DOC_0", doc_id, "Chunk A.", 1),
            ("STABLE_DOC_1", doc_id, "Chunk B.", 2),
        ],
    )
    con.close()

    deleted = delete_orphaned_chunks(db_path, doc_id, {"STABLE_DOC_0", "STABLE_DOC_1"}, table="main.chunks")
    assert deleted == 0, "No deletions expected when all chunk_ids are retained"


def test_get_existing_chunk_ids_returns_empty_for_new_doc(tmp_path):
    """First ingestion: no pre-existing chunk_ids for a new doc_id."""
    db_path = tmp_path / "empty.duckdb"

    # Database doesn't exist yet — should return empty set, not raise
    result = get_existing_chunk_ids(db_path, "BRAND_NEW_DOC", table="main.chunks")
    assert result == set()


def test_all_chunks_orphaned_when_doc_removed(tmp_path):
    """If new_chunk_ids is empty (doc removed), all existing chunks are deleted."""
    db_path = tmp_path / "test.duckdb"
    doc_id = "REMOVED_DOC"

    con = duckdb.connect(str(db_path))
    _create_table(con)
    _insert_chunks(
        con,
        [
            ("REMOVED_DOC_0", doc_id, "Will be orphaned.", 1),
            ("REMOVED_DOC_1", doc_id, "Also orphaned.", 2),
        ],
    )
    con.close()

    deleted = delete_orphaned_chunks(db_path, doc_id, set(), table="main.chunks")
    assert deleted == 2

    con = duckdb.connect(str(db_path))
    assert _count_chunks(con, doc_id) == 0
    con.close()
