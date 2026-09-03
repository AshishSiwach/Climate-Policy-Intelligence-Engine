"""
Index manifest and ingestion tombstone utilities — Phase 0 Baseline Hardening.

Manifest
--------
After every ingestion run, write ``data/processed/index_manifest.json`` with a
summary of what was ingested.  The file is in ``data/processed/`` which is
gitignored — it tracks run-time state, not source.

Tombstones
----------
dlt's ``merge`` write-disposition updates/inserts chunks by primary key
(``chunk_id``).  If a document shrinks (e.g. 3 chunks → 2 chunks), the
orphaned third chunk survives in DuckDB because dlt only touches chunk_ids
it receives.  ``delete_orphaned_chunks()`` closes this gap by deleting any
chunk_id for a doc_id that is NOT in the new set.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import duckdb

logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("data/processed/index_manifest.json")


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------


def build_document_entry(
    doc_id: str,
    filename: str,
    content_hash: str,
    chunk_count: int,
    ingested_at: str | None = None,
) -> dict:
    """Build a single document entry for the manifest."""
    return {
        "filename": filename,
        "content_hash": content_hash,
        "chunk_count": chunk_count,
        "ingested_at": ingested_at or datetime.now(timezone.utc).isoformat(),
    }


def write_manifest(
    documents: dict[str, dict],
    pipeline_config_fingerprint: str = "unknown",
    manifest_path: Path = MANIFEST_PATH,
) -> None:
    """Write the index manifest to disk.

    Args:
        documents: ``{doc_id: build_document_entry(...)}`` mapping.
        pipeline_config_fingerprint: 8-char SHA-256 hex from ``settings_fingerprint()``.
        manifest_path: Where to write the manifest (default: MANIFEST_PATH).
    """
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_config_fingerprint": pipeline_config_fingerprint,
        "documents": documents,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Index manifest written to %s (%d documents)", manifest_path, len(documents))


def sha256_file(path: str | Path) -> str:
    """Return hex SHA-256 of a file's raw bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tombstone / orphan deletion
# ---------------------------------------------------------------------------


def get_existing_chunk_ids(db_path: str | Path, doc_id: str, table: str = "cpie.chunks") -> set[str]:
    """Query DuckDB for all chunk_ids belonging to a doc_id.

    Returns an empty set if the table doesn't exist yet (first run).
    """
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        try:
            rows = con.execute(
                f"SELECT chunk_id FROM {table} WHERE doc_id = ?",  # noqa: S608
                [doc_id],
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            con.close()
    except Exception as exc:
        # Table may not exist on first run — that's OK.
        logger.debug("Could not query existing chunk_ids for doc_id=%s: %s", doc_id, exc)
        return set()


def delete_orphaned_chunks(
    db_path: str | Path,
    doc_id: str,
    new_chunk_ids: set[str],
    table: str = "cpie.chunks",
) -> int:
    """Delete chunk_ids for ``doc_id`` that are NOT in ``new_chunk_ids``.

    This is the tombstone step: after a document is re-ingested with fewer
    chunks, the orphaned old chunk rows are removed.

    Returns the number of rows deleted (0 if nothing to clean up).
    """
    existing = get_existing_chunk_ids(db_path, doc_id, table)
    orphans = existing - new_chunk_ids
    if not orphans:
        return 0

    placeholders = ", ".join("?" * len(orphans))
    try:
        con = duckdb.connect(str(db_path))
        try:
            con.execute(
                f"DELETE FROM {table} WHERE chunk_id IN ({placeholders})",  # noqa: S608
                list(orphans),
            )
            count = len(orphans)
            logger.info(
                "Tombstoned %d orphaned chunk(s) for doc_id=%s: %s",
                count,
                doc_id,
                sorted(orphans),
            )
            return count
        finally:
            con.close()
    except Exception as exc:
        logger.warning("Failed to delete orphaned chunks for doc_id=%s: %s", doc_id, exc)
        return 0
