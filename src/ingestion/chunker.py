"""
Sliding-window chunker.

Splits page text into fixed-size chunks using tiktoken cl100k_base.
Chunk size: 400 tokens, overlap: 80 tokens (locked in config.yaml).
Floor: 50 tokens — discard fragments below this.
Ceiling: 512 tokens — reranker constraint. Assert fires after heading injection.

Heading prefix (from pdf_loader.py Tier 2 injection on table pages) is prepended
to every chunk from that page.
"""

import json
import logging
from pathlib import Path

import tiktoken

from ingestion.pdf_loader import DOC_REGISTRY, load_pdf

logger = logging.getLogger(__name__)

_tokenizer = tiktoken.get_encoding("cl100k_base")

CHUNK_SIZE = 400
OVERLAP = 80
MIN_TOKENS = 50
MAX_TOKENS = 512


def chunk_page(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into token-bounded chunks with overlap. Discards fragments below MIN_TOKENS."""
    tokens = _tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        if len(chunk_tokens) >= MIN_TOKENS:
            chunks.append(_tokenizer.decode(chunk_tokens))
        start += chunk_size - overlap
    return chunks


def chunk_document(path: str | Path, output_dir: str | Path) -> list[dict]:
    """
    Load a PDF, apply sliding-window chunking, validate, save to output_dir.
    Returns the list of chunk dicts saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pages = load_pdf(path)
    if not pages:
        logger.warning("No pages extracted from %s", path)
        return []

    all_chunks = []
    chunk_idx = 0

    for page in pages:
        heading_prefix = page.get("heading_prefix", "")
        meta = {k: v for k, v in page.items() if k != "heading_prefix"}

        for chunk_text in chunk_page(page["text"]):
            final_text = f"{heading_prefix}\n{chunk_text}" if heading_prefix else chunk_text
            token_count = len(_tokenizer.encode(final_text))
            assert token_count <= MAX_TOKENS, (
                f"Chunk exceeds {MAX_TOKENS}-token ceiling after heading injection: "
                f"{token_count} tokens in {meta['doc_id']} page {meta['page_number']}"
            )
            all_chunks.append({
                **meta,
                "text": final_text,
                "chunk_index": chunk_idx,
                "token_count": token_count,
            })
            chunk_idx += 1

    if not all_chunks:
        logger.warning("No chunks produced for %s", pages[0]["doc_id"])
        return []

    doc_id = pages[0]["doc_id"]
    out_path = output_dir / f"{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    avg_tokens = sum(c["token_count"] for c in all_chunks) / len(all_chunks)
    logger.info(
        "Chunked %s: %d chunks, avg %.0f tokens, max %d tokens",
        doc_id, len(all_chunks), avg_tokens,
        max(c["token_count"] for c in all_chunks),
    )
    return all_chunks


def chunk_all(raw_dir: str | Path, output_dir: str | Path) -> dict:
    """Chunk all registered PDFs in raw_dir. Returns summary stats per doc."""
    raw_dir = Path(raw_dir)
    stats = {}

    for filename, meta in DOC_REGISTRY.items():
        pdf_path = raw_dir / filename
        if not pdf_path.exists():
            logger.warning("PDF not found, skipping: %s", pdf_path)
            continue
        chunks = chunk_document(pdf_path, output_dir)
        if chunks:
            stats[meta["doc_id"]] = {
                "chunk_count": len(chunks),
                "avg_tokens": sum(c["token_count"] for c in chunks) / len(chunks),
                "max_tokens": max(c["token_count"] for c in chunks),
            }

    total = sum(s["chunk_count"] for s in stats.values())
    logger.info("Total chunks across all documents: %d", total)
    return stats
