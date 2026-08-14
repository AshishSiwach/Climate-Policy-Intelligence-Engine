"""
Sliding-window chunker.

Splits page text into fixed-size chunks using tiktoken cl100k_base.
Chunk size: 400 tokens, overlap: 80 tokens (locked in config.yaml).
Floor: 50 tokens — discard fragments below this.
Ceiling: 512 tokens — reranker constraint. Assert fires after heading injection.

Heading prefix (from pdf_loader.py Tier 2 injection on table pages) is prepended
to every chunk from that page.

Persistence lives in `dlt_pipeline.py` (Week 5): the dlt resource iterates
DOC_REGISTRY, calls `chunk_document()` here, and writes chunks to DuckDB.
This module returns chunks; it no longer writes JSON.
"""

import logging
from pathlib import Path

import tiktoken

from ingestion.pdf_loader import load_pdf

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


def chunk_document(path: str | Path) -> list[dict]:
    """
    Load a PDF, apply sliding-window chunking + Tier 2 heading injection,
    validate token ceiling. Returns a list of chunk dicts; does NOT persist.

    Each chunk dict carries: doc_id, institution, doc_type, jurisdiction,
    publication_date, page_number, chunk_type, chunk_index, token_count, text.
    """
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
            all_chunks.append(
                {
                    **meta,
                    "text": final_text,
                    "chunk_index": chunk_idx,
                    "token_count": token_count,
                }
            )
            chunk_idx += 1

    if not all_chunks:
        logger.warning("No chunks produced for %s", pages[0]["doc_id"])
        return []

    doc_id = pages[0]["doc_id"]
    avg_tokens = sum(c["token_count"] for c in all_chunks) / len(all_chunks)
    logger.info(
        "Chunked %s: %d chunks, avg %.0f tokens, max %d tokens",
        doc_id,
        len(all_chunks),
        avg_tokens,
        max(c["token_count"] for c in all_chunks),
    )
    return all_chunks
