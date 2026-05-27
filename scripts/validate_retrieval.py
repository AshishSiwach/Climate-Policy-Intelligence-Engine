"""
Week 2 Step 5 — Retrieval pipeline validation.

Uses 3 pilot PDFs (already validated in audit notebook) to build a
minimal in-memory corpus and run all four retrieval modules end-to-end.

This is a smoke test: confirms module interfaces work before the full
ingestion pipeline is built in Week 3.

Usage:
    uv run python scripts/validate_retrieval.py
"""

import logging
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF
import tiktoken

# ── project root on path ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
logger = logging.getLogger(__name__)

# ── config (locked Week 2) ────────────────────────────────────────────
CHUNK_SIZE = 400
OVERLAP = 80
MIN_TOKENS = 50
TOKENIZER = tiktoken.get_encoding('cl100k_base')

PILOT_DOCS = [
    {
        'path': ROOT / 'data/raw/Smart-Secure-Electricity-Systems-Implementing-the-load-control-licensing-regime-consultation.pdf',
        'doc_id': 'OFGEM_SSES_2025',
        'institution': 'Ofgem',
        'doc_type': 'consultation',
        'jurisdiction': 'UK',
        'publication_date': '2025',
    },
    {
        'path': ROOT / 'data/raw/results-of-the-2021-climate-biennial-exploratory-scenario.pdf',
        'doc_id': 'BOE_CBES_RESULTS_2022',
        'institution': 'FCA/PRA',
        'doc_type': 'results_report',
        'jurisdiction': 'UK',
        'publication_date': '2022',
    },
    {
        'path': ROOT / 'data/raw/WorldEnergyOutlook2025.pdf',
        'doc_id': 'IEA_WEO_2025',
        'institution': 'IEA',
        'doc_type': 'outlook_report',
        'jurisdiction': 'Global',
        'publication_date': '2025',
    },
]

VALIDATION_QUERIES = [
    {
        'query': 'What load control licensing requirements does Ofgem propose?',
        'expected_source': 'OFGEM_SSES_2025',
    },
    {
        'query': 'What aggregate loan losses did UK banks face under the CBES early action scenario?',
        'expected_source': 'BOE_CBES_RESULTS_2022',
    },
    {
        'query': 'What does the IEA project for peak global fossil fuel demand?',
        'expected_source': 'IEA_WEO_2025',
    },
]


# ── helpers ───────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    return len(TOKENIZER.encode(text))


def chunk_text(text: str, doc_meta: dict) -> list[dict]:
    """Tokenise → sliding window → decode back to strings."""
    tokens = TOKENIZER.encode(text)
    step = CHUNK_SIZE - OVERLAP
    chunks = []
    for start in range(0, len(tokens), step):
        token_slice = tokens[start: start + CHUNK_SIZE]
        if len(token_slice) < MIN_TOKENS:
            break
        chunk_text = TOKENIZER.decode(token_slice)
        chunks.append({
            'text': chunk_text,
            'doc_id': doc_meta['doc_id'],
            'institution': doc_meta['institution'],
            'doc_type': doc_meta['doc_type'],
            'jurisdiction': doc_meta['jurisdiction'],
            'publication_date': doc_meta['publication_date'],
            'page_number': -1,       # not tracked at this level in the smoke test
            'chunk_index': len(chunks),
            'token_count': len(token_slice),
            'chunk_type': 'prose',   # simplified — no table detection in smoke test
        })
    return chunks


def extract_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    text = '\n'.join(page.get_text('text') for page in doc)
    doc.close()
    return text


def build_corpus() -> list[dict]:
    all_chunks = []
    for meta in PILOT_DOCS:
        path = meta['path']
        if not path.exists():
            logger.error("PDF not found: %s — skipping", path)
            continue
        logger.info("Extracting: %s", path.name)
        text = extract_text(path)
        chunks = chunk_text(text, meta)
        logger.info("  → %d chunks (doc_id=%s)", len(chunks), meta['doc_id'])
        all_chunks.extend(chunks)
    logger.info("Corpus total: %d chunks across %d docs\n", len(all_chunks), len(PILOT_DOCS))
    return all_chunks


# ── validation ────────────────────────────────────────────────────────

def print_separator(label: str) -> None:
    print(f'\n{"="*70}')
    print(f'  {label}')
    print('='*70)


def validate_bm25(chunks: list[dict]) -> BM25Retriever:
    print_separator('BM25 RETRIEVER')
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    bm25.build(chunks)

    for vq in VALIDATION_QUERIES:
        results = bm25.query(vq['query'], top_k=5)
        top_source = results[0]['doc_id'] if results else 'NO RESULTS'
        hit = '[PASS]' if top_source == vq['expected_source'] else '[FAIL]'
        print(f'\n  {hit} Q: {vq["query"][:65]}')
        print(f'     Expected source: {vq["expected_source"]}')
        for rank, r in enumerate(results[:3], 1):
            print(f'     [{rank}] score={r["bm25_score"]:.3f}  src={r["doc_id"]}')
            print(f'         {r["text"][:100].replace(chr(10), " ")}')
    return bm25


def validate_dense(chunks: list[dict]) -> DenseRetriever:
    print_separator('DENSE RETRIEVER  (BAAI/bge-base-en-v1.5)')
    dense = DenseRetriever(
        persist_dir=ROOT / 'data/processed/chroma_db_validate',
        collection_name='cpie_validate',
    )
    dense.build(chunks)

    for vq in VALIDATION_QUERIES:
        results = dense.query(vq['query'], top_k=5)
        top_source = results[0]['doc_id'] if results else 'NO RESULTS'
        hit = '[PASS]' if top_source == vq['expected_source'] else '[FAIL]'
        print(f'\n  {hit} Q: {vq["query"][:65]}')
        print(f'     Expected source: {vq["expected_source"]}')
        for rank, r in enumerate(results[:3], 1):
            print(f'     [{rank}] score={r["dense_score"]:.3f}  src={r["doc_id"]}')
            print(f'         {r["text"][:100].replace(chr(10), " ")}')
    return dense


def validate_hybrid(bm25: BM25Retriever, dense: DenseRetriever) -> HybridRetriever:
    print_separator('HYBRID RETRIEVER  (RRF k=60)')
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)

    for vq in VALIDATION_QUERIES:
        results = hybrid.retrieve(vq['query'], top_k=5)
        top_source = results[0]['doc_id'] if results else 'NO RESULTS'
        hit = '[PASS]' if top_source == vq['expected_source'] else '[FAIL]'
        print(f'\n  {hit} Q: {vq["query"][:65]}')
        print(f'     Expected source: {vq["expected_source"]}')
        for rank, r in enumerate(results[:3], 1):
            print(f'     [{rank}] rrf={r["rrf_score"]:.5f}  src={r["doc_id"]}')
            print(f'         {r["text"][:100].replace(chr(10), " ")}')
    return hybrid


def validate_reranker(hybrid: HybridRetriever) -> None:
    print_separator('RERANKER  (cross-encoder/ms-marco-MiniLM-L-6-v2)')
    reranker = Reranker(top_k=5)

    for vq in VALIDATION_QUERIES:
        candidates = hybrid.retrieve(vq['query'], top_k=20)
        results = reranker.rerank(vq['query'], candidates)
        top_source = results[0]['doc_id'] if results else 'NO RESULTS'
        hit = '[PASS]' if top_source == vq['expected_source'] else '[FAIL]'
        latency = results[0].get('rerank_latency_ms', 0) if results else 0
        print(f'\n  {hit} Q: {vq["query"][:65]}  [{latency:.0f}ms]')
        print(f'     Expected source: {vq["expected_source"]}')
        for rank, r in enumerate(results[:3], 1):
            print(f'     [{rank}] rerank={r["rerank_score"]:.3f}  src={r["doc_id"]}')
            print(f'         {r["text"][:100].replace(chr(10), " ")}')


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    print('\nCPIE — Retrieval Pipeline Validation (Week 2 Step 5)')
    print('Pilot corpus: Ofgem SSES + CBES Results + WEO 2025\n')

    chunks = build_corpus()
    if not chunks:
        logger.error("No chunks built — check PDF paths in PILOT_DOCS.")
        sys.exit(1)

    bm25 = validate_bm25(chunks)
    dense = validate_dense(chunks)
    hybrid = validate_hybrid(bm25, dense)
    validate_reranker(hybrid)

    # Clean up test Chroma collection
    dense.delete_collection()

    print('\n' + '='*70)
    print('  Validation complete.')
    print('  [PASS] = top-1 result was from expected source document.')
    print('='*70 + '\n')


if __name__ == '__main__':
    main()
