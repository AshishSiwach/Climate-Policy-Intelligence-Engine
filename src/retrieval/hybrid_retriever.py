"""
Hybrid retriever: BM25 + dense fusion via Reciprocal Rank Fusion (RRF).

Locked decisions (Week 2 audit):
  - RRF k = 60  (Cormack et al. 2009 default; conservative for small corpus)
  - top_k = 20  candidates passed to reranker
"""

import logging

from retrieval.bm25_retriever import BM25Retriever
from retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int) -> float:
    """Reciprocal Rank Fusion score for a document at position `rank`."""
    return 1.0 / (k + rank)


class HybridRetriever:
    """Fuses BM25 and dense retrieval results using RRF.

    Parameters
    ----------
    bm25 : BM25Retriever
        Built BM25 index.
    dense : DenseRetriever
        Built dense index (Chroma).
    rrf_k : int
        RRF smoothing constant. Locked at 60.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.rrf_k = rrf_k

    def retrieve(self, query: str, top_k: int = 20) -> list[dict]:
        """Retrieve and fuse results from BM25 and dense, return top-k by RRF score.

        Queries each retriever for 2×top_k candidates, fuses with RRF,
        and returns the top-k unique chunks by combined score.

        Each returned dict contains all chunk metadata plus:
          - ``rrf_score`` (float)  — combined RRF score
          - ``bm25_rank`` (int)    — rank in BM25 results (if retrieved)
          - ``dense_rank`` (int)   — rank in dense results (if retrieved)
        """
        candidate_pool = top_k * 2

        bm25_results = self.bm25.query(query, top_k=candidate_pool)
        dense_results = self.dense.query(query, top_k=candidate_pool)

        # Accumulate RRF scores keyed by stable chunk identifier
        rrf_scores: dict[str, float] = {}
        chunk_meta: dict[str, dict] = {}

        for rank, chunk in enumerate(bm25_results, 1):
            cid = _chunk_id(chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank, self.rrf_k)
            if cid not in chunk_meta:
                chunk_meta[cid] = chunk

        for rank, chunk in enumerate(dense_results, 1):
            cid = _chunk_id(chunk)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + _rrf_score(rank, self.rrf_k)
            if cid not in chunk_meta:
                chunk_meta[cid] = chunk

        sorted_ids = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)[:top_k]

        results = []
        for cid in sorted_ids:
            chunk = {**chunk_meta[cid], 'rrf_score': round(rrf_scores[cid], 6)}
            results.append(chunk)

        logger.debug(
            "HybridRetriever: query='%s...' → %d candidates → %d returned",
            query[:60],
            len(rrf_scores),
            len(results),
        )
        return results


def _chunk_id(chunk: dict) -> str:
    """Stable string key for a chunk. Uses doc_id + chunk_index."""
    doc_id = chunk.get('doc_id', 'unknown')
    chunk_index = chunk.get('chunk_index', 0)
    return f"{doc_id}__{chunk_index}"
