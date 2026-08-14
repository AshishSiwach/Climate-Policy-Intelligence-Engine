"""
Retrieval package — lazy imports for heavy ML deps.

BM25Retriever + HybridRetriever are cheap to import (rank-bm25, pure Python).
DenseRetriever + Reranker pull in sentence_transformers + torch, which is
slow (~2s) and can crash import on some Windows setups. Lazy-loaded via
PEP 562 __getattr__ so:

  from retrieval import BM25Retriever            # cheap
  from retrieval import DenseRetriever           # triggers heavy import here

Tests that don't need the dense stack (test_ingestion, test_synthesis,
test_main) never pay the cost.
"""

from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever

__all__ = ["BM25Retriever", "DenseRetriever", "HybridRetriever", "Reranker"]


def __getattr__(name: str):
    if name == "DenseRetriever":
        from retrieval.dense_retriever import DenseRetriever

        return DenseRetriever
    if name == "Reranker":
        from retrieval.reranker import Reranker

        return Reranker
    raise AttributeError(f"module 'retrieval' has no attribute {name!r}")
