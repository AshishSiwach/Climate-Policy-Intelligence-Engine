"""
Cross-encoder reranker.

Locked decisions (Week 2 audit):
  - Model:          cross-encoder/ms-marco-MiniLM-L-6-v2
  - Lazy loading:   model loaded on first rerank() call, not at import
  - Top-k input:    20 candidates from hybrid retriever
  - Top-k output:   5 chunks returned to synthesiser
  - Latency:        logged separately for monitoring dashboard
"""

import logging
import time

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = 'cross-encoder/ms-marco-MiniLM-L-6-v2'


class Reranker:
    """Cross-encoder reranker.

    Scores each (query, candidate) pair and returns the top-k
    by reranker score. Lazy-loads the model on first call.

    Parameters
    ----------
    top_k : int
        Number of chunks to return after reranking (default 5).
    model_name : str
        HuggingFace model identifier. Locked to ms-marco-MiniLM-L-6-v2.
    """

    def __init__(
        self,
        top_k: int = 5,
        model_name: str = MODEL_NAME,
    ) -> None:
        self.top_k = top_k
        self.model_name = model_name
        self._model: CrossEncoder | None = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _get_model(self) -> CrossEncoder:
        """Load model on first call. Subsequent calls return cached model."""
        if self._model is None:
            logger.info("Loading reranker model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
            logger.info("Reranker loaded.")
        return self._model

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score candidates against query and return top-k.

        Parameters
        ----------
        query : str
            The original user query.
        candidates : list[dict]
            Up to 20 chunk dicts from HybridRetriever.retrieve().

        Returns
        -------
        list[dict]
            Top-k chunk dicts, sorted by rerank_score descending.
            Each dict is extended with:
              - ``rerank_score``      (float)  — cross-encoder logit
              - ``rerank_rank``       (int)    — final rank, 1-indexed
              - ``rerank_latency_ms`` (float)  — latency for this batch
        """
        if not candidates:
            logger.warning("Reranker called with empty candidate list.")
            return []

        model = self._get_model()
        pairs = [(query, c['text']) for c in candidates]

        t0 = time.time()
        scores = model.predict(pairs)
        latency_ms = (time.time() - t0) * 1000

        logger.info(
            "Reranker: scored %d candidates in %.0fms",
            len(candidates),
            latency_ms,
        )

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        return [
            {
                **chunk,
                'rerank_score': float(score),
                'rerank_rank': rank,
                'rerank_latency_ms': round(latency_ms, 1),
            }
            for rank, (score, chunk) in enumerate(ranked[: self.top_k], 1)
        ]
