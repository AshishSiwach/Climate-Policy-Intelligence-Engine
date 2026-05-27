"""
BM25 sparse retriever.

Wraps rank-bm25 with a consistent chunk-dict interface.
Serialisable to disk so the index survives between sessions.
"""

import logging
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25 sparse retriever over a chunked document corpus.

    Parameters
    ----------
    k1 : float
        BM25 term-frequency saturation (default 1.5, locked in config.yaml).
    b : float
        BM25 length normalisation (default 0.75, locked in config.yaml).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._index: BM25Okapi | None = None
        self._chunks: list[dict] = []

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, strip punctuation, split on whitespace."""
        return re.sub(r'[^\w\s]', ' ', text.lower()).split()

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict]) -> None:
        """Build BM25 index from a list of chunk dicts.

        Each chunk dict must have at minimum a ``'text'`` key.
        All other keys are carried through as metadata on retrieval.
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")
        self._chunks = list(chunks)
        tokenized = [self._tokenize(c['text']) for c in self._chunks]
        self._index = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        logger.info("BM25 index built: %d chunks", len(self._chunks))

    def query(self, text: str, top_k: int = 20) -> list[dict]:
        """Return top-k chunks by BM25 score, descending.

        Chunks with score=0 (no term overlap) are excluded.
        Each returned dict is the original chunk dict extended with:
          - ``bm25_score`` (float)
          - ``bm25_rank``  (int, 1-indexed)
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() first.")
        scores = self._index.get_scores(self._tokenize(text))
        top_idx = scores.argsort()[::-1][:top_k]
        results = []
        for rank, i in enumerate(top_idx, 1):
            if float(scores[i]) <= 0:
                break
            results.append({
                **self._chunks[i],
                'bm25_score': float(scores[i]),
                'bm25_rank': rank,
            })
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the index and chunk list to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(
                {'index': self._index, 'chunks': self._chunks, 'k1': self.k1, 'b': self.b},
                f,
            )
        logger.info("BM25 index saved to %s", path)

    @classmethod
    def load(cls, path: str | Path) -> 'BM25Retriever':
        """Load a previously saved BM25 index from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        obj = cls(k1=data['k1'], b=data['b'])
        obj._index = data['index']
        obj._chunks = data['chunks']
        logger.info("BM25 index loaded from %s: %d chunks", path, len(obj._chunks))
        return obj
