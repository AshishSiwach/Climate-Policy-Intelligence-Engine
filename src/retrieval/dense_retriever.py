"""
Dense retriever using BAAI/bge-base-en-v1.5 + Chroma vector store.

Locked decisions (Week 2 audit):
  - Model:   BAAI/bge-base-en-v1.5  (beat all-MiniLM-L6-v2 on top-5 relevance)
  - Device:  cuda                    (RTX 4050 confirmed)
  - Store:   Chroma (cosine space)
  - Top-k:   20 candidates for reranker
"""

import logging
import time
from pathlib import Path

import torch
import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = 'BAAI/bge-base-en-v1.5'
_CHROMA_BATCH = 500   # Chroma upsert limit per call


def _default_device() -> str:
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _safe_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Chroma IDs must be non-empty strings with no special chars."""
    safe = doc_id.replace('/', '_').replace(' ', '_')
    return f"{safe}__{chunk_index}"


def _metadata_safe(chunk: dict) -> dict:
    """Chroma metadata values must be str | int | float | bool."""
    return {
        k: v for k, v in chunk.items()
        if k != 'text' and isinstance(v, (str, int, float, bool))
    }


class DenseRetriever:
    """Dense retriever backed by BAAI/bge-base-en-v1.5 and Chroma.

    Parameters
    ----------
    persist_dir : str | Path
        Directory where Chroma persists its data.
    collection_name : str
        Chroma collection name. Use different names for test vs production.
    device : str
        Torch device. Defaults to 'cuda' if available, else 'cpu'.
    """

    def __init__(
        self,
        persist_dir: str | Path = 'data/processed/chroma_db',
        collection_name: str = 'cpie_v1',
        device: str | None = None,
    ) -> None:
        self.persist_dir = str(persist_dir)
        self.collection_name = collection_name
        self.device = device or _default_device()
        self._model: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    # ------------------------------------------------------------------
    # Lazy-loaded internals
    # ------------------------------------------------------------------

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s on %s", MODEL_NAME, self.device)
            self._model = SentenceTransformer(MODEL_NAME, device=self.device)
        return self._model

    def _get_collection(self):
        if self._collection is None:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={'hnsw:space': 'cosine'},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Index lifecycle
    # ------------------------------------------------------------------

    def build(self, chunks: list[dict], batch_size: int = 64) -> None:
        """Encode chunks and upsert into Chroma.

        Safe to call multiple times — Chroma upserts by ID so re-indexing
        the same doc_id/chunk_index pair overwrites rather than duplicates.
        """
        if not chunks:
            raise ValueError("Cannot build index from an empty chunk list.")

        model = self._get_model()
        collection = self._get_collection()

        texts = [c['text'] for c in chunks]
        logger.info("Encoding %d chunks with %s...", len(chunks), MODEL_NAME)
        t0 = time.time()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        logger.info("Encoding complete: %.1fs", time.time() - t0)

        for start in range(0, len(chunks), _CHROMA_BATCH):
            batch_chunks = chunks[start: start + _CHROMA_BATCH]
            batch_embs = embeddings[start: start + _CHROMA_BATCH]
            collection.upsert(
                ids=[_safe_chunk_id(c['doc_id'], c['chunk_index']) for c in batch_chunks],
                embeddings=batch_embs.tolist(),
                documents=[c['text'] for c in batch_chunks],
                metadatas=[_metadata_safe(c) for c in batch_chunks],
            )

        logger.info(
            "Indexed %d chunks into Chroma collection '%s'",
            len(chunks),
            self.collection_name,
        )

    def query(self, text: str, top_k: int = 20) -> list[dict]:
        """Encode query and return top-k chunks by cosine similarity.

        Each returned dict contains all stored metadata plus:
          - ``dense_score`` (float, cosine similarity 0–1)
          - ``dense_rank``  (int, 1-indexed)
        """
        model = self._get_model()
        collection = self._get_collection()

        n_items = collection.count()
        if n_items == 0:
            raise RuntimeError(
                f"Chroma collection '{self.collection_name}' is empty. Call build() first."
            )
        k = min(top_k, n_items)

        embedding = model.encode([text], convert_to_numpy=True)[0]
        results = collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=k,
            include=['documents', 'metadatas', 'distances'],
        )

        output = []
        for doc, meta, dist in zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0],
        ):
            output.append({
                'text': doc,
                **meta,
                'dense_score': float(1.0 - dist),   # cosine distance → similarity
                'dense_rank': len(output) + 1,
            })
        return output

    def delete_collection(self) -> None:
        """Drop the Chroma collection. Useful for test teardown."""
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("Deleted Chroma collection '%s'", self.collection_name)
