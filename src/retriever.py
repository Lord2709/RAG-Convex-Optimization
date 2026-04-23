"""
retriever.py
------------
Two retrieval backends:
  - BM25Retriever   : sparse lexical retrieval via rank-bm25
  - DenseRetriever  : dense semantic retrieval via sentence-transformers + FAISS

Both share the same interface so they are interchangeable and easy to swap.
"""

import logging
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
class BM25Retriever:
    """Sparse BM25 retriever backed by rank-bm25 (BM25Okapi variant)."""

    def __init__(self, config: Dict[str, Any]):
        self.top_k = config["retrieval"]["top_k"]
        self._bm25: BM25Okapi | None = None
        self._corpus: List[Dict] | None = None

    # ------------------------------------------------------------------
    def index(self, corpus: List[Dict]) -> None:
        self._corpus = corpus
        tokenized = [doc["text"].lower().split() for doc in corpus]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built  docs=%d", len(corpus))

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, score) pairs sorted by descending BM25 score."""
        k = top_k or self.top_k
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)
        top_idx = np.argsort(scores)[::-1][:k]
        return [(self._corpus[i]["doc_id"], float(scores[i])) for i in top_idx]

    # ------------------------------------------------------------------
    def scores_for_ids(self, query: str, doc_ids: List[str]) -> Dict[str, float]:
        """Return BM25 scores for a specific list of doc_ids."""
        id_to_idx = {doc["doc_id"]: i for i, doc in enumerate(self._corpus)}
        raw = self._bm25.get_scores(query.lower().split())
        return {
            did: float(raw[id_to_idx[did]])
            for did in doc_ids
            if did in id_to_idx
        }


# ---------------------------------------------------------------------------
class DenseRetriever:
    """Dense retriever: sentence-transformers embeddings + FAISS inner-product index."""

    def __init__(self, config: Dict[str, Any]):
        self.model_name = config["retrieval"]["embedding_model"]
        self.top_k = config["retrieval"]["top_k"]
        self._model = SentenceTransformer(self.model_name)
        self._corpus: List[Dict] | None = None
        self._doc_embeddings: np.ndarray | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None
        self._id_to_idx: Dict[str, int] = {}

    # ------------------------------------------------------------------
    def index(self, corpus: List[Dict]) -> None:
        self._corpus = corpus
        self._id_to_idx = {doc["doc_id"]: i for i, doc in enumerate(corpus)}

        texts = [doc["text"] for doc in corpus]
        logger.info(
            "Encoding %d documents with %s …", len(texts), self.model_name
        )
        self._doc_embeddings = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,  # unit vectors → cosine via dot product
        ).astype(np.float32)

        dim = self._doc_embeddings.shape[1]
        self._faiss_index = faiss.IndexFlatIP(dim)
        self._faiss_index.add(self._doc_embeddings)
        logger.info("FAISS index built  docs=%d  dim=%d", len(corpus), dim)

    # ------------------------------------------------------------------
    def encode_query(self, query: str) -> np.ndarray:
        return self._model.encode(
            [query], normalize_embeddings=True
        )[0].astype(np.float32)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int | None = None) -> List[Tuple[str, float]]:
        """Return top-k (doc_id, score) pairs sorted by descending cosine similarity."""
        k = top_k or self.top_k
        q_emb = self.encode_query(query).reshape(1, -1)
        scores, indices = self._faiss_index.search(q_emb, k)
        return [
            (self._corpus[idx]["doc_id"], float(scores[0][i]))
            for i, idx in enumerate(indices[0])
        ]

    # ------------------------------------------------------------------
    def get_embedding(self, doc_id: str) -> np.ndarray:
        """Return the stored embedding for a doc_id."""
        return self._doc_embeddings[self._id_to_idx[doc_id]]
