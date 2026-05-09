"""
CatalogIndexer
--------------
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProductMeta:
    """Lightweight metadata stored alongside each indexed embedding"""

    image_path: str
    product_id: str
    category: str = ""
    name: str = ""
    extra: Dict = field(default_factory=dict)


class CatalogIndexer:
    """
    Manages a FAISS flat inner-product index over L2-normalised embeddings
    (Inner product on unit vectors == cosine similarity)
    """

    def __init__(self, embedding_dim: int = 512, use_gpu: bool = False) -> None:
        self.embedding_dim = embedding_dim
        self.use_gpu = use_gpu
        self._index: Optional[faiss.Index] = None
        self._metadata: List[ProductMeta] = []

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(
        self,
        embeddings: np.ndarray,
        metadata: List[ProductMeta],
    ) -> None:
        """
        Build the index from a pre-computed embedding matrix

        Parameters
        ----------
        embeddings : np.ndarray, shape (N, embedding_dim), float32
        metadata   : list of ProductMeta, length N
        """
        if embeddings.shape[0] != len(metadata):
            raise ValueError(
                f"embeddings ({embeddings.shape[0]}) and metadata ({len(metadata)}) "
                "must have the same length."
            )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)

        # Flat IP index — exact search, no quantisation loss
        index = faiss.IndexFlatIP(self.embedding_dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index moved to GPU.")

        index.add(embeddings)
        self._index = index
        self._metadata = metadata
        logger.info("Index built with %d items.", index.ntotal)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Save index + metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        cpu_index = (
            faiss.index_gpu_to_cpu(self._index)
            if self.use_gpu
            else self._index
        )
        faiss.write_index(cpu_index, str(directory / "catalog.index"))

        with open(directory / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)

        with open(directory / "index_info.json", "w") as f:
            json.dump(
                {"embedding_dim": self.embedding_dim, "total_items": len(self._metadata)},
                f,
                indent=2,
            )

        logger.info("Index saved to %s (%d items).", directory, len(self._metadata))

    @classmethod
    def load(cls, directory: str | Path, use_gpu: bool = False) -> "CatalogIndexer":
        """Load a previously saved index from disk."""
        directory = Path(directory)

        with open(directory / "index_info.json") as f:
            info = json.load(f)

        instance = cls(embedding_dim=info["embedding_dim"], use_gpu=use_gpu)
        instance._index = faiss.read_index(str(directory / "catalog.index"))

        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance._index = faiss.index_cpu_to_gpu(res, 0, instance._index)

        with open(directory / "metadata.pkl", "rb") as f:
            instance._metadata = pickle.load(f)

        logger.info(
            "Index loaded from %s (%d items).", directory, instance._index.ntotal
        )
        return instance

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Dict]:
        """
        Return the top-k most similar catalog items for a query embedding.

        Parameters
        ----------
        query_embedding : np.ndarray, shape (embedding_dim,) or (1, embedding_dim)
        top_k           : int

        Returns
        -------
        List of dicts with keys: rank, score, product_id, image_path, category, name
        """
        if self._index is None:
            raise RuntimeError("Index not built. Call build() or load() first.")

        q = np.ascontiguousarray(
            query_embedding.reshape(1, -1), dtype=np.float32
        )
        scores, indices = self._index.search(q, top_k)

        results = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                {
                    "rank": rank + 1,
                    "score": float(score),
                    "product_id": meta.product_id,
                    "image_path": meta.image_path,
                    "category": meta.category,
                    "name": meta.name,
                }
            )
        return results

    @property
    def total_items(self) -> int:
        return len(self._metadata)
