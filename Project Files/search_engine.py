"""
FashionSearchEngine
-------------------
High-level API that combines FashionEmbedder and CatalogIndexer
into a single interface. This is what the Streamlit app talks to.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from src.embedder.fashion_embedder import FashionEmbedder
from src.indexer.catalog_indexer import CatalogIndexer

logger = logging.getLogger(__name__)


class FashionSearchEngine:
    """
    Unified interface for fashion visual search.

    Usage
    -----
    engine = FashionSearchEngine()
    engine.load_index("models/index")
    results = engine.search_by_image("query.jpg", top_k=8)
    results = engine.search_by_text("red floral summer dress", top_k=8)
    """

    def __init__(
        self,
        model_id: str = "Marqo/marqo-fashionCLIP",
        device: str | None = None,
    ) -> None:
        self.embedder = FashionEmbedder(model_id=model_id, device=device)
        self.indexer: CatalogIndexer | None = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def load_index(self, directory: str | Path, use_gpu: bool = False) -> None:
        """Load a pre-built FAISS catalog index from disk."""
        self.indexer = CatalogIndexer.load(directory, use_gpu=use_gpu)
        logger.info("Search engine ready. Catalog size: %d", self.indexer.total_items)

    def get_index(self) -> CatalogIndexer:
        if self.indexer is None:
            raise RuntimeError(
                "No index loaded. Call load_index() or build an index with scripts/build_index.py."
            )
        return self.indexer

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search_by_image(
        self,
        image: Union[str, Path, Image.Image],
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Find catalog items visually similar to a query image.

        Parameters
        ----------
        image : file path or PIL Image
        top_k : int  number of results to return

        Returns
        -------
        list of result dicts (rank, score, product_id, image_path, category, name)
        """
        embedding = self.embedder.embed_single_image(image)
        return self.get_index().search(embedding, top_k=top_k)

    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Find catalog items matching a text description (cross-modal search).

        Parameters
        ----------
        query : str  e.g. "navy blue slim-fit blazer"
        top_k : int

        Returns
        -------
        list of result dicts
        """
        embedding = self.embedder.embed_text([query])[0]
        return self.get_index().search(embedding, top_k=top_k)

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search directly from a pre-computed embedding vector."""
        return self.get_index().search(embedding, top_k=top_k)

    @property
    def catalog_size(self) -> int:
        return self.get_index().total_items
