"""
FashionSearchEngine
-------------------
High-level API that combines FashionEmbedder and CatalogIndexer
into a single interface. This is what the Streamlit app talks to
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from PIL import Image

from fashion_embedder import FashionEmbedder
from catalog_indexer import CatalogIndexer

logger = logging.getLogger(__name__)

# -- Auto-detect index path --------------------------------------------------
def _find_index_dir() -> Path:
    """Search common locations for the FAISS index directory."""
    candidates = [
        Path(__file__).parent / "models" / "index",
        Path(__file__).parent.parent / "models" / "index",
        Path("models") / "index",
        Path("../models/index"),
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.iterdir()):
            logger.info("Found index at: %s", candidate.resolve())
            return candidate.resolve()

    # None found — give a helpful error
    searched = "\n  ".join(str(c.resolve()) for c in candidates)
    raise RuntimeError(
        f"No catalog index found. Searched:\n  {searched}\n\n"
        f"Run this to build it:\n"
        f"  cd 'Project Files'\n"
        f"  python build_index.py --images clothes\n"
    )
# ----------------------------------------------------------------------------


class FashionSearchEngine:
    """
    Unified interface for fashion visual search

    Usage
    -----
    engine = FashionSearchEngine()
    engine.load_index()           # auto-detects index location
    engine.load_index("models/index")  # or provide path explicitly
    results = engine.search_by_image("query.jpg", top_k=8)
    results = engine.search_by_text("red floral summer dress", top_k=8)
    """

    def __init__(
        self,
        model_id: str = "Marqo/marqo-fashionCLIP",
        device: str | None = None,
    ) -> None:
        self.embedder = FashionEmbedder(device=device)
        self.indexer: CatalogIndexer | None = None

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def load_index(self, directory: str | Path | None = None, use_gpu: bool = False) -> None:
        """Load a pre-built FAISS catalog index from disk.
        If no directory is given, auto-detects the index location.
        """
        if directory is None:
            directory = _find_index_dir()
        directory = Path(directory)
        if not directory.exists():
            raise RuntimeError(
                f"Index directory not found: {directory.resolve()}\n\n"
                f"Run this to build it:\n"
                f"  cd 'Project Files'\n"
                f"  python build_index.py --images clothes\n"
            )
        self.indexer = CatalogIndexer.load(directory, use_gpu=use_gpu)
        logger.info("Search engine ready. Catalog size: %d", self.indexer.total_items)

    def get_index(self) -> CatalogIndexer:
        if self.indexer is None:
            raise RuntimeError(
                "No index loaded. Call load_index() first or run:\n"
                "  cd 'Project Files'\n"
                "  python build_index.py --images clothes"
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
        Find catalog items visually similar to a query image
        """
        
        embedding = self.embedder.embed_single_image(image)
        return self.get_index().search(embedding, top_k=top_k)

    def search_by_text(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict]:
        """
        Find catalog items matching a text description (cross-modal search)
        """
        
        embedding = self.embedder.embed_text([query])[0]
        return self.get_index().search(embedding, top_k=top_k)

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10,
    ) -> List[Dict]:
        """Search directly from a pre-computed embedding vector"""
        return self.get_index().search(embedding, top_k=top_k)

    @property
    def catalog_size(self) -> int:
        return self.get_index().total_items