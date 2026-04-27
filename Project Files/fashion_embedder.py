"""
FashionEmbedder
---------------
Wraps Marqo-FashionCLIP to produce L2-normalised image and text embeddings
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

logger = logging.getLogger(__name__)

MODEL_ID = "Marqo/marqo-fashionCLIP"


class FashionEmbedder:
    """
    Produces dense visual embeddings for clothing images using
    Marqo-FashionCLIP, a ViT-B/16 model fine-tuned on 1M+ fashion products
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = str(cache_dir) if cache_dir else None

        logger.info("Loading %s on %s …", model_id, self.device)
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
        ).to(self.device)
        self.model.eval()
        logger.info("Model ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_images(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Embed a list of images
        """
        pil_images = [self._load_image(img) for img in images]
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(pil_images), batch_size):
            batch = pil_images[start : start + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            all_embeddings.append(self._normalise(features).cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_text(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of text queries (for cross-modal search)
        """
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self.processor(text=batch, return_tensors="pt", padding=True).to(
                self.device
            )
            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
            all_embeddings.append(self._normalise(features).cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_single_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """Convenience wrapper for a single query image."""
        return self.embed_images([image])[0]

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _load_image(source: Union[str, Path, Image.Image]) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB")
        return Image.open(source).convert("RGB")

    @staticmethod
    def _normalise(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(tensor, p=2, dim=-1)
