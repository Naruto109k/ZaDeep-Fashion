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
import open_clip

logger = logging.getLogger(__name__)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"


class FashionEmbedder:
    """
    Produces dense visual embeddings for clothing images using
    Marqo-FashionCLIP (ViT-B/32), fine-tuned on fashion products.
    """

    def __init__(
        self,
        device: str | None = None,
        model_name: str = MODEL_NAME,
        pretrained: str = PRETRAINED,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model_name = model_name
        self.pretrained = pretrained

        logger.info("Loading OpenCLIP model (%s) on %s …", model_name, self.device)

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
        )

        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        self.model = self.model.to(self.device)
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

            image_tensors = torch.stack(
                [self.preprocess(img) for img in batch]
            ).to(self.device)

            with torch.no_grad():
                features = self.model.encode_image(image_tensors)

            all_embeddings.append(self._normalize(features).cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_text(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Embed a list of text queries (for cross-modal search)
        """
        all_embeddings: List[np.ndarray] = []

        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]

            text_tokens = self.tokenizer(batch).to(self.device)

            with torch.no_grad():
                features = self.model.encode_text(text_tokens)

            all_embeddings.append(self._normalize(features).cpu().numpy())

        return np.vstack(all_embeddings).astype(np.float32)

    def embed_single_image(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
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
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(tensor, p=2, dim=-1)