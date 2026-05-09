"""
build_index.py
--------------
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# make src importable when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fashion_embedder import FashionEmbedder
from catalog_indexer import CatalogIndexer
from dataset_utils import load_kaggle_fashion_csv, scan_image_directory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ZaDeep Fashion FAISS index.")
    p.add_argument("--csv", type=str, default=None, help="Path to styles.csv (Kaggle dataset)")
    p.add_argument("--images", type=str, required=True, help="Root directory of product images")
    p.add_argument("--output", type=str, default="models/index", help="Where to save the index")
    p.add_argument("--batch_size", type=int, default=64, help="Embedding batch size")
    p.add_argument("--max_items", type=int, default=None, help="Cap catalog size (testing)")
    p.add_argument("--device", type=str, default=None, help="cuda / cpu / None (auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # --- load metadata ---
    if args.csv:
        logger.info("Loading metadata from CSV: %s", args.csv)
        metadata = load_kaggle_fashion_csv(
            csv_path=args.csv,
            image_root=args.images,
            max_items=args.max_items,
        )
    else:
        logger.info("No CSV provided — scanning image directory: %s", args.images)
        metadata = scan_image_directory(args.images)
        if args.max_items:
            metadata = metadata[: args.max_items]

    if not metadata:
        logger.error("No images found. Check your paths.")
        sys.exit(1)

    logger.info("Catalog size: %d items", len(metadata))
    image_paths = [m.image_path for m in metadata]

    # --- embed ---
    embedder = FashionEmbedder(device=args.device)
    logger.info("Embedding %d images (batch_size=%d) …", len(image_paths), args.batch_size)
    t0 = time.time()
    embeddings = embedder.embed_images(image_paths, batch_size=args.batch_size)
    elapsed = time.time() - t0
    logger.info(
        "Embedding done in %.1fs (%.1f img/s). Shape: %s",
        elapsed,
        len(image_paths) / elapsed,
        embeddings.shape,
    )

    # --- build + save index ---
    indexer = CatalogIndexer(embedding_dim=embeddings.shape[1])
    indexer.build(embeddings, metadata)
    indexer.save(args.output)
    logger.info("Done. Index saved to: %s", args.output)


if __name__ == "__main__":
    main()
