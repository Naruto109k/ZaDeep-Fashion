"""
Dataset utilities
-----------------
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional

from catalog_indexer import ProductMeta

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def scan_image_directory(
    root: str | Path,
    category_from_parent: bool = True,
) -> List[ProductMeta]:
    """
    Recursively scan a directory of product images
    """
    root = Path(root).resolve()
    items: List[ProductMeta] = []

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        category = path.parent.name if category_from_parent else ""
        items.append(
            ProductMeta(
                image_path=str(path),
                product_id=path.stem,
                category=category,
                name=path.stem.replace("_", " ").replace("-", " ").title(),
            )
        )

    logger.info("Scanned %d images from %s.", len(items), root)
    return items


def load_kaggle_fashion_csv(
    csv_path: str | Path,
    image_root: str | Path,
    image_col: str = "id",
    category_col: str = "articleType",
    name_col: str = "productDisplayName",
    max_items: Optional[int] = None,
) -> List[ProductMeta]:
    """
    Load metadata from the Kaggle Fashion Product Images dataset CSV
    """
    csv_path = Path(csv_path)
    image_root = Path(image_root)
    items: List[ProductMeta] = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = image_root / f"{row[image_col]}.jpg"
            if not img_path.exists():
                continue

            items.append(
                ProductMeta(
                    image_path=str(img_path),
                    product_id=row[image_col],
                    category=row.get(category_col, ""),
                    name=row.get(name_col, ""),
                    extra={
                        "gender": row.get("gender", ""),
                        "colour": row.get("baseColour", ""),
                        "season": row.get("season", ""),
                    },
                )
            )

            if max_items and len(items) >= max_items:
                break

    logger.info("Loaded %d items from CSV.", len(items))
    return items