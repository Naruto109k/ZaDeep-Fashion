"""
UI components for the ZaDeep Fashion Streamlit app
Each function is self-contained and returns data back to app.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image


def render_sidebar() -> Tuple[int, str]:
    """Render sidebar controls. Returns (top_k, search_mode)"""
    with st.sidebar:
        st.markdown("### ⚙️ Search settings")

        search_mode = st.radio(
            "Search mode",
            options=["Image", "Text"],
            index=0,
            help="Image: upload a clothing photo. Text: describe what you want.",
        )

        top_k = st.slider(
            "Results to show",
            min_value=4,
            max_value=20,
            value=8,
            step=4,
        )

        st.divider()
        st.markdown("### ℹ️ About")
        st.markdown(
            "**ZaDeep Fashion** uses [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) "
            "embeddings and a FAISS vector index to find visually similar clothing items. "
            "Both image-to-image and text-to-image search are supported."
        )

    return top_k, search_mode


def render_upload_zone(
    search_mode: str,
) -> Tuple[Optional[Image.Image], Optional[str]]:
    """
    Render the query input area
    Returns (pil_image, text_query) — one will be None depending on mode
    """
    query_image: Optional[Image.Image] = None
    query_text: Optional[str] = None

    col_input, col_preview = st.columns([1, 1], gap="large")

    with col_input:
        if search_mode == "Image":
            uploaded = st.file_uploader(
                "Upload a clothing image",
                type=["jpg", "jpeg", "png", "webp"],
                label_visibility="collapsed",
            )
            if uploaded:
                query_image = Image.open(uploaded).convert("RGB")
        else:
            query_text = st.text_input(
                "Describe the item you're looking for",
                placeholder="e.g. red floral midi dress with puff sleeves",
                label_visibility="collapsed",
            )

    with col_preview:
        if query_image is not None:
            st.image(query_image, caption="Your query image", use_container_width=True)

    return query_image, query_text


def render_result_grid(results: List[Dict], columns: int = 4) -> None:
    """Render search results in a responsive grid with similarity scores."""
    if not results:
        st.info("No results found.")
        return

    rows = [results[i : i + columns] for i in range(0, len(results), columns)]

    for row in rows:
        cols = st.columns(columns, gap="small")
        for col, result in zip(cols, row):
            with col:
                img_path = Path(result["image_path"])
                if img_path.exists():
                    img = Image.open(img_path).convert("RGB")
                    st.image(img, use_container_width=True)
                else:
                    st.markdown("*(image not found)*")

                score_pct = round(result["score"] * 100, 1)
                category = result.get("category", "")
                name = result.get("name", f"Product {result['product_id']}")

                st.markdown(
                    f"<div style='text-align:center; font-size:13px;'>"
                    f"<b>{name[:28]}</b><br>"
                    f"<span style='color:#888; font-size:11px;'>{category}</span><br>"
                    f"<span class='score-badge'>{score_pct}% match</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
