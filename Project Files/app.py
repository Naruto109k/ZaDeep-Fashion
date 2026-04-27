"""
ZaDeep Fashion — Streamlit Demo
=============================
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.components import render_result_grid, render_sidebar, render_upload_zone
from src.search.search_engine import FashionSearchEngine

# ── page config 
st.set_page_config(
    page_title="ZaDeep Fashion",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS 
st.markdown(
    """
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; padding-bottom: 2rem;}
    .stImage > img {border-radius: 10px;}
    .result-card {
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 8px;
        text-align: center;
        background: #fafafa;
    }
    .score-badge {
        background: #0f6e56;
        color: white;
        border-radius: 8px;
        padding: 2px 8px;
        font-size: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

INDEX_DIR = Path("models/index")


# ── model loading (cached across reruns) 
@st.cache_resource(show_spinner="Loading ZaDeep Fashion model…")
def load_engine() -> FashionSearchEngine:
    engine = FashionSearchEngine()
    if INDEX_DIR.exists():
        engine.load_index(INDEX_DIR)
    return engine


# ── sidebar
top_k, search_mode = render_sidebar()

# ── main layout 
st.markdown("## 🔍 ZaDeep Fashion")
st.markdown(
    "Upload a clothing image **or** describe what you're looking for — "
    "ZaDeep Fashion finds the most visually similar items in the catalog."
)
st.divider()

engine = load_engine()

if not INDEX_DIR.exists():
    st.warning(
        "No catalog index found at `models/index/`. "
        "Run `python scripts/build_index.py` first to index your product images.",
        icon="⚠️",
    )
    st.stop()

# ── input zone
query_image, query_text = render_upload_zone(search_mode)

if st.button("Search", type="primary", use_container_width=True):
    if search_mode == "Image" and query_image is not None:
        with st.spinner("Finding similar items…"):
            results = engine.search_by_image(query_image, top_k=top_k)
        st.success(f"Top {len(results)} matches from {engine.catalog_size:,} products")
        render_result_grid(results, columns=4)

    elif search_mode == "Text" and query_text:
        with st.spinner("Searching catalog…"):
            results = engine.search_by_text(query_text, top_k=top_k)
        st.success(f"Top {len(results)} matches for: *{query_text}*")
        render_result_grid(results, columns=4)

    else:
        st.info("Please provide an image or a text query to search.")
