# ZaDeep Fashion 🔍

> Visual similarity search for clothing — powered by [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) and FAISS.

Upload a clothing photo (or type a description) and ZaDeep Fashion returns the most visually similar items from your product catalog. Think Google Lens, but purpose-built for fashion.

---

## How it works

```
Query image / text
       │
       ▼
Marqo-FashionCLIP encoder   ← fine-tuned ViT-B/16, trained on 1M+ fashion products
       │
       ▼  512-dim L2-normalised embedding
       │
       ▼
FAISS flat inner-product index   ← cosine similarity at scale
       │
       ▼
Top-K similar catalog items
```

Both **image-to-image** and **text-to-image** (cross-modal) search are supported using the same shared embedding space.

---

## Project structure

```
zadeep-fashion/
├── src/
│   ├── embedder/
│   │   └── fashion_embedder.py   # wraps Marqo-FashionCLIP
│   ├── indexer/
│   │   └── catalog_indexer.py    # builds/saves/queries FAISS index
│   ├── search/
│   │   └── search_engine.py      # high-level API combining embedder + indexer
│   └── utils/
│       └── dataset_utils.py      # catalog scanning and CSV loading
├── app/
│   ├── app.py                    # Streamlit entry point
│   └── components.py             # UI components
├── scripts/
│   └── build_index.py            # one-time embedding + indexing script
├── colab_quickstart.ipynb        # end-to-end Colab notebook
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/zadeep-fashion.git
cd zadeep-fashion
pip install -r requirements.txt
```

### 2. Get a dataset

The easiest option is the [Kaggle Fashion Product Images](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small) dataset (~600MB, 44K products with metadata CSV).

```bash
kaggle datasets download -d paramaggarwal/fashion-product-images-small -p data/ --unzip
```

Or point `build_index.py` at any folder of product images.

### 3. Build the index

```bash
python scripts/build_index.py \
    --csv data/fashion-dataset/styles.csv \
    --images data/fashion-dataset/images \
    --output models/index \
    --batch_size 64
```

This embeds every image with Marqo-FashionCLIP and saves a FAISS index to `models/index/`. On a T4 GPU it takes ~15 minutes for 44K images.

Use `--max_items 5000` for a quick smoke test.

### 4. Run the app

```bash
streamlit run app/app.py
```

---

## Usage as a library

```python
from src.search.search_engine import FashionSearchEngine

engine = FashionSearchEngine()
engine.load_index("models/index")

# image-to-image
results = engine.search_by_image("query.jpg", top_k=8)

# text-to-image (cross-modal)
results = engine.search_by_text("navy blue slim-fit blazer", top_k=8)

for r in results:
    print(r["rank"], r["score"], r["category"], r["image_path"])
```

---

## Run on Google Colab

Open `colab_quickstart.ipynb` — it walks through dataset download, indexing, search, and launching the Streamlit app via ngrok in ~30 minutes on a free T4 GPU.

---

## Tech stack

| Component | Library |
|---|---|
| Visual embeddings | [Marqo-FashionCLIP](https://huggingface.co/Marqo/marqo-fashionCLIP) (ViT-B/16) |
| Vector index | [FAISS](https://github.com/facebookresearch/faiss) — flat inner-product |
| Deep learning | PyTorch + HuggingFace Transformers |
| UI | Streamlit |
| Dataset | Kaggle Fashion Product Images / DeepFashion2 |

---

## Extending the project

- **Add garment detection**: pipe images through YOLOv8 + Fashionpedia before embedding to support multi-item outfit photos.
- **Swap the vector DB**: replace FAISS with [Qdrant](https://qdrant.tech/) or [Pinecone](https://www.pinecone.io/) for a managed cloud index.
- **Fine-tune the embedder**: use triplet loss on your own catalog to push retrieval accuracy higher.
- **Add metadata filtering**: filter FAISS results by category, gender, or colour before returning them.

---

## License

MIT
