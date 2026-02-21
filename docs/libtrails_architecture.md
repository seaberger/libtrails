# LibTrails: Building a Topic Discovery Engine for 100 Classic Books

LibTrails is a tool for discovering conceptual connections across a book collection. It parses EPUBs into text chunks, uses LLMs to extract granular topics from each chunk, generates semantic embeddings, and builds a multi-tier topic hierarchy — all stored locally in SQLite. The result is a searchable, browsable universe of ideas that connects Homer's *Iliad* to Dostoevsky's *Crime and Punishment* through shared concepts like guilt, duty, and divine justice.

This post covers the architecture behind the demo at [libtrails.app](https://libtrails.app), running on a 100-book Project Gutenberg collection on a $3.50/month Lightsail t3.micro instance with 1 GB RAM.

---

## The Pipeline

### Stage 1: Parse and Chunk

EPUBs are parsed with [selectolax](https://github.com/rushter/selectolax) (HTML block tags converted to paragraph breaks) and recursively split into ~500-word chunks — first at paragraph boundaries, then sentences, then words as a last resort. The 100-book demo library produces **31,849 chunks**.

### Stage 2: Two-Pass LLM Topic Extraction

**Pass 1 — Book Themes (gemma3:27b):** A larger model reads the first few chunks plus Calibre metadata (tags, description, series) to extract 5–10 book-level themes in a single call. These themes anchor the next pass.

**Pass 2 — Chunk Topics (gemma3:12b):** A smaller model extracts chunk-level topics, contextualized with the book themes from Pass 1. This produces domain-specific noun phrases instead of generic single words. The phrase "energy manipulation" from a fantasy novel stays distinct from "energy manipulation" in a psychology text because each extraction is grounded in the book's theme context.

The two models run on separate machines: the 27b model on a MacBook Pro via LM Studio (MLX backend), the 12b model on a remote Windows PC with an RTX 3090 via LM Studio. This parallelizes extraction — while the 3090 grinds through chunks, the Mac processes the next book's themes.

Result: **121,118 deduplicated topics** across 100 books.

### Stage 3: Embed and Deduplicate

All topics get 384-dimensional embeddings via [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5). A two-tier deduplication then merges near-duplicates:

- **Cosine > 0.95**: merge unconditionally (clearly the same concept)
- **Cosine 0.85–0.95**: merge only if both topics share at least one book — preventing cross-domain conflation

This two-tier approach matters. Without the book-overlap guard, "energy manipulation" from a fantasy novel merges with the same phrase from a psychology text, losing the distinction that makes topic extraction useful.

### Stage 4: Graph Construction

A KNN graph is built from two types of edges:

1. **Co-occurrence edges:** Topic pairs that appear in the same chunk get Pointwise Mutual Information (PMI) scores, weighted with a book-count boost: `PMI × (1 + log(1 + book_count))`. Only genuinely surprising co-occurrences make the cut (PMI ≥ 1.0).

2. **Embedding-similarity edges:** Each topic's 10 nearest neighbors by cosine similarity (≥ 0.65) form additional edges.

The KNN computation uses [FAISS-GPU](https://github.com/facebookresearch/faiss) on the RTX 3090 — 30 seconds for 837K vectors vs. 115 minutes on the MacBook CPU. Results are cached as `.npz` files so clustering reruns don't need GPU access.

### Stage 5: Three-Tier Leiden Clustering

**Hub removal:** High-degree topics (95th percentile) are removed before clustering. These generic terms (e.g., "conflict," "identity") would create artificially large, incoherent groups.

**First Leiden pass — Topic Clusters (2,468):** The [Leiden algorithm](https://www.nature.com/articles/s41598-019-41695-z) runs with the CPM (Constant Potts Model) partition type at resolution 0.001, producing fine-grained clusters. Each cluster groups tightly related topics — "categorical imperative," "moral duty," and "Kantian ethics" all land in one cluster. The cluster label comes from its most representative topic: "Kantian Ethics & Moral Philosophy."

**Second Leiden pass — Topic Communities (202):** The same graph with a coarser resolution groups clusters into broader communities — the browsable entries on the Topics page and the spheres in the Universe view. Each community is a coherent theme like "Victorian Social Norms" or "Ancient Greek Philosophy."

**K-Means — Themes (26):** K-means groups cluster centroids into 26 high-level themes based on embedding similarity: "Adventure & Human Folly," "Literary Classics & Philosophy," "Nature & Rural Life," etc. Labels are LLM-generated and human-refined.

### Stage 6: Universe Visualization

Community centroids are projected into 3D with [UMAP](https://umap-learn.readthedocs.io/) (cosine metric, n_neighbors=15). Theme embeddings are mapped onto a single PCA axis, then each theme's position becomes a hue value — so semantically similar themes share similar colors. The result is rendered with [React Three Fiber](https://docs.pmnd.rs/react-three-fiber) using instanced meshes for performance.

---

## Lightweight Embeddings with ONNX Runtime

The demo runs on a Lightsail t3.micro with 1 GB RAM. The naive approach — loading [sentence-transformers](https://sbert.net/) with PyTorch — pulls in ~300 MB of dependencies just for the embedding model. On a 1 GB instance, that's 30% of RAM burned before serving a single request, and the process regularly swaps to disk.

### The Problem

`sentence-transformers` depends on PyTorch, which is a 200+ MB library designed for GPU training. We don't need any of that for inference — we just need to run a single forward pass through a small transformer model to produce a 384-dim vector.

### The Solution: ONNX Runtime

[ONNX Runtime](https://onnxruntime.ai/) is Microsoft's optimized inference engine. It loads a pre-exported `.onnx` model file and runs inference with minimal dependencies — no PyTorch, no CUDA toolkit, no autograd. The relevant dependencies are:

- `onnxruntime` (~15 MB) — the inference engine
- `tokenizers` (~6 MB, from HuggingFace) — fast Rust-based tokenizer

Total: ~20 MB vs. ~300 MB for sentence-transformers + PyTorch.

### Exporting the Model

A one-time export converts the sentence-transformers model to ONNX format using HuggingFace's [optimum](https://huggingface.co/docs/optimum/) library:

```python
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

model = ORTModelForFeatureExtraction.from_pretrained(model_path, export=True)
model.save_pretrained(onnx_output_dir)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.save_pretrained(onnx_output_dir)
```

The exported model is ~33 MB. It gets SCP'd to the server alongside the tokenizer files.

### Dual Backend Architecture

The embedding module (`embeddings.py`) auto-detects which backend to use:

1. Check if `models/bge-small-onnx/model.onnx` exists and `onnxruntime` is installed → use ONNX
2. Otherwise → fall back to sentence-transformers

Both backends produce **identical embeddings** (verified: cosine similarity = 1.000000 between outputs). The ONNX path uses CLS pooling with L2 normalization, matching BGE's default strategy:

```python
def _onnx_encode(texts: list[str]) -> np.ndarray:
    encodings = _onnx_tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    outputs = _onnx_session.run(None, {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })

    # CLS pooling (index 0)
    cls_embeddings = outputs[0][:, 0, :]

    # L2 normalize
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    return cls_embeddings / np.maximum(norms, 1e-12)
```

The `get_model()` function still loads sentence-transformers for bulk operations (indexing on a dev machine), without overriding the active backend — so `embed_text()` and `embed_texts()` continue using ONNX for server requests even if sentence-transformers gets loaded for a batch job.

### Impact

| Metric | sentence-transformers | ONNX Runtime |
|--------|----------------------|--------------|
| Process RSS | ~500 MB | ~200 MB |
| Model load time | ~3s | ~0.5s |
| Embedding latency | ~15ms/query | ~12ms/query |
| PyTorch required | Yes (~200 MB) | No |
| Swap pressure (1 GB RAM) | Frequent | None |

The server now starts in under a second and stays well within the 1 GB memory budget with room for SQLite page cache, ONNX inference, and the Astro SSR process.

---

## Optimizing Hybrid Search for a $3.50 Server

The initial hybrid search implementation was designed on a MacBook Pro where every query ran in ~200ms. Deployed to the Lightsail t3.micro, things looked different.

### The Original Design: 7-Signal Book Search

Book search originally fused seven independent retrieval signals via [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (RRF):

1. **FTS5 book metadata** — keyword search over title, author, description
2. **FTS5 topic labels** — keyword search across 121K topic labels, mapped back to books
3. **FTS5 chunk text** — keyword search across 31,849 chunk passages, mapped back to books
4. **Semantic topic vectors** — cosine similarity against 121K topic embeddings via sqlite-vec
5. **Semantic book theme vectors** — cosine similarity against book-level theme embeddings
6. **Semantic book vectors** — cosine similarity against whole-book embeddings
7. **Semantic chunk vectors** — cosine similarity against 31,849 chunk embeddings, best per book

RRF combines these by scoring each result as `1/(k + rank)` across all lists, where `k=60`. A book that ranks #1 in keyword search and #5 in semantic search scores higher than one that ranks #2 in both — rewarding results that excel in at least one signal.

Cluster search similarly used 5 signals: topic FTS, **semantic topic vectors**, cluster label matching, chunk FTS, and chunk semantics.

### The Bottleneck: 121K Vector Scan

Profiling on the server revealed that **signal 4 (semantic topic vectors) took 2.7 seconds per query**. sqlite-vec performs exact KNN — it scans all 121,000 topic vectors and computes cosine distance against each one. On a single-core t3.micro, that's a lot of floating-point math.

The other vector signals were fast because they searched much smaller tables:

| Table | Rows | Scan time |
|-------|------|-----------|
| `topic_vectors` | 121,118 | **2.7s** |
| `chunk_vectors` | 31,849 | ~0.5s |
| `book_theme_vectors` | ~800 | ~0.01s |
| `book_vectors` | 100 | ~0.005s |

The chunk vectors (31K rows, ~0.5s) already provided equivalent semantic coverage to the topic vectors — every chunk that mentions a concept will have a nearby embedding, and the chunk→book mapping captures the same books that topic→book would. The topic vector signal was redundant given the chunk signal, but 5x more expensive.

### The Fix: Drop to 6 Signals

Removing the topic vector scan from both search paths:

- **Book search**: 7 signals → 6 (drop signal 4)
- **Cluster search**: 5 signals → 4 (drop topic vector signal)

Total search time dropped from ~3.5s to ~0.8s with no measurable difference in result quality — the chunk and theme vectors already cover the same semantic ground.

### Related Books: From 19 Seconds to 50 Milliseconds

The `find_related_books` endpoint (shown on every book detail page) was even worse. The original implementation called `hybrid_search_books` with a query constructed from the book's title, author, and all themes — a 373-character query with 50+ tokens. Every FTS5 query matched nearly everything, and the topic vector scan ran on top of that. Total: **19 seconds per request**.

The rewrite uses only the fast vector signals:

```python
def find_related_books(conn, book_id, limit=12):
    # Build a short embedding query from title + top themes
    query = f"{title} by {author}. {', '.join(themes[:5])}"
    query_bytes = embedding_to_bytes(embed_text(query))

    # 3 fast signals only (~0.05s total)
    book_direct = _semantic_search_books_direct(conn, query, query_bytes=query_bytes)   # 100 rows
    theme_books = _semantic_search_book_themes(conn, query, query_bytes=query_bytes)     # ~800 rows
    fts_books = _fts_search_books(conn, title)                                          # FTS on 100 books

    return rrf_fuse([book_direct, theme_books, fts_books])
```

Three signals over small tables: **50ms total**. The quality is arguably better — the old approach produced noisy results because the over-long FTS query matched too broadly.

### What Was Compromised

The topic vector signal *did* add value for certain queries — particularly abstract concept searches like "existentialism" or "manifest destiny" where the relevant topics might not appear in chunk text verbatim. But the chunk embeddings capture these concepts well enough (the chunks *about* existentialism have nearby embeddings even if the word doesn't appear), and the 5x speedup on every single query is worth the marginal quality loss on edge cases.

If we move to a larger server or add approximate nearest neighbor (ANN) indexing, we could add the signal back. But for a $3.50/month t3.micro, 0.8s search is the right tradeoff.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Topic extraction | gemma3 via Ollama / Gemini / LM Studio |
| Embeddings | BGE-small-en-v1.5 (384 dims) via ONNX Runtime |
| Search | Hybrid: FTS5 (BM25) + sqlite-vec (cosine) + RRF fusion |
| Clustering | Leiden (python-igraph + leidenalg), two-pass |
| 3D projection | UMAP + PCA semantic colors |
| Backend | FastAPI + SQLite |
| Frontend | Astro + React Three Fiber |
| Hosting | AWS Lightsail t3.micro (1 GB RAM, $3.50/month) |
| Reverse proxy | Caddy (auto TLS) |

---

## Numbers

| Metric | Value |
|--------|-------|
| Books | 100 (Project Gutenberg classics, ~800 BC – 1925 AD) |
| Text chunks | 31,849 |
| Extracted topics | 121,118 |
| Leiden clusters | 2,468 |
| Topic communities | 202 |
| Themes | 26 |
| Search signals (books) | 6 |
| Search signals (clusters) | 4 |
| Server memory budget | 1 GB |
| API process RSS | ~200 MB (with ONNX) |
| Search latency | ~0.8s (was ~3.5s) |
| Related books latency | ~0.05s (was ~19s) |

---

## Links

- **Live demo**: [libtrails.app](https://libtrails.app)
- **Source**: [github.com/seaberger/libtrails](https://github.com/seaberger/libtrails)
- **Inspiration**: [Pieter Maes' "Reading Across Books"](https://pieterma.es/syntopic-reading-claude/) and the [Trails visualization](https://trails.pieterma.es)
