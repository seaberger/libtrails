# libtrails

Trail-finding across your book library using semantic search and topic clustering.

Inspired by [Pieter Maes' "Reading Across Books" project](https://pieterma.es/syntopic-reading-claude/) and the [Trails visualization](https://trails.pieterma.es).

## What is this?

libtrails helps you discover conceptual connections across your book collection. It extracts topics from your EPUBs using a local LLM, generates semantic embeddings, and builds a hierarchical topic graph—all stored locally in SQLite.

**Use cases:**
- Find books that discuss similar themes
- Discover unexpected connections between books
- Build "trails" of related excerpts across your library
- Explore your reading interests through topic clusters

## How it works

```
EPUB → Chunks (500 words) → Topic Extraction (Ollama) → Normalize → Embed → Deduplicate → Cluster
                                                                      ↓
                                                               sqlite-vec
                                                                      ↓
                                                            Semantic Search
```

1. **Parse**: Extract text from EPUB files
2. **Chunk**: Split into ~500 word segments
3. **Extract**: Use local LLM to identify 5 topics per chunk
4. **Embed**: Generate semantic embeddings with BGE-small-en-v1.5
5. **Deduplicate**: Merge similar topics (cosine similarity > 0.85)
6. **Cluster**: Group related topics using Leiden algorithm
7. **Search**: Query topics semantically with sqlite-vec

## Features

- **100% Local**: All processing happens on your machine (Ollama + local embeddings)
- **Calibre Integration**: Reads metadata from your Calibre library
- **Semantic Search**: Find topics by meaning, not just keywords
- **Topic Clustering**: Automatic hierarchical organization
- **Co-occurrence Analysis**: Discover topics that appear together
- **SQLite Storage**: Everything in one portable database file

## Installation

Requires Python 3.10+ and [uv](https://github.com/astral-sh/uv) for package management.

```bash
# Clone the repository
git clone https://github.com/seaberger/libtrails.git
cd libtrails

# Create virtual environment and install
uv venv
uv pip install -r requirements.txt
uv pip install -e .

# Install Ollama for topic extraction
# See: https://ollama.ai
ollama pull gemma3:4b
```

### Configuration

Edit `src/libtrails/config.py` to set your Calibre library path:

```python
CALIBRE_LIBRARY_PATH = Path.home() / "Calibre Library"  # Adjust to your path
```

## Quick Start

```bash
# Check status
uv run libtrails status

# Index a book (parse, chunk, extract topics)
uv run libtrails index --title "Siddhartha"

# Run the post-processing pipeline
uv run libtrails process

# Semantic search
uv run libtrails search-semantic "spiritual journey"

# Browse topic clusters
uv run libtrails tree
```

## CLI Reference

### Indexing

| Command | Description |
|---------|-------------|
| `libtrails status` | Show library stats and indexing progress |
| `libtrails index <id>` | Index a book by database ID |
| `libtrails index --title "Name"` | Index a book by title |
| `libtrails index --all` | Index all books |
| `libtrails index --dry-run` | Parse and chunk without topic extraction |
| `libtrails topics <id>` | Show extracted topics for a book |
| `libtrails models` | List available Ollama models |

### Post-Processing

| Command | Description |
|---------|-------------|
| `libtrails process` | Run full pipeline (embed → dedupe → cluster) |
| `libtrails embed` | Generate embeddings for all topics |
| `libtrails embed --force` | Regenerate all embeddings |
| `libtrails dedupe` | Deduplicate similar topics |
| `libtrails dedupe --dry-run` | Preview without making changes |
| `libtrails cluster` | Cluster topics using Leiden algorithm |

### Discovery

| Command | Description |
|---------|-------------|
| `libtrails search-semantic <query>` | Semantic search for topics and books |
| `libtrails search <query>` | Text search in topics |
| `libtrails tree` | Browse full topic hierarchy |
| `libtrails tree <topic>` | Search for specific topics |
| `libtrails related <topic>` | Find related topics via graph |
| `libtrails cooccur <topic>` | Topics that co-occur in chunks |

## Example Output

```
$ uv run libtrails search-semantic "spiritual journey"

                     Matching Topics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┓
┃ Topic                      ┃ Similarity ┃ Occurrences ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━┩
│ spirituality               │ 0.803      │ 4           │
│ spiritual awakening        │ 0.792      │ 14          │
│ river journey              │ 0.774      │ 2           │
│ spiritual practices        │ 0.768      │ 2           │
│ pilgrimage                 │ 0.765      │ 5           │
└────────────────────────────┴────────────┴─────────────┘

Books with matching topics:
┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃ Title      ┃ Author         ┃ Relevance ┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│ Siddhartha │ Hermann Hesse  │ 0.803     │
└────────────┴────────────────┴───────────┘
```

## Architecture

### Database Schema

```sql
books              -- Book metadata (matched to Calibre)
chunks             -- Text chunks (~500 words each)
chunk_topics       -- Raw extracted topics per chunk
topics             -- Normalized topics with embeddings
chunk_topic_links  -- Many-to-many: chunks ↔ topics
topic_cooccurrences -- Co-occurrence counts with PMI scores
topic_vectors      -- sqlite-vec virtual table for vector search
```

### Key Dependencies

| Package | Purpose |
|---------|---------|
| `selectolax` | Fast EPUB/HTML parsing |
| `sentence-transformers` | BGE embeddings |
| `sqlite-vec` | Vector similarity search |
| `python-igraph` | Graph construction |
| `leidenalg` | Community detection |
| `click` + `rich` | CLI interface |

### Embedding Model

Uses [BGE-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5):
- 384 dimensions
- Optimized for semantic textual similarity
- Cached locally in `models/` directory (first run downloads ~130MB)
- Cosine similarity for distance metric

## Roadmap

### Near Term
- [ ] Batch indexing with progress persistence
- [ ] LLM-generated cluster labels
- [ ] Export topic graph for visualization (Gephi, etc.)

### Future
- [ ] Cross-book trail generation
- [ ] Book recommendations based on topic overlap
- [ ] Web UI for browsing
- [ ] Calibre plugin integration

## Contributing

Contributions welcome! Please open an issue to discuss major changes.

## Acknowledgments

- [Pieter Maes](https://pieterma.es) for the original inspiration and approach
- [sqlite-vec](https://github.com/asg017/sqlite-vec) for embedded vector search
- [BGE embeddings](https://huggingface.co/BAAI/bge-small-en-v1.5) from BAAI

## License

MIT
