# Calibre Library Discovery & Trail-Finding Project

## Project Vision

Inspired by [Pieter Maes' "Reading Across Books" project](https://pieterma.es/syntopic-reading-claude/), this project aims to build tools for **discovering conceptual connections** across a personal book library. The goal is to use the curated 969-book iPad reading library as a "lens" to surface valuable books from the larger 40k Calibre library.

### Key Inspiration
- [Trails visualization](https://trails.pieterma.es) - Shows connected excerpts across books
- [Personal reading graph](https://reading.pieterma.es/book/four-thousand-weeks) - Highlights with cross-book connections
- [HN book collection](https://trails.pieterma.es/books/) - 100 books from most-mentioned on Hacker News

---

## Implemented Architecture

| Component | Technology | Status |
|-----------|------------|--------|
| EPUB parsing | `selectolax` | ✅ |
| Text chunking | Custom (~500 words) | ✅ |
| Topic extraction | Ollama (gemma3:4b/27b) | ✅ |
| Embeddings | `BAAI/bge-small-en-v1.5` (384 dims) | ✅ |
| Vector search | `sqlite-vec` (cosine distance) | ✅ |
| Topic graph | `python-igraph` | ✅ |
| Clustering | `leidenalg` (Surprise quality) | ✅ |
| Storage | SQLite + FTS | ✅ |
| CLI | `click` + `rich` | ✅ |

### Pipeline Flow

```
EPUB → selectolax → Chunks (500 words) → Ollama → Raw Topics
                                                      ↓
                                              Normalize (lowercase, strip)
                                                      ↓
                                              BGE Embeddings (384 dims)
                                                      ↓
                                    ┌─────────────────┴─────────────────┐
                                    ↓                                   ↓
                            sqlite-vec index                    Deduplication
                            (cosine distance)                   (similarity > 0.85)
                                    ↓                                   ↓
                            Semantic Search              Co-occurrence Graph (PMI)
                                                                        ↓
                                                         Leiden Clustering
                                                                        ↓
                                                            Topic Hierarchy
```

---

## Data Sources

### 1. Main Calibre Library
- **Location**: `/Users/seanbergman/Calibre_Main_Library/`
- **Database**: `metadata.db` (SQLite)
- **Size**: **39,374 books**
- **Metadata available**:
  - 17,677 unique authors
  - 14,173 unique tags
  - 5,133 unique series
  - Descriptions/comments for many books
  - Identifiers (ISBN, etc.)
  - Formats: Mix of EPUB, MOBI, LIT (older), PDF

**Access pattern**: Read-only queries against `metadata.db`. Do NOT modify directly - work from copies for any writes.

```sql
-- Example: Get all books with metadata
SELECT
    b.id, b.title, b.author_sort, b.pubdate,
    GROUP_CONCAT(DISTINCT t.name) as tags,
    GROUP_CONCAT(DISTINCT a.name) as authors,
    s.name as series,
    c.text as description
FROM books b
LEFT JOIN books_authors_link bal ON b.id = bal.book
LEFT JOIN authors a ON bal.author = a.id
LEFT JOIN books_tags_link btl ON b.id = btl.book
LEFT JOIN tags t ON btl.tag = t.id
LEFT JOIN books_series_link bsl ON b.id = bsl.book
LEFT JOIN series s ON bsl.series = s.id
LEFT JOIN comments c ON b.id = c.book
GROUP BY b.id;
```

### 2. iPad Reading Library (MapleRead SE)
- **Size**: **969 books** - curated, high-value collection
- **Access**: HTTP server from iPad when enabled
- **Server URL**: `http://192.168.1.124:8082` (when iPad server is running)
- **Data scraped to**: `data/ipad_library.json`

**To refresh iPad library data**:
1. Open MapleRead SE on iPad
2. Enable "Book Sharing" server
3. Run `python3 utils/scrape_mapleread.py` or `utils/scrape_mapleread_full.py`

---

## Database Schema

### Core Tables

```sql
-- Books from iPad library (matched to Calibre)
CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    title TEXT,
    author TEXT,
    calibre_id INTEGER,          -- FK to Calibre metadata.db
    description TEXT,
    -- ... other metadata
);

-- Text chunks from parsed EPUBs
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY,
    book_id INTEGER REFERENCES books(id),
    chunk_index INTEGER,
    content TEXT,
    word_count INTEGER
);

-- Raw topics extracted per chunk (before normalization)
CREATE TABLE chunk_topics (
    id INTEGER PRIMARY KEY,
    chunk_id INTEGER REFERENCES chunks(id),
    topic TEXT
);

-- Normalized, deduplicated topics with embeddings
CREATE TABLE topics (
    id INTEGER PRIMARY KEY,
    label TEXT UNIQUE NOT NULL,       -- normalized: lowercase, trimmed
    embedding BLOB,                    -- 384-dim float32 vector
    cluster_id INTEGER,               -- Leiden cluster assignment
    parent_topic_id INTEGER,          -- for hierarchical structure
    occurrence_count INTEGER DEFAULT 0
);

-- Many-to-many: chunks ↔ normalized topics
CREATE TABLE chunk_topic_links (
    chunk_id INTEGER REFERENCES chunks(id),
    topic_id INTEGER REFERENCES topics(id),
    PRIMARY KEY (chunk_id, topic_id)
);

-- Topic co-occurrence with PMI scores
CREATE TABLE topic_cooccurrences (
    topic1_id INTEGER,
    topic2_id INTEGER,
    count INTEGER DEFAULT 0,
    pmi REAL,                         -- Pointwise Mutual Information
    PRIMARY KEY (topic1_id, topic2_id)
);

-- sqlite-vec virtual table for vector search
CREATE VIRTUAL TABLE topic_vectors USING vec0(
    topic_id INTEGER PRIMARY KEY,
    embedding FLOAT[384] distance_metric=cosine
);
```

---

## Key Implementation Details

### Embedding Model

Using **BGE-small-en-v1.5** instead of all-MiniLM-L6-v2:
- Better performance on semantic textual similarity (STS) tasks
- Same 384 dimensions, similar speed
- Top performer on MTEB benchmark for its size

```python
# src/libtrails/embeddings.py
MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION = 384
MODEL_CACHE_DIR = PROJECT_ROOT / "models"  # Cached locally
```

Model is downloaded on first use and cached in `models/` directory (~130MB).

### Vector Search with Cosine Distance

sqlite-vec configured for cosine similarity (not L2/Euclidean):

```python
# src/libtrails/vector_search.py
conn.execute(f"""
    CREATE VIRTUAL TABLE IF NOT EXISTS topic_vectors USING vec0(
        topic_id INTEGER PRIMARY KEY,
        embedding FLOAT[{dim}] distance_metric=cosine
    )
""")
```

**Important**: sqlite-vec requires `k=?` in WHERE clause, not LIMIT:
```sql
-- Correct
SELECT topic_id, distance FROM topic_vectors
WHERE embedding MATCH ? AND k = 20
ORDER BY distance;

-- Wrong (will error)
SELECT topic_id, distance FROM topic_vectors
WHERE embedding MATCH ?
ORDER BY distance LIMIT 20;
```

### Topic Normalization

```python
# src/libtrails/topic_extractor.py
def normalize_topic(topic: str) -> str:
    normalized = topic.strip().lower().replace("_", " ")
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    return normalized
```

### Deduplication Algorithm

Uses Union-Find to merge topics with cosine similarity > 0.85:

1. Compute pairwise similarity matrix for all embeddings
2. Build connected components where similarity > threshold
3. Keep most frequent label as canonical
4. Update all chunk_topic_links to point to canonical topic
5. Delete duplicate topics

### Leiden Clustering

Uses igraph + leidenalg with Surprise quality function:

```python
import leidenalg
partition = leidenalg.find_partition(
    graph,
    leidenalg.SurpriseVertexPartition
)
```

Graph edges come from:
1. **Embedding similarity**: cosine > 0.5 threshold
2. **Co-occurrence**: topics appearing in same chunks (weighted by PMI)

---

## CLI Commands

```bash
# Indexing
uv run libtrails status                    # Library stats
uv run libtrails index --title "Book"      # Index single book
uv run libtrails index --all               # Index all books
uv run libtrails topics 123                # Show book's topics

# Post-processing pipeline
uv run libtrails embed                     # Generate embeddings
uv run libtrails embed --force             # Regenerate all (after model change)
uv run libtrails dedupe --dry-run          # Preview deduplication
uv run libtrails dedupe                    # Merge similar topics
uv run libtrails cluster                   # Leiden clustering
uv run libtrails process                   # Run full pipeline

# Discovery
uv run libtrails search-semantic "query"   # Semantic search
uv run libtrails tree                      # Browse topic hierarchy
uv run libtrails related "topic"           # Graph-connected topics
uv run libtrails cooccur "topic"           # Co-occurring topics

# Domain (Super-cluster) Management
uv run libtrails regenerate-domains        # Generate super-clusters from Leiden
uv run libtrails load-domains              # Load domain labels into database
```

### Domain Regeneration Workflow

After re-running Leiden clustering, domains (super-clusters) need to be regenerated:

```bash
# 1. Generate new super-clusters (uses K-means on cluster centroids)
uv run libtrails regenerate-domains

# 2. Review auto-labels and update the mapping in:
#    experiments/domain_labels_final.py (REFINED_LABELS dict)

# 3. Generate final domain labels JSON
uv run python experiments/domain_labels_final.py

# 4. Load into database
uv run libtrails load-domains
```

The domain system groups ~600-800 Leiden clusters into ~20-25 high-level themes:
- **Literary Worlds**: Fantasy fiction (Mistborn, Harry Potter, WoT)
- **Human Condition**: Philosophy, ethics, identity
- **Wild Earth**: Nature, paleontology, crafts
- **Financial Strategy**: Investment, risk management
- **Machine Learning**: AI/ML technical topics
- etc.

---

## File Structure

```
calibre_lib/
├── CLAUDE.md                    # This file (internal dev notes)
├── README.md                    # Public-facing documentation
├── pyproject.toml               # Package config
├── requirements.txt             # Dependencies for uv
├── data/
│   ├── ipad_library.db          # SQLite database (primary)
│   ├── ipad_library.json        # Raw scraped data
│   ├── ipad_library_enriched.json
│   ├── ipad_unmatched.json
│   └── ipad_tags.json
├── docs/
│   ├── read_books_with_claude_code.md
│   ├── calibre_plugin_research.md
│   ├── calibre-rag-prd.md
│   ├── calibre-plugin-guide.md
│   ├── calibre-rag-pseudocode.md
│   └── plugin_idea.md
├── models/                      # Cached embedding model (gitignored)
│   └── BAAI_bge-small-en-v1.5/
├── src/libtrails/
│   ├── __init__.py
│   ├── cli.py                   # Click CLI with all commands
│   ├── config.py                # Paths, thresholds, model settings
│   ├── database.py              # SQLite operations, schema
│   ├── epub_parser.py           # selectolax EPUB extraction
│   ├── chunker.py               # Text chunking (~500 words)
│   ├── topic_extractor.py       # Ollama topic extraction + normalization
│   ├── embeddings.py            # BGE model, caching, utilities
│   ├── vector_search.py         # sqlite-vec search functions
│   ├── deduplication.py         # Union-Find topic merging
│   ├── topic_graph.py           # igraph construction, co-occurrence
│   ├── clustering.py            # Leiden algorithm wrapper
│   └── domains.py               # Domain (super-cluster) generation
└── utils/
    ├── scrape_mapleread.py      # Basic iPad scraper
    ├── scrape_mapleread_full.py # Full scraper with tags
    ├── match_to_calibre.py      # Match iPad→Calibre
    └── create_ipad_db.py        # Initial DB creation
```

---

## Database Versions (V1 vs V2)

**IMPORTANT: V1 and V2 are independent databases. Never cross-query or cross-index between them unless explicitly requested to compare results.**

| | V1 | V2 |
|---|---|---|
| **File** | `data/ipad_library.db` | `data/ipad_library_v2.db` |
| **Env var** | (default) | `LIBTRAILS_DB=v2` |
| **Pipeline** | Single-pass, gemma3:4b, no book themes | Two-pass: gemma3:27b themes + gemma3:4b/Gemini/LM Studio chunk topics |
| **Topic style** | Single-word generic ("Security", "Wall Street") | Multi-word noun phrases ("benghazi attack timeline", "wall street trading practices of the 1920s") |
| **Status** | All 927 books indexed (V1 pipeline) | 338+ books indexed (V2 pipeline), will become primary |

To switch databases, set the environment variable:
```bash
# Use V2 database
LIBTRAILS_DB=v2 uv run libtrails status

# Use V1 database (default)
uv run libtrails status
```

The `LIBTRAILS_DB` env var is read in `config.py` and controls `IPAD_DB_PATH`.

---

## Configuration

Edit `src/libtrails/config.py`:

```python
# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IPAD_DB_PATH = DATA_DIR / "ipad_library.db"  # or ipad_library_v2.db via LIBTRAILS_DB=v2

# Calibre library (read-only)
CALIBRE_LIBRARY_PATH = Path.home() / "Calibre_Main_Library"
CALIBRE_DB_PATH = CALIBRE_LIBRARY_PATH / "metadata.db"

# LLM settings
DEFAULT_MODEL = "gemma3:4b"  # Use 4b for topic extraction (faster, sufficient quality)
OLLAMA_HOST = "http://localhost:11434"

# Chunking settings
CHUNK_TARGET_WORDS = 500
CHUNK_MIN_WORDS = 100

# Topic extraction
TOPICS_PER_CHUNK = 5

# Deduplication settings
DEDUP_SIMILARITY_THRESHOLD = 0.85

# Graph/clustering settings
EMBEDDING_EDGE_THRESHOLD = 0.5
COOCCURRENCE_MIN_COUNT = 2
PMI_MIN_THRESHOLD = 0.0
```

---

## Current Progress (Jan 30, 2025)

### Completed ✅
- [x] Scraped 969 books from iPad MapleRead library (with tags)
- [x] Matched 927/969 (96%) to Calibre metadata
- [x] Created SQLite database with full-text search
- [x] EPUB parsing with selectolax
- [x] Text chunking (~500 words per chunk)
- [x] Topic extraction via Ollama (gemma3:4b)
- [x] Topic normalization and deduplication
- [x] BGE embeddings with local caching
- [x] sqlite-vec with cosine distance
- [x] Co-occurrence analysis with PMI
- [x] Leiden clustering with igraph
- [x] Full CLI interface

### Current Stats (1 book indexed: Siddhartha)
| Metric | Value |
|--------|-------|
| Chunks | 83 |
| Raw topics | 223 |
| Normalized topics | 223 |
| Embedded topics | 223 |
| Clusters | 70 |

### Search Quality
With BGE + cosine distance, semantic search produces realistic scores:
- "spiritual journey" → "spirituality": 0.803
- "spiritual journey" → "spiritual awakening": 0.792
- "spiritual journey" → "pilgrimage": 0.765

---

## Next Steps

### Short Term
1. [ ] Index more books from iPad library (currently 1/927)
2. [ ] Add batch indexing with progress persistence (resume on crash)
3. [ ] Improve topic extraction prompts for better quality
4. [ ] LLM-generated cluster labels (use Ollama to name clusters)

### Medium Term
5. [ ] Export topic graph for visualization (Gephi, D3.js)
6. [ ] Cross-book trail generation (find excerpts across books on a topic)
7. [ ] Book recommendations based on topic overlap
8. [ ] Web UI for browsing topics and trails

### Long Term
9. [ ] Index descriptions from full 40k Calibre library
10. [ ] Use iPad library topics as "lens" to surface related books
11. [ ] Integration with reading highlights from MapleRead
12. [ ] Calibre plugin for in-app discovery

---

## Known Issues & Fixes

### sqlite-vec k=? syntax
sqlite-vec requires `k=?` in WHERE clause:
```sql
-- Use this
WHERE embedding MATCH ? AND k = 20

-- Not this (errors)
ORDER BY distance LIMIT 20
```

### Vector table recreation
When changing distance metric (L2 → cosine), must recreate table:
```python
rebuild_vector_index(conn, force_recreate=True)
```

### Model caching
First run downloads ~130MB model. Subsequent runs use local cache in `models/`.

---

## Safety Guidelines

### Tool Usage — Serena Stale Cache Risk
- **NEVER use Serena's `replace_content` or `replace_symbol_body` for file edits.** Serena's language server can cache stale file versions, and `replace_content` may operate on the cached version rather than the actual file on disk. This has caused silent data loss (388 lines deleted from `topic_extractor.py` in Feb 2025).
- **Always use the `Edit` tool** (Claude Code's native tool) for all file modifications — it reads directly from disk.
- **Always use the `Read` tool** to verify file contents before editing — never trust Serena's `find_symbol` body output as ground truth for what's on disk.
- Serena's `find_symbol`, `get_symbols_overview`, and `find_referencing_symbols` are safe for **navigation and discovery** (finding symbol names, locations, references), but always confirm bodies with `Read` before editing.

### Rich Progress Bars and Background Processes
- **Rich's `Progress` with `SpinnerColumn` deadlocks when stdout is not a TTY** (e.g. `nohup > file`). The live-rendering thread blocks on terminal operations.
- For background/headless runs, use `screen` + `script` to provide a proper PTY, or disable Rich progress bars entirely.
- When adding new Progress bars, consider TTY-safe fallbacks for non-interactive use.

### Process Management
- **NEVER use broad `pkill -f` patterns** - they can match system processes
- When killing processes, use specific PIDs: `kill <PID>` instead of `pkill -f "pattern"`
- If using `pkill`, be as specific as possible and verify the pattern first with `pgrep -f "pattern"`
- Example of what NOT to do: `pkill -f "libtrails"` (too broad, matches system processes)
- Example of safe approach: `pgrep -f "libtrails dedupe" | xargs kill` (verify then kill)

---

## Coding Standards

### Imports
- **All imports go at the top of the file** - never bury imports inside functions
- Exception: Heavy ML libraries (`sentence_transformers`, `docling`) may be lazy-loaded for CLI performance
- Use `ruff check --select=F401,F841` to find unused imports

### Error Handling
- Keep try blocks small and focused - don't wrap 20+ lines in a single try
- Extract logic into helper functions, let the helper handle or propagate errors
- Prefer early returns over deep nesting

### Code Organization
- Keep files under ~500 lines when practical
- Extract reusable logic into helper functions
- Use descriptive function names that indicate what they return

---

## Notes

- Calibre database should be treated as **read-only** - create separate index
- iPad MapleRead server only available when manually enabled on device
- User reads on MapleRead SE, has highlights (lower priority for now)
- Book formats: Recent = EPUB, older = MOBI/LIT (may need conversion for parsing)
