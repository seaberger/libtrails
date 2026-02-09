# LibTrails CLI Guide

## Setup

```bash
# Install dependencies
uv sync

# Use v2 database (recommended for new work)
export LIBTRAILS_DB=v2
```

All commands below assume `LIBTRAILS_DB=v2` is set. Without it, the v1 database is used.

---

## Indexing Books

### Index a single book

```bash
# By ID
libtrails index 517

# By title (fuzzy match)
libtrails index --title "Siddhartha"

# Multiple books by ID
libtrails index --id 517 --id 702 --id 146
```

### Index the entire library

```bash
# First run (skips books that already have chunks)
libtrails index --all

# Re-index everything (re-parse, re-chunk, re-extract)
# Resumes automatically — skips books that already have topics
libtrails index --all --reindex
```

### Model options

```bash
# Default: local Ollama (free, ~1s/chunk)
libtrails index --all --reindex \
  --theme-model gemma3:27b \
  --chunk-model gemma3:4b \
  --batch-size 5

# Gemini API (faster, better quality, ~$31 for full library)
libtrails index --all --reindex \
  --theme-model gemini/gemini-3-flash-preview \
  --chunk-model gemini/gemini-2.5-flash-lite \
  --parallel --workers 30

# Set both models at once
libtrails index --all --model gemma3:4b

# Legacy mode (v1 pipeline, no themes, no batching)
libtrails index --all --legacy
```

### Controlling chunk size and book size

```bash
# Larger chunks (fewer API calls, less granular topics)
libtrails index --all --chunk-size-words 1000

# Skip very large books (collected works, omnibus editions)
libtrails index --all --max-words 500000
```

### Dry run (parse and chunk without topic extraction)

```bash
libtrails index --title "Dune" --dry-run
```

### Safety features

- **Battery monitoring**: Pauses at 15% battery, resumes at 50% (configurable with `--min-battery`)
- **Interrupt handling**: Ctrl+C saves progress; run the same command again to resume
- **Resume on reindex**: `--reindex` skips books that already have extracted topics

---

## Post-Processing Pipeline

### Run the full pipeline at once

```bash
libtrails process
```

This runs: embed → dedupe → cluster → refresh-stats

### Run individual steps

```bash
# Generate embeddings for all topics
libtrails embed
libtrails embed --force  # Regenerate all (after model change)

# Deduplicate similar topics
libtrails dedupe --dry-run              # Preview merges
libtrails dedupe --dry-run -t 0.97      # Stricter threshold
libtrails dedupe                        # Execute deduplication
libtrails dedupe -s 5000               # Test on 5K sample

# Cluster topics with Leiden algorithm
libtrails cluster                              # Defaults (CPM, knn)
libtrails cluster --resolution 0.01            # Fewer, larger clusters
libtrails cluster --resolution 0.5             # More, smaller clusters
libtrails cluster --mode cooccurrence          # Sparse graph (faster)
libtrails cluster --skip-cooccur               # Reuse existing co-occurrences
libtrails cluster --remove-hubs                # Remove hub topics first
libtrails cluster --dry-run --resolution 0.2   # Test without saving

# Refresh materialized stats (runs automatically after cluster)
libtrails refresh-stats
```

---

## Discovery & Search

### Text search

```bash
libtrails search "machine learning"
```

### Semantic search (requires embeddings)

```bash
libtrails search-semantic "spiritual journey"
libtrails search-semantic "investment strategies" -n 20
```

### Explore topic relationships

```bash
# Topics that co-occur in the same chunks
libtrails cooccur "spiritual awakening"

# Topics connected via the graph
libtrails related "desert ecology" -n 15

# Browse the topic hierarchy
libtrails tree
libtrails tree "philosophy"
```

### Book-level exploration

```bash
# Show all topics for a book
libtrails topics 517
libtrails topics --title "Dune"

# Show which clusters a book belongs to
libtrails book-clusters 517
libtrails book-clusters --title "Siddhartha"
```

---

## Domains & Clusters

### Generate cluster labels

```bash
libtrails label-clusters                    # Label all clusters
libtrails label-clusters --limit 50         # Label top 50 only
libtrails label-clusters --model gemma3:27b # Use larger model
libtrails label-clusters --force            # Re-label all
```

### Generate domains (super-clusters)

```bash
# 1. Generate super-clusters from Leiden clusters
libtrails regenerate-domains
libtrails regenerate-domains -n 25          # Target 25 domains

# 2. Review auto-labels, update REFINED_LABELS in:
#    experiments/domain_labels_final.py

# 3. Generate final domain labels JSON
uv run python experiments/domain_labels_final.py

# 4. Load into database
libtrails load-domains
libtrails load-domains -f data/domain_labels.json
```

---

## Diagnostics

### Library status

```bash
libtrails status    # Indexing progress, chunk counts, topic counts
libtrails formats   # EPUB vs PDF distribution
libtrails models    # Available Ollama models
```

### Clustering diagnostics

```bash
# Analyze hub topics causing clustering problems
libtrails diagnose-hubs
libtrails diagnose-hubs --top-n 100
```

---

## API Server

```bash
libtrails serve                        # Default: localhost:8000
libtrails serve --port 9000            # Custom port
libtrails serve --reload               # Auto-reload for development
```

---

## Database Management

### Prepare a v2 database

```bash
# Copies current DB, clears topic tables, keeps books + chunks
libtrails prepare-v2

# Then use v2 for all subsequent commands
export LIBTRAILS_DB=v2
```

### Switch between databases

```bash
LIBTRAILS_DB=v2 libtrails status   # v2 database
libtrails status                    # v1 database (default)
```

---

## Visualization

### Generate galaxy universe coordinates

```bash
libtrails generate-universe
libtrails generate-universe --n-neighbors 15 --min-dist 0.2
libtrails generate-universe --dry-run   # Show counts only
```

---

## iPad Sync

```bash
# Sync new books from MapleRead (iPad server must be running)
libtrails sync -i http://192.168.1.124:8082
libtrails sync -i http://192.168.1.124:8082 --dry-run   # Preview
libtrails sync -i http://192.168.1.124:8082 --skip-index # Add without indexing
libtrails sync -i http://192.168.1.124:8082 --save-url   # Remember URL
```

---

## Full Library Workflow

```bash
# 1. Set database
export LIBTRAILS_DB=v2

# 2. Index all books (resumes if interrupted)
libtrails index --all --reindex \
  --theme-model gemini/gemini-3-flash-preview \
  --chunk-model gemini/gemini-2.5-flash-lite \
  --parallel --workers 30

# 3. Post-process
libtrails process

# 4. Label and organize
libtrails label-clusters
libtrails regenerate-domains
# ... edit domain labels ...
libtrails load-domains

# 5. Generate visualization data
libtrails generate-universe

# 6. Start the web interface
libtrails serve
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LIBTRAILS_DB` | *(unset)* | Database variant (`v2` → `ipad_library_v2.db`) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama API endpoint |
| `GEMINI_API_KEY` | *(required for Gemini)* | Google AI API key |
| `OLLAMA_NUM_CTX` | `8192` | Ollama context window size |
