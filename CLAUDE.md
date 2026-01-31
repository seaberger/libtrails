# Calibre Library Discovery & Trail-Finding Project

## Project Vision

Inspired by [Pieter Maes' "Reading Across Books" project](https://pieterma.es/syntopic-reading-claude/), this project aims to build tools for **discovering conceptual connections** across a personal book library. The goal is to use the curated 969-book iPad reading library as a "lens" to surface valuable books from the larger 40k Calibre library.

### Key Inspiration
- [Trails visualization](https://trails.pieterma.es) - Shows connected excerpts across books
- [Personal reading graph](https://reading.pieterma.es/book/four-thousand-weeks) - Highlights with cross-book connections
- [HN book collection](https://trails.pieterma.es/books/) - 100 books from most-mentioned on Hacker News

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
3. Run `python3 scrape_mapleread.py` or `scrape_mapleread_full.py`

**Current data captured**:
- Book ID (MD5 hash)
- Title
- Author
- Format (epub/pdf)
- Tags (via full scrape)

### 3. External Enrichment Sources (potential)
- **Goodreads** - API deprecated, but web data exists
- **Open Library** - Free API, has many books
- **Google Books API** - Descriptions, categories
- **ISBNdb** - Comprehensive but paid

---

## Technical Architecture (from Pieter's implementation)

| Component | Technology |
|-----------|------------|
| EPUB parsing | `selectolax` |
| Sentence splitting | `wtpsplit` (sat-6l-sm) |
| Storage | SQLite + `sqlite-vec` |
| Topic extraction | Gemini 2.5 Flash Lite |
| Embeddings | `google/embeddings-gemma-300m` |
| Reranking | `BAAI/bge-reranker-v2-m3` |
| Topic graph | `igraph` + Leiden clustering |
| Agent interface | Claude Code with CLI tools |

---

## Possible Directions

### Option A: Metadata-First Discovery
Start with just metadata (titles, authors, descriptions, tags) from both libraries.

**Approach**:
1. Extract all metadata from Calibre's 40k library
2. Use iPad's 969 books as "taste profile"
3. Generate embeddings for descriptions/titles
4. Build topic tree from existing tags
5. Create CLI tools for Claude to explore connections

**Pros**: Fast to implement, no book parsing needed, low cost
**Cons**: Limited depth - can't find connections within book content

### Option B: Deep Read of iPad Library
Use local LLM (Gemma-3) to deeply analyze the 969 iPad books.

**Approach**:
1. Parse EPUBs from iPad library
2. Chunk into ~500 word segments
3. Extract topics per chunk using local LLM
4. Build rich topic tree from actual content
5. Use these deep topics to search/rank the 40k library

**Pros**: Rich understanding of your curated collection
**Cons**: Requires EPUB access, processing time, local LLM setup

### Option C: External Enrichment
Augment all 40k books with Goodreads/OpenLibrary data.

**Approach**:
1. Match books by ISBN/title/author to external sources
2. Pull summaries, genres, similar books, ratings
3. Build comprehensive metadata database
4. Enable search/discovery across enriched data

**Pros**: Covers entire library quickly, crowd-sourced wisdom
**Cons**: Depends on external data quality, some books won't match

### Option D: Hybrid Approach (Recommended)
Combine metadata-first for breadth with deep read for depth.

**Approach**:
1. **Phase 1**: Index all 40k metadata + generate embeddings
2. **Phase 2**: Deep analyze 969 iPad books as "seed corpus"
3. **Phase 3**: Use seed corpus topics to surface related books in main library
4. **Phase 4**: Optionally enrich with external data where gaps exist

---

## Potential CLI Tools for Claude

Based on Pieter's approach, these tools would enable agentic exploration:

```
lib search <query>        # Semantic search across library
lib topics <book_id>      # Show topics extracted from a book
lib similar <book_id>     # Find books similar to this one
lib tree browse           # Navigate topic hierarchy
lib tree find <topic>     # Find all books under a topic
lib trail suggest         # Propose new trail ideas
lib trail create <idea>   # Build a trail from an idea
lib compare <id1> <id2>   # Compare themes between books
lib taste                 # Analyze reading preferences from iPad library
lib surface               # Surface main library books based on taste
```

---

## Current Progress (Jan 30, 2025)

### Completed
- [x] Scraped 969 books from iPad MapleRead library (with tags)
- [x] Matched 927/969 (96%) to Calibre metadata
- [x] Created SQLite database with full-text search

### iPad Library Database Stats
| Metric | Count |
|--------|-------|
| Total books | 969 |
| Matched to Calibre | 927 (96%) |
| Unique authors | 810 |
| iPad tags | 955 |
| Calibre tags | 1,068 |
| With descriptions | 901 |
| With ISBN | 820 |
| In a series | 133 |

### Top Reading Interests (by Calibre tags)
1. Fiction (234)
2. Science Fiction (155)
3. Fantasy (116)
4. History (104)
5. Classics (95)
6. Philosophy (62)
7. Politics (56)
8. Business (50)
9. Writing (49)

---

## Development Setup

This project uses **uv** for package management. All commands should be run with `uv run`.

```bash
# Create virtual environment
uv venv

# Install dependencies
uv pip install -r requirements.txt
uv pip install -e .

# Run CLI commands
uv run libtrails status
uv run libtrails index --title "Book Name" --dry-run
uv run libtrails index 123
uv run libtrails models
```

## File Structure

```
calibre_lib/
├── CLAUDE.md                    # This file
├── pyproject.toml               # Package config
├── requirements.txt             # Dependencies for uv
├── README.md                    # Package readme
├── data/
│   ├── ipad_library.db          # SQLite database with FTS (primary)
│   ├── ipad_library.json        # Raw scraped data
│   ├── ipad_library_enriched.json  # With Calibre metadata
│   ├── ipad_unmatched.json      # 42 books not in Calibre
│   └── ipad_tags.json           # Unique tags list
├── docs/
│   ├── read_books_with_claude_code.md  # Pieter's article
│   ├── calibre_plugin_research.md       # Plugin research
│   ├── calibre-rag-prd.md              # PRD for RAG plugin
│   ├── calibre-plugin-guide.md         # Calibre plugin dev guide
│   ├── calibre-rag-pseudocode.md       # Search pipeline pseudocode
│   └── plugin_idea.md                  # Original plugin concept
├── src/libtrails/               # Main package
│   ├── __init__.py
│   ├── cli.py                   # Click CLI interface
│   ├── config.py                # Paths and settings
│   ├── database.py              # SQLite operations
│   ├── epub_parser.py           # EPUB text extraction
│   ├── chunker.py               # Text chunking (~500 words)
│   └── topic_extractor.py       # Ollama topic extraction
├── scrape_mapleread.py          # Basic iPad library scraper
├── scrape_mapleread_full.py     # Full scraper with tags
├── match_to_calibre.py          # Match iPad books to Calibre
└── create_ipad_db.py            # Create SQLite database
```

---

## Key Decisions to Make

1. **Scope**: Focus on discovery/trails first, or build toward a Calibre plugin?

2. **Processing depth**:
   - Metadata only (fast, cheap)
   - iPad books deep read (moderate)
   - All 40k books (expensive, slow)

3. **LLM strategy**:
   - Local (Gemma-3) for cost-free processing
   - Gemini API for quality topic extraction
   - Hybrid based on task

4. **Primary interface**:
   - CLI tools for Claude Code exploration
   - Web UI for browsing
   - Calibre plugin integration

---

## Next Steps

### Completed
- [x] Match iPad books to Calibre library entries
- [x] Build metadata extraction pipeline
- [x] Create SQLite database with FTS

### Up Next: Deep Reading for Trail-Finding
1. [ ] Access EPUB files from Calibre library for the 927 matched books
2. [ ] Parse EPUBs with `selectolax`, chunk into ~500 word segments
3. [ ] Extract topics per chunk (Gemini API or local Gemma-3)
4. [ ] Generate embeddings for chunks (`sqlite-vec`)
5. [ ] Build topic tree using Leiden clustering
6. [ ] Create CLI tools for Claude to explore connections

### Later: Surface Books from Main Library
7. [ ] Generate embeddings for all 40k book descriptions
8. [ ] Use iPad library topics as "lens" to rank main library
9. [ ] Build trail-finding agentic workflow

---

## Notes

- Calibre database should be treated as **read-only** - create separate index
- iPad MapleRead server only available when manually enabled on device
- User reads on MapleRead SE, has highlights (lower priority for now)
- Book formats: Recent = EPUB, older = MOBI/LIT (may need conversion for parsing)
