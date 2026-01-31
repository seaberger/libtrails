# Calibre RAG Search Plugin

## Project Overview
Advanced search plugin for Calibre that implements RAG (Retrieval-Augmented Generation) using local vector search and Google Gemini API to enable fast, intelligent searching across a 40,000+ book library.

## Primary Objectives
1. Build a Calibre plugin that provides sub-100ms semantic search
2. Integrate Gemini API for natural language query understanding
3. Create hybrid BM25 + vector search with intelligent reranking
4. Support natural language questions about the library
5. Maintain all search indices locally for privacy and speed

## Technical Stack
- **Language**: Python 3.11+ (Calibre 8.x requirement)
- **Vector DB**: sqlite-vec extension for SQLite
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **LLM**: Google Gemini API (gemini-1.5-flash)
- **UI Framework**: PyQt6 (Calibre's GUI framework)
- **Search**: SQLite FTS5 + vector similarity

## Project Structure
```
calibre-rag-search/
├── __init__.py                 # Plugin metadata and entry point
├── main.py                     # Main plugin logic
├── config.py                   # Configuration handling
├── search_engine.py            # Core search implementation
├── indexer.py                  # Book indexing logic
├── gemini_client.py           # Gemini API integration
├── vector_store.py            # Vector database management
├── ui.py                      # Qt-based user interface
├── utils.py                   # Helper functions
├── plugin-import-name-rag_search.txt  # Empty file for multi-file support
├── resources/
│   ├── icon.png              # Plugin icon
│   └── styles.css            # UI styling
└── sqlite_vec/
    ├── vec0.so               # Linux
    ├── vec0.dll              # Windows
    └── vec0.dylib            # macOS
```

## Development Phases

### Phase 1: Foundation (Current Focus)
- [ ] Set up plugin structure with proper Calibre integration
- [ ] Implement SQLite with sqlite-vec extension
- [ ] Create basic indexing for book metadata
- [ ] Build hybrid BM25 + vector search
- [ ] Design search UI dialog
- [ ] Add configuration management

### Phase 2: Intelligence
- [ ] Integrate Gemini API client
- [ ] Implement query understanding
- [ ] Add result reranking with LLM
- [ ] Support natural language questions
- [ ] Create smart snippet generation

### Phase 3: Optimization
- [ ] Add comprehensive caching layers
- [ ] Implement incremental indexing
- [ ] Optimize memory usage for 40k books
- [ ] Add search history and saved searches
- [ ] Performance profiling and tuning

## Key Implementation Details

### Database Schema
```sql
-- Book metadata
CREATE TABLE book_metadata (
    book_id INTEGER PRIMARY KEY,
    title TEXT,
    authors TEXT,
    tags TEXT,
    description TEXT,
    indexed_at TIMESTAMP
);

-- Vector storage
CREATE VIRTUAL TABLE book_vectors USING vec0(
    book_id INTEGER PRIMARY KEY,
    title_embedding FLOAT[384],
    description_embedding FLOAT[384],
    combined_embedding FLOAT[384]
);

-- Full-text search
CREATE VIRTUAL TABLE book_fts USING fts5(
    book_id, title, authors, tags, description
);
```

### Configuration Options
```python
{
    "gemini_api_key": "",  # User must provide
    "gemini_model": "gemini-1.5-flash",
    "embedding_model": "all-MiniLM-L6-v2",
    "max_results": 50,
    "use_gemini": true,
    "cache_embeddings": true,
    "batch_size": 100,
    "vector_dimensions": 384,
    "quantize_vectors": true
}
```

### Search Pipeline
1. Parse query and extract filters
2. Run parallel keyword + vector search
3. Merge results with weighted scoring
4. Optional: Rerank with Gemini API
5. Return formatted results with explanations

## Development Guidelines

### Code Style
- Follow PEP 8 with 4-space indentation
- Use type hints for all functions
- Add docstrings for classes and public methods
- Keep functions under 50 lines
- Use descriptive variable names

### Error Handling
- Always wrap API calls in try-except
- Provide fallback search methods
- Show user-friendly error messages
- Log errors for debugging
- Never crash the main Calibre application

### Performance Requirements
- Initial indexing: < 30 minutes for 40k books
- Search response: < 100ms without Gemini
- With Gemini: < 2 seconds total
- Memory usage: < 500MB active
- Index size: < 4GB on disk

### Testing Strategy
- Test with libraries of different sizes (100, 1k, 10k, 40k books)
- Verify search accuracy with known queries
- Monitor API usage and costs
- Test error recovery mechanisms
- Profile memory and CPU usage

## Current Development Focus

Start with `__init__.py` to define the plugin metadata, then implement:

1. **Basic Plugin Structure** (`__init__.py`, `main.py`)
   - Plugin metadata and registration
   - Menu integration and hotkey
   - Basic UI dialog

2. **Database Setup** (`vector_store.py`)
   - SQLite database creation
   - sqlite-vec extension loading
   - Table schema implementation

3. **Indexing System** (`indexer.py`)
   - Book metadata extraction
   - Batch processing logic
   - Progress indicators

4. **Search Implementation** (`search_engine.py`)
   - Keyword search with FTS5
   - Vector similarity search
   - Result merging

5. **Embedding Generation** (`utils.py`)
   - Load sentence-transformers model
   - Generate embeddings for text
   - Caching mechanism

## Important Calibre APIs

### Get Book Metadata
```python
db = self.gui.current_db.new_api
book_ids = db.all_book_ids()
metadata = db.get_metadata(book_id)
```

### Show Progress
```python
from calibre.gui2.progress_indicator import ProgressIndicator
with ProgressIndicator(self.gui, 'Indexing...', max=total) as pi:
    pi.increment()
```

### Background Jobs
```python
from calibre.gui2.threaded_jobs import ThreadedJob
job = ThreadedJob('name', 'description', func, callback)
self.gui.job_manager.run_threaded_job(job)
```

### Configuration Storage
```python
from calibre.utils.config import JSONConfig
prefs = JSONConfig('plugins/rag_search')
prefs['gemini_api_key'] = api_key
```

## Dependencies to Install

The plugin should bundle these or download on first run:
- sentence-transformers (~30MB)
- sqlite-vec extension (~5MB per platform)
- numpy (if not already in Calibre)

## Gemini API Integration

### Rate Limits
- 60 requests per minute (RPM)
- 1 million tokens per minute (TPM)
- Implement exponential backoff

### Cost Optimization
- Cache all API responses
- Batch multiple queries when possible
- Use gemini-1.5-flash for cost efficiency
- Log token usage for monitoring

## Debugging Commands

```bash
# Run Calibre in debug mode
calibre-debug -g

# Install plugin from directory
calibre-customize -b calibre-rag-search/

# Remove plugin
calibre-customize -r "RAG Search"

# View console output
calibre-debug -g 2>&1 | grep "RAG Search"
```

## Known Challenges

1. **Large Library Indexing**: 40k books require careful memory management
2. **Platform Compatibility**: sqlite-vec needs platform-specific binaries
3. **API Rate Limits**: Must implement proper queuing and caching
4. **UI Responsiveness**: All heavy operations must run in background
5. **Index Updates**: Need efficient incremental indexing

## Success Criteria

- [ ] Searches 40k library in under 100ms
- [ ] Integrates Gemini API successfully
- [ ] Provides better results than native Calibre search
- [ ] Handles errors gracefully
- [ ] Uses less than $5/month in API costs
- [ ] Installs cleanly as standard plugin
- [ ] Updates index automatically for new books

## Next Steps

1. Create the basic plugin structure
2. Implement local search without Gemini
3. Test with small library subset
4. Add Gemini integration
5. Scale testing to full 40k library
6. Optimize performance
7. Add advanced features

## Notes for Claude

- Always check if operations should run in background thread
- Use Calibre's built-in dialogs and widgets when possible
- Test each component in isolation before integration
- Keep API keys secure and never log them
- Prioritize search speed over feature completeness
- Make Gemini optional - plugin should work without it