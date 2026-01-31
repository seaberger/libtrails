# Product Requirements Document
## Calibre RAG Search Plugin with Gemini API

### Executive Summary
A personal Calibre plugin that implements an advanced RAG (Retrieval-Augmented Generation) search system for a 40,000+ book library, combining fast local vector search with Google Gemini API for intelligent query understanding and result reranking.

### Problem Statement
- Current Calibre search is extremely slow with 40k books
- Keyword search misses semantically related content
- No intelligent query understanding or context-aware search
- Unable to search by concepts, themes, or vague descriptions
- No way to ask natural language questions about the library

### Solution Overview
A hybrid search plugin that:
1. Pre-indexes all books using local embeddings (one-time setup)
2. Performs fast semantic + keyword search locally
3. Uses Gemini API for query enhancement and intelligent reranking
4. Provides natural language question answering about books
5. Maintains sub-second response times for most queries

### Target User
- Single power user (yourself)
- 40,000+ book library
- Technical expertise to configure API keys
- Willing to wait for initial indexing
- Has Google Gemini API access

### Core Features

#### Phase 1: Foundation (Week 1-2)
- **Fast Local Search**
  - SQLite FTS5 for keyword search
  - sqlite-vec for vector similarity
  - Hybrid BM25 + semantic scoring
  - Sub-100ms query response

- **Smart Indexing**
  - Batch process 40k books efficiently
  - Store title, author, tags, description embeddings
  - Incremental updates for new books
  - Progress indicator for initial index

#### Phase 2: Intelligence (Week 3-4)
- **Gemini Integration**
  - Query understanding and expansion
  - Natural language question support
  - Context-aware reranking
  - Smart snippet generation

- **Advanced Features**
  - "Find books similar to X but different in Y"
  - "Books I haven't read about topic Z"
  - Cross-reference multiple books
  - Theme and concept extraction

#### Phase 3: Optimization (Week 5-6)
- **Performance Tuning**
  - Query result caching
  - Embedding cache management
  - Batch API calls optimization
  - Background index updates

- **UI Enhancement**
  - Custom search dialog
  - Rich result previews
  - Search history
  - Saved searches

### Technical Requirements

#### Dependencies
- Python 3.11+ (Calibre 8.x requirement)
- sqlite-vec extension
- sentence-transformers library
- Google Generative AI SDK
- NumPy for vector operations

#### Storage
- Initial index: ~2-3GB for 40k books
- Embedding cache: ~500MB
- Query cache: ~100MB
- Total: ~4GB additional storage

#### Performance Targets
- Initial indexing: <30 minutes for 40k books
- Query response: <100ms for local search
- With Gemini: <2 seconds for enhanced results
- Memory usage: <500MB active

### User Workflow

1. **Installation**
   - Install plugin via Calibre
   - Enter Gemini API key
   - Configure search preferences

2. **Initial Setup**
   - Plugin indexes library in background
   - Shows progress (books/minute)
   - Notifies when ready

3. **Daily Usage**
   - Hotkey triggers search dialog
   - Type natural language query
   - Get instant local results
   - Enhanced results load async

4. **Search Examples**
   - "sci-fi books about AI ethics"
   - "similar to Dune but shorter"
   - "non-fiction about habit formation published after 2020"
   - "books I added last month about cooking"

### Configuration Options

```python
{
    "gemini_api_key": "your-key-here",
    "gemini_model": "gemini-1.5-flash",  # or gemini-1.5-pro
    "embedding_model": "all-MiniLM-L6-v2",
    "max_results": 50,
    "use_gemini": true,
    "cache_embeddings": true,
    "index_full_text": false,  # Start with metadata only
    "batch_size": 100,
    "vector_dimensions": 384,
    "quantize_vectors": true,  # INT8 quantization
    "search_history_size": 100,
    "api_timeout": 5.0,
    "max_tokens_per_query": 1000
}
```

### Success Metrics
- Search speed: 10x faster than native Calibre
- Relevant results in top 10: >90% accuracy
- Zero-result queries: <5%
- API costs: <$5/month for personal use
- User satisfaction: Actually want to use search

### Risk Mitigation
- **API Costs**: Implement caching, batch requests
- **Rate Limits**: Queue and throttle API calls  
- **Index Corruption**: Backup and rebuild capability
- **Memory Issues**: Lazy loading, garbage collection
- **Slow Initial Index**: Incremental indexing option

### Future Enhancements (Post-MVP)
- Full-text book content search
- Reading history integration
- Personalized recommendations
- Multi-library support
- Export search results
- Integration with Calibre tags/series
- Smart Collections based on searches
- Natural language library statistics

### Development Priorities
1. **Must Have**
   - Fast local search
   - Gemini query enhancement
   - Progress indicators
   - Error handling

2. **Should Have**
   - Search history
   - Result caching
   - Incremental indexing
   - Configuration UI

3. **Nice to Have**
   - Full-text search
   - Export capabilities
   - Advanced analytics
   - Multiple API providers

### Acceptance Criteria
- [ ] Searches 40k library in <100ms
- [ ] Integrates with Gemini API successfully
- [ ] Handles API errors gracefully
- [ ] Persists index between sessions
- [ ] Updates index for new books automatically
- [ ] Provides better results than native search
- [ ] Uses <$5/month in API costs
- [ ] Installs cleanly via standard plugin mechanism