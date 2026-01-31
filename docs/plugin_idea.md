Calibre plugin idea

I have an advanced RAG pipeline in Sightline that I would like to adapt into an agentic search plugin for the calibre book library. Please research calibre and plugins and suggest how I might develop and publish a plugin of this nature to calibre. The challenge will be the LLM API. I would like to see if there is a way to offer a free preview with a certain number of uses, and then users would need to pay something to cover the inference costs. But for prototyping, I'm interested in developing something that works for myself. One issue with calibre search is that it is really slow if you have a lot of books in the library. I'd like to index all of the titles, authors, and metadata (tags) and book descriptions to use in a RAG+ hybrid search pipeline.

I have calibre 8.8 with 40k books and keyword or phrase search is still horribly slow. I would like to develop a faster and more advanced search that generates a separate index and incorporates LLM reasoning. Please evaluate the feasibility of downloading a local model LLM from huggingface as part of this deployment. Can this be done for users that don't have a huggingface account when they download a plugin? Alternatively, I'm thinking of fine-tuning a small LLM for this application like the brand new Google Gemma 3 270M which is on the order of a 500MB download. It could be fine-tuned and optimized for this search application and result in a small-lite LLM inference engine if there is any way to integrate that into a plugin download for Calibre. In that case it could be free.

Initially I would like to develop this plugin just for me using my Google Gemini API key. If I can get it working well, then I would consider proceeding with an open source project. Please generate a project plan for my personal plugin. I will develop it in claude code so please develop a PRD, info about Calibre plugins, pseudocode, and a claude.md file for me to get the project started

I'll evaluate the feasibility of integrating local LLM models into a Calibre plugin for your 40k book library, focusing on deployment strategies and technical constraints.Based on my research, I can provide a comprehensive evaluation of integrating local LLMs into a Calibre plugin for your 40k book library:
Feasibility Assessment
Local Model Download Without HuggingFace Account
Yes, this is entirely feasible. Hugging Face models can be downloaded programmatically without requiring user authentication using the huggingface_hub library or direct HTTP downloads. Models are publicly accessible and can be downloaded using methods like AutoModel.from_pretrained() which automatically caches them locally.
For a Calibre plugin, you could:
1. Download models programmatically on first run using snapshot_download() or hf_hub_download()
2. Cache them in Calibre's configuration directory
3. No user account or API keys required for public models
Google Gemma 3 270M Integration
The new Gemma 3 270M is particularly promising for your use case:
* Size: Requires around 550MB of memory and the INT4-quantized version uses just 0.75% battery for 25 conversations on mobile devices
* Performance: Outperforms similarly-sized models like SmollLM2-360M and Qwen 2.5 0.5B on instruction-following benchmarks
* Training: Trained on 6 trillion tokens with knowledge cutoff of August 2024
* Optimization: Production-ready INT4 quantization available with minimal performance degradation
Critical Challenges for Plugin Distribution
However, there are significant obstacles to distributing LLMs within Calibre plugins:
1. Plugin Size Constraints: Calibre plugins are distributed as ZIP files, typically ranging from a few KB to a few MB. The plugin ecosystem through MobileRead Forums has no precedent for 500MB+ plugins.
2. Distribution Infrastructure: The informal distribution through forum posts isn't designed for large files. Users expect quick downloads, not half-gigabyte packages.
3. Memory Requirements: Even the 270M model needs approximately 550MB for FP16 or 170MB for INT4 quantization, plus additional memory for inference operations.
Recommended Implementation Strategy
Two-Stage Deployment Model
Instead of bundling the LLM, implement a progressive enhancement approach:
class RAGSearchPlugin(InterfaceActionBase):
    def __init__(self):
        self.model_manager = ModelManager()
        
    def initialize(self):
        # Stage 1: Fast keyword + embedding search
        self.init_sqlite_vec()  # ~5MB sqlite extension
        self.init_small_embedder()  # ~30MB model like all-MiniLM
        
        # Stage 2: Optional LLM enhancement
        if self.config.get('enable_llm'):
            self.download_llm_async()
Smart Download Strategy
1. Initial Plugin: Ship with:
    * sqlite-vec extension (~5MB)
    * Small embedding model (30-50MB compressed)
    * BM25 + vector search implementation
2. Optional LLM Download:
    * Prompt user to download Gemma 270M on first run
    * Use background thread with progress indicator
    * Store in ~/.calibre/plugins/rag_search/models/
    * Implement resume capability for interrupted downloads
Optimized Architecture for 40k Books
# Efficient indexing for large libraries
class BookIndexer:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.conn.load_extension('vec0')  # sqlite-vec
        
    def create_indexes(self):
        # Hybrid search tables
        self.conn.execute('''
            CREATE VIRTUAL TABLE book_search 
            USING fts5(title, author, tags, description)
        ''')
        
        self.conn.execute('''
            CREATE VIRTUAL TABLE book_vectors 
            USING vec0(embedding FLOAT[384])
        ''')
        
    def batch_index(self, batch_size=100):
        # Process in chunks to handle 40k books
        for offset in range(0, 40000, batch_size):
            books = self.get_books_batch(offset, batch_size)
            embeddings = self.generate_embeddings(books)
            self.store_vectors(embeddings)
Performance Optimizations
For your 40k book library:
1. Incremental Indexing: Only process changed books
2. Binary Quantization: Reduce vector storage by 75%
3. Lazy Loading: Load LLM only when needed for reasoning
4. Cached Results: LRU cache for frequent queries
5. Background Processing: Index updates in separate thread
Free Tier Implementation
To keep it free while managing costs:
1. Local-First: All processing on user's machine
2. Optional Cloud Fallback: Allow users to add their own API keys
3. Hybrid Approach:
    * Fast local search (always free)
    * Optional LLM enhancement (local Gemma 270M)
    * Premium API features (user provides keys)
Technical Implementation Details
Model Loading
def download_gemma_model(self):
    from huggingface_hub import snapshot_download
    
    # Download without authentication
    model_path = snapshot_download(
        repo_id="google/gemma-3-270m-it",  # Instruction-tuned version
        cache_dir=self.model_cache_dir,
        allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
        ignore_patterns=["*.bin", "*.onnx"]  # Skip duplicates
    )
    
    # Load with quantization for efficiency
    self.model = AutoModelForCausalLM.from_pretrained(
        model_path,
        load_in_4bit=True,  # INT4 quantization
        device_map="auto"
    )
Search Pipeline
def hybrid_search(self, query, use_llm=False):
    # Stage 1: Fast retrieval
    bm25_results = self.bm25_search(query)
    vector_results = self.vector_search(query)
    
    # Merge and rerank
    candidates = self.merge_results(bm25_results, vector_results)
    
    # Stage 2: Optional LLM reasoning
    if use_llm and self.llm_available:
        context = self.prepare_context(candidates)
        refined_query = self.llm_reasoning(query, context)
        return self.rerank_with_llm(candidates, refined_query)
    
    return candidates
Conclusion
While technically feasible, distributing a 500MB+ LLM within a Calibre plugin faces practical challenges. The recommended approach is:
1. Ship a lightweight plugin (under 50MB) with fast local search
2. Download LLM on-demand with user consent
3. Use Gemma 3 270M INT4 for best size/performance ratio
4. Implement progressive enhancement - works without LLM, better with it
5. Focus on speed first - your 40k library needs efficient indexing more than LLM reasoning for most queries
This approach gives users a immediately useful, fast search that can be enhanced with LLM capabilities without forcing a massive initial download or compromising the plugin distribution model.

I'll create a comprehensive project plan for your personal Calibre RAG search plugin using the Gemini API. Let me generate the necessary documentation and files to get you started.Perfect! I've created a comprehensive project plan for your personal Calibre RAG search plugin. Here's what I've prepared for you:
📁 Deliverables Created
1. Product Requirements Document (PRD)
A detailed specification covering:
* Problem statement and solution overview
* Phased development plan (6 weeks total)
* Technical requirements and performance targets
* Configuration options and success metrics
* Risk mitigation strategies
2. Calibre Plugin Development Guide
Essential technical information including:
* Complete plugin structure and file organization
* Key Calibre APIs with code examples
* Database access patterns
* GUI integration with PyQt6
* Platform-specific considerations
* Testing and debugging workflows
3. Core Pseudocode
Detailed algorithms for:
* Main search pipeline with Gemini integration
* Efficient indexing system for 40k books
* Vector and keyword search implementation
* Result merging and scoring logic
* Caching strategies
* Error recovery mechanisms
4. claude.md File
Project instructions optimized for Claude Code containing:
* Clear project objectives and structure
* Development phases with checkboxes
* Database schemas and configuration
* Important Calibre APIs reference
* Debugging commands
* Known challenges and solutions
🚀 Getting Started
1. Set up your development environment: # Create project directory
2. mkdir calibre-rag-search
3. cd calibre-rag-search
4. 
5. # Initialize with the structure from claude.md
6. touch __init__.py main.py config.py search_engine.py
7. 
8. Start with the basics (Week 1):
    * Implement the plugin skeleton
    * Get it loading in Calibre
    * Create the search dialog UI
    * Set up SQLite with vector tables
9. Add local search (Week 2):
    * Implement BM25 keyword search
    * Add vector similarity search
    * Create the indexing system
    * Test with 100-1000 books first
10. Integrate Gemini (Week 3):
    * Add API client with rate limiting
    * Implement query enhancement
    * Add intelligent reranking
    * Monitor API costs
💡 Key Design Decisions
* Hybrid Search: 30% keyword (BM25) + 70% semantic (vectors) for optimal results
* Local-First: Everything works without Gemini, API just enhances results
* Incremental Indexing: Only process changed books after initial index
* Smart Caching: Three-tier cache (query, embedding, API) to minimize costs
* Background Processing: All heavy operations use ThreadedJob to keep UI responsive
⚡ Performance Optimizations
For your 40k book library:
* Batch processing in groups of 100 books
* INT8 quantization for embeddings (75% size reduction)
* SQLite transactions for bulk inserts
* Memory cleanup every 1000 books
* Lazy loading with generators
🎯 Next Actions
1. Copy the claude.md file into your Claude Code project
2. Start implementing __init__.py with plugin metadata
3. Create a minimal UI that appears when you press Ctrl+Shift+S
4. Set up the SQLite database with vector tables
5. Test basic integration before adding complex features
The architecture is designed to be modular - you can get basic search working quickly, then progressively add Gemini integration, caching, and advanced features. The system will gracefully degrade if Gemini is unavailable, ensuring you always have functional search.
Would you like me to generate the actual Python code for any specific component to help you get started? I'd recommend beginning with the __init__.py and basic plugin structure.
