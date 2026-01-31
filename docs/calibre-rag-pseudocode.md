# RAG Search Plugin - Core Pseudocode

## Main Search Flow

```pseudocode
FUNCTION search(query, use_gemini=true):
    // Stage 1: Fast local retrieval
    start_time = now()
    
    // Parse and enhance query
    parsed_query = parse_query(query)
    
    // Parallel search
    PARALLEL:
        keyword_results = keyword_search(parsed_query)
        vector_results = vector_search(parsed_query)
    END PARALLEL
    
    // Merge results with weighted scoring
    candidates = merge_results(
        keyword_results * 0.3,
        vector_results * 0.7
    )
    
    // Take top candidates for reranking
    top_candidates = candidates[:100]
    
    // Stage 2: Gemini enhancement (if enabled)
    IF use_gemini AND gemini_available():
        enhanced_results = gemini_rerank(query, top_candidates)
        log_api_usage(query, tokens_used)
        RETURN enhanced_results[:max_results]
    ELSE:
        RETURN top_candidates[:max_results]
    END IF
END FUNCTION
```

## Indexing System

```pseudocode
CLASS BookIndexer:
    FUNCTION index_library(force_rebuild=false):
        IF index_exists() AND NOT force_rebuild:
            RETURN update_incremental()
        END IF
        
        books = get_all_books()
        total = length(books)
        
        // Initialize progress
        show_progress("Indexing library...", 0, total)
        
        // Process in batches for memory efficiency
        batch_size = 100
        FOR batch IN chunks(books, batch_size):
            PARALLEL:
                metadata_batch = extract_metadata(batch)
                embeddings_batch = generate_embeddings(batch)
            END PARALLEL
            
            // Store in database transaction
            BEGIN TRANSACTION:
                store_metadata(metadata_batch)
                store_vectors(embeddings_batch)
                update_fts_index(metadata_batch)
            END TRANSACTION
            
            update_progress(current_position)
            
            // Memory management
            IF processed % 1000 == 0:
                garbage_collect()
            END IF
        END FOR
        
        // Create indices
        create_vector_indices()
        optimize_database()
        
        hide_progress()
        RETURN success
    END FUNCTION
    
    FUNCTION update_incremental():
        last_index_time = get_last_index_time()
        new_books = get_books_modified_after(last_index_time)
        
        IF empty(new_books):
            RETURN no_updates
        END IF
        
        FOR book IN new_books:
            update_single_book(book)
        END FOR
        
        update_last_index_time(now())
        RETURN updated_count
    END FUNCTION
END CLASS
```

## Vector Search Implementation

```pseudocode
FUNCTION vector_search(query):
    // Generate query embedding
    query_embedding = embed_text(query.text)
    
    // Build SQL with vector similarity
    sql = """
        SELECT 
            book_id,
            title,
            authors,
            description,
            vec_distance_l2(combined_embedding, ?1) as distance
        FROM book_vectors
        WHERE combined_embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT ?2
    """
    
    results = execute_query(sql, query_embedding, max_candidates)
    
    // Convert distances to scores (0-1)
    FOR result IN results:
        result.score = 1.0 / (1.0 + result.distance)
    END FOR
    
    RETURN results
END FUNCTION

FUNCTION keyword_search(query):
    // Prepare FTS5 query
    fts_query = prepare_fts_query(query.text)
    
    sql = """
        SELECT 
            book_id,
            title,
            authors,
            description,
            bm25(book_fts) as score
        FROM book_fts
        WHERE book_fts MATCH ?1
        ORDER BY score DESC
        LIMIT ?2
    """
    
    results = execute_query(sql, fts_query, max_candidates)
    
    // Normalize BM25 scores
    max_score = results[0].score IF results ELSE 1.0
    FOR result IN results:
        result.score = result.score / max_score
    END FOR
    
    RETURN results
END FUNCTION
```

## Gemini Integration

```pseudocode
CLASS GeminiClient:
    FUNCTION initialize(api_key):
        self.client = GenerativeAI(api_key)
        self.model = client.get_model('gemini-1.5-flash')
        self.rate_limiter = RateLimiter(60, 60)  // 60 requests per minute
    END FUNCTION
    
    FUNCTION rerank_results(query, candidates):
        // Check cache first
        cache_key = hash(query + candidate_ids)
        IF cached_result = get_from_cache(cache_key):
            RETURN cached_result
        END IF
        
        // Prepare context for Gemini
        context = format_candidates_context(candidates)
        
        prompt = """
        Given this search query: {query}
        
        And these book candidates:
        {context}
        
        Please:
        1. Understand the user's intent
        2. Rerank books by relevance (return top 20 IDs)
        3. Generate a brief explanation for top 5 matches
        4. Suggest related search terms
        
        Return as JSON:
        {
            "ranked_ids": [...],
            "explanations": {...},
            "related_searches": [...]
        }
        """
        
        // Rate limited API call
        WITH rate_limiter.acquire():
            response = model.generate_content(
                prompt.format(query=query, context=context),
                generation_config={
                    'temperature': 0.3,
                    'max_output_tokens': 1000,
                    'response_mime_type': 'application/json'
                }
            )
        END WITH
        
        result = parse_json(response.text)
        
        // Cache result
        cache_result(cache_key, result, ttl=3600)
        
        RETURN result
    END FUNCTION
    
    FUNCTION natural_language_query(question, library_context):
        prompt = """
        You are a librarian assistant for a personal library.
        
        Library statistics:
        {library_context}
        
        User question: {question}
        
        Provide a helpful answer based on the library data.
        Include specific book recommendations if relevant.
        """
        
        response = model.generate_content(
            prompt.format(
                library_context=library_context,
                question=question
            )
        )
        
        RETURN response.text
    END FUNCTION
END CLASS
```

## Embedding Generation

```pseudocode
CLASS EmbeddingGenerator:
    FUNCTION initialize():
        // Use small, fast model for local embeddings
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dimension = 384
        self.cache = LRUCache(max_size=10000)
    END FUNCTION
    
    FUNCTION generate_book_embeddings(book):
        // Create combined text representation
        text_parts = []
        
        // Title with high weight
        text_parts.append(repeat(book.title, 3))
        
        // Authors with medium weight  
        IF book.authors:
            text_parts.append(repeat(join(book.authors), 2))
        END IF
        
        // Tags and series
        IF book.tags:
            text_parts.append(join(book.tags))
        END IF
        
        IF book.series:
            text_parts.append(book.series)
        END IF
        
        // Description (truncated if needed)
        IF book.description:
            desc = clean_html(book.description)
            desc = truncate(desc, max_tokens=500)
            text_parts.append(desc)
        END IF
        
        combined_text = join(text_parts, " ")
        
        // Check cache
        cache_key = hash(combined_text)
        IF cached = self.cache.get(cache_key):
            RETURN cached
        END IF
        
        // Generate embeddings
        embedding = self.model.encode(
            combined_text,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        
        // Cache result
        self.cache.put(cache_key, embedding)
        
        RETURN embedding
    END FUNCTION
    
    FUNCTION batch_encode(texts, batch_size=32):
        embeddings = []
        
        FOR batch IN chunks(texts, batch_size):
            batch_embeddings = self.model.encode(
                batch,
                convert_to_tensor=False,
                normalize_embeddings=True,
                show_progress_bar=False
            )
            embeddings.extend(batch_embeddings)
        END FOR
        
        RETURN embeddings
    END FUNCTION
END CLASS
```

## Result Merging and Scoring

```pseudocode
FUNCTION merge_results(keyword_results, vector_results):
    // Create score map
    scores = {}
    
    // Add keyword scores
    FOR result IN keyword_results:
        scores[result.book_id] = {
            'keyword_score': result.score,
            'vector_score': 0,
            'metadata': result
        }
    END FOR
    
    // Add/update with vector scores
    FOR result IN vector_results:
        IF result.book_id IN scores:
            scores[result.book_id]['vector_score'] = result.score
        ELSE:
            scores[result.book_id] = {
                'keyword_score': 0,
                'vector_score': result.score,
                'metadata': result
            }
        END IF
    END FOR
    
    // Calculate combined scores
    merged = []
    FOR book_id, score_data IN scores:
        combined_score = (
            score_data['keyword_score'] * KEYWORD_WEIGHT +
            score_data['vector_score'] * VECTOR_WEIGHT
        )
        
        // Boost for exact matches
        IF query_matches_title_exactly(score_data['metadata']):
            combined_score *= 1.5
        END IF
        
        merged.append({
            'book_id': book_id,
            'score': combined_score,
            'metadata': score_data['metadata']
        })
    END FOR
    
    // Sort by combined score
    merged.sort(by=score, descending=True)
    
    RETURN merged
END FUNCTION
```

## Caching Strategy

```pseudocode
CLASS SearchCache:
    FUNCTION initialize():
        self.query_cache = LRUCache(max_size=1000)
        self.embedding_cache = DiskCache(
            path='embeddings.cache',
            max_size_mb=500
        )
        self.api_cache = TTLCache(
            max_size=100,
            ttl_seconds=3600
        )
    END FUNCTION
    
    FUNCTION get_cached_search(query, search_type):
        cache_key = hash(query + search_type)
        
        IF result = self.query_cache.get(cache_key):
            // Check if still fresh (< 5 minutes old)
            IF result.age < 300:
                RETURN result
            END IF
        END IF
        
        RETURN null
    END FUNCTION
    
    FUNCTION cache_search_result(query, search_type, results):
        cache_key = hash(query + search_type)
        self.query_cache.put(cache_key, {
            'results': results,
            'timestamp': now()
        })
    END FUNCTION
    
    FUNCTION get_book_embedding(book_id):
        RETURN self.embedding_cache.get(f"book_{book_id}")
    END FUNCTION
    
    FUNCTION cache_book_embedding(book_id, embedding):
        // Compress embedding for storage
        compressed = quantize_int8(embedding)
        self.embedding_cache.put(f"book_{book_id}", compressed)
    END FUNCTION
END CLASS
```

## Query Parser

```pseudocode
FUNCTION parse_query(raw_query):
    parsed = {
        'text': raw_query,
        'filters': {},
        'modifiers': [],
        'intent': 'search'
    }
    
    // Extract filters (author:name, tag:fiction, etc.)
    filter_pattern = /(\w+):("[^"]+"|[^\s]+)/
    FOR match IN find_all(filter_pattern, raw_query):
        field = match[1]
        value = match[2].strip('"')
        parsed['filters'][field] = value
        raw_query = remove(match, raw_query)
    END FOR
    
    // Detect special modifiers
    IF contains(raw_query, "similar to"):
        parsed['modifiers'].append('similarity')
        parsed['intent'] = 'similarity_search'
    END IF
    
    IF contains(raw_query, "but not"):
        parsed['modifiers'].append('exclusion')
    END IF
    
    // Detect question intent
    question_words = ['what', 'which', 'who', 'how many', 'when']
    FOR word IN question_words:
        IF starts_with(lower(raw_query), word):
            parsed['intent'] = 'question'
            BREAK
        END IF
    END FOR
    
    // Clean final text
    parsed['text'] = trim(raw_query)
    
    RETURN parsed
END FUNCTION
```

## Error Recovery

```pseudocode
FUNCTION resilient_search(query):
    try_count = 0
    max_retries = 3
    
    WHILE try_count < max_retries:
        TRY:
            // Try full search pipeline
            results = search(query, use_gemini=true)
            RETURN results
            
        CATCH GeminiError as e:
            log_error("Gemini API failed", e)
            IF try_count == 0:
                // Retry without Gemini
                RETURN search(query, use_gemini=false)
            END IF
            
        CATCH VectorDBError as e:
            log_error("Vector search failed", e)
            IF try_count == 0:
                // Fallback to keyword only
                RETURN keyword_search(query)
            END IF
            
        CATCH Exception as e:
            log_error("Search failed", e)
            try_count += 1
            
            IF try_count >= max_retries:
                // Last resort: basic SQL search
                RETURN basic_sql_search(query)
            END IF
            
            // Exponential backoff
            sleep(2 ** try_count)
        END TRY
    END WHILE
END FUNCTION
```