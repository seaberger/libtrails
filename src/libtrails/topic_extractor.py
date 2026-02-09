"""Topic extraction using local LLM via Ollama or Gemini API."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx

from .config import (
    BATCH_SIZE,
    CHUNK_MODEL,
    DEFAULT_MODEL,
    OLLAMA_NUM_CTX,
    THEME_MODEL,
    TOPIC_STOPLIST,
    TOPICS_PER_CHUNK,
)

# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Reusable HTTP client for connection pooling
_client: Optional[httpx.Client] = None

# Number of parallel workers for topic extraction
# Ollama queues requests, but parallelism helps with HTTP overhead
NUM_WORKERS = 4


def _is_gemini_model(model: str) -> bool:
    """Check if a model string refers to a Gemini API model."""
    return model.startswith("gemini/")


def _get_client() -> httpx.Client:
    """Get or create a reusable HTTP client."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120.0)
    return _client


def _call_gemini(
    prompt: str,
    model: str,
    response_schema: dict | None = None,
    timeout: float = 120.0,
    system_prompt: str | None = None,
) -> str:
    """Call Gemini API via litellm and return the response text.

    Uses response_format for JSON output when a schema is provided.
    Requires GEMINI_API_KEY in environment (loaded from .env by caller).

    When system_prompt is provided, it is sent as a cached system message
    (Gemini context caching — 90% discount on repeated prefix tokens).
    The prompt then becomes the user message with per-call content.
    """
    import litellm

    messages = []
    if system_prompt:
        messages.append({
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        })
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "timeout": timeout,
    }
    if response_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}

    response = litellm.completion(**kwargs)
    return response.choices[0].message.content.strip()



def extract_topics_batch(
    chunks: list[str],
    model: str = DEFAULT_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
    progress_callback: Optional[callable] = None,
) -> list[list[str]]:
    """
    Extract topics from multiple chunks in parallel (legacy single-call mode).

    Returns a list of topic lists, one per chunk.
    """
    results = [None] * len(chunks)
    completed = 0

    def process_chunk(idx: int, text: str) -> tuple[int, list[str]]:
        topics = extract_topics(text, model, num_topics)
        return idx, _filter_topics(topics)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            idx, topics = future.result()
            results[idx] = topics
            completed += 1
            if progress_callback:
                progress_callback(completed, len(chunks))

    return results



def _unwrap_topic(t) -> str:
    """Extract a string from a topic that may be a dict or other type.

    Some models return {"topic": "..."} or {"topic label": "..."} instead
    of plain strings. This unwraps those to the string value.
    """
    if isinstance(t, dict):
        return str(t.get("topic", t.get("topic label", t.get("name", next(iter(t.values()), t)))))
    return str(t)


def normalize_topic(topic: str) -> Optional[str]:
    """
    Normalize a topic label for deduplication.

    Returns None for stoplist matches (generic single-word topics).
    """
    normalized = topic.strip().lower().replace("_", " ")
    # Collapse multiple spaces
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")

    if not normalized:
        return None

    # Filter out generic single-word topics from stoplist
    if normalized in TOPIC_STOPLIST:
        return None

    return normalized


def extract_topics(
    text: str, model: str = DEFAULT_MODEL, num_topics: int = TOPICS_PER_CHUNK
) -> list[str]:
    """
    Extract topics from a text chunk using Ollama HTTP API (legacy mode).

    Returns a list of topic strings.
    """
    prompt = f'''Extract {num_topics} topic labels from this book passage.

Passage: "{text}"'''

    schema = {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": num_topics,
                "maxItems": num_topics,
            }
        },
        "required": ["topics"],
    }

    try:
        client = _get_client()
        response = client.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "format": schema,
                "stream": False,
                "options": {"num_ctx": OLLAMA_NUM_CTX},
            },
        )
        response.raise_for_status()
        output = response.json().get("response", "").strip()
        parsed = json.loads(output)
        if isinstance(parsed, dict) and "topics" in parsed:
            return [_unwrap_topic(t).strip() for t in parsed["topics"] if t][:num_topics]
        return _parse_topics(output)

    except httpx.TimeoutException:
        return []
    except Exception:
        return []


def strip_html(html: str) -> str:
    """Strip HTML tags and clean up whitespace from Calibre descriptions."""
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Remove non-breaking spaces
    text = text.replace("\xa0", " ")
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Tags to remove entirely — pure noise with no topical signal
_NOISE_TAGS = frozenset({
    "general", "fiction", "unknown", "adult", "non-fiction", "nonfiction",
})

# Patterns for compound tag variants
_PREFIX_RE = re.compile(
    r"^(?:fiction|general|non-fiction)\s*[-–/]\s*", re.IGNORECASE
)
_SUFFIX_RE = re.compile(
    r"\s*[-–/]\s*(?:general|fiction)$", re.IGNORECASE
)


def clean_calibre_tags(tags: list[str]) -> list[str]:
    """
    Clean Calibre tags for use in theme extraction prompts.

    1. Remove noise tags ("General", "Fiction", "Unknown", etc.)
    2. Normalize compound variants ("Fiction - Science Fiction" → "Science Fiction",
       "Science Fiction - General" → "Science Fiction")
    3. Remove substring duplicates, keeping the more specific tag
    4. Cap at 10 tags
    """
    if not tags:
        return []

    # Step 1-2: Filter noise and normalize compounds
    cleaned = []
    seen_lower = set()
    for tag in tags:
        if tag.lower().strip() in _NOISE_TAGS:
            continue
        # Strip noise prefixes/suffixes
        normalized = _PREFIX_RE.sub("", tag).strip()
        normalized = _SUFFIX_RE.sub("", normalized).strip()
        if not normalized or normalized.lower() in _NOISE_TAGS:
            continue
        # Deduplicate case-insensitive
        if normalized.lower() not in seen_lower:
            seen_lower.add(normalized.lower())
            cleaned.append(normalized)

    # Step 3: Remove substring duplicates (keep longer/more specific)
    cleaned.sort(key=len, reverse=True)
    result = []
    for tag in cleaned:
        tag_lower = tag.lower()
        if not any(tag_lower in existing.lower() for existing in result):
            result.append(tag)

    # Step 4: Cap at 10
    return result[:10]


def extract_book_themes(
    title: str,
    author: str,
    tags: list[str] | None = None,
    description: str | None = None,
    series: str | None = None,
    sample_text: str | None = None,
    model: str = THEME_MODEL,
) -> list[str]:
    """
    Extract 5-8 high-level themes for a book using a larger model.

    One call per book. Uses title, author, Calibre tags, description,
    series, and the first ~1000 words to produce specific noun-phrase themes.

    Tags are cleaned (noise removed, duplicates collapsed) and descriptions
    are stripped of HTML before inclusion in the prompt.

    Returns a list of theme strings (e.g., "epic fantasy", "magic power systems").
    """
    parts = [f"Title: {title}", f"Author: {author}"]

    if series:
        parts.append(f"Series: {series}")

    if tags:
        cleaned_tags = clean_calibre_tags(tags)
        if cleaned_tags:
            parts.append(f"Tags: {', '.join(cleaned_tags)}")

    if description:
        desc = strip_html(description)[:500].strip()
        if desc:
            parts.append(f"Description: {desc}")

    if sample_text:
        # First ~1000 words
        words = sample_text.split()[:1000]
        parts.append(f"Opening text: {' '.join(words)}")

    book_info = "\n".join(parts)

    prompt = f'''You are a librarian categorizing a book. Based on the information below, extract 5 to 8 high-level themes that describe what this book is about.

Rules:
- Use specific multi-word noun phrases (e.g., "epic fantasy worldbuilding", "algorithmic trading strategies")
- Do NOT use generic single words like "conflict", "power", "love", "death"
- Each theme should be specific enough to distinguish this book from unrelated books
- Include the book's genre/domain as the first theme

{book_info}'''

    schema = {
        "type": "object",
        "properties": {
            "themes": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 5,
                "maxItems": 8,
            }
        },
        "required": ["themes"],
    }

    try:
        if _is_gemini_model(model):
            output = _call_gemini(prompt, model, response_schema=schema)
        else:
            client = _get_client()
            response = client.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": schema,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_NUM_CTX},
                },
                timeout=120.0,
            )
            response.raise_for_status()
            output = response.json().get("response", "").strip()

        parsed = json.loads(output)
        if isinstance(parsed, dict) and "themes" in parsed:
            return [_unwrap_topic(t).strip().lower() for t in parsed["themes"] if t]
        # Fallback to generic parsing
        themes = _parse_topics(output)
        return [t.strip().lower() for t in themes if t.strip()]

    except httpx.TimeoutException:
        return []
    except Exception:
        return []


def extract_topics_batched(
    chunks: list[str],
    book_title: str,
    author: str,
    book_themes: list[str] | None = None,
    model: str = CHUNK_MODEL,
    batch_size: int = BATCH_SIZE,
    num_topics: int = TOPICS_PER_CHUNK,
    progress_callback: Optional[callable] = None,
) -> list[list[str]]:
    """
    Extract topics from chunks in batches, contextualized with book metadata.

    Groups consecutive chunks into batches of `batch_size` and asks for
    per-chunk topic arrays in a single call. Falls back to individual
    extraction if JSON parsing fails for a batch.

    Topics are normalized and filtered through the stoplist before returning.

    Returns list[list[str]] matching input chunk order.
    """
    results: list[list[str]] = [[] for _ in range(len(chunks))]
    completed = 0
    batch_successes = 0
    batch_fallbacks = 0

    # Build context header
    context_parts = [f"Book: {book_title} by {author}"]
    if book_themes:
        context_parts.append(f"Book themes: {', '.join(book_themes)}")
    context = "\n".join(context_parts)

    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]

        batch_results = _extract_batch(
            batch_chunks, context, model, num_topics, batch_start
        )

        if batch_results is not None:
            batch_successes += 1
            for i, topics in enumerate(batch_results):
                results[batch_start + i] = _filter_topics(topics)
        else:
            # Fallback: extract individually
            batch_fallbacks += 1
            for i, chunk in enumerate(batch_chunks):
                topics = _extract_single_contextualized(
                    chunk, context, model, num_topics
                )
                results[batch_start + i] = _filter_topics(topics)

        completed += len(batch_chunks)
        if progress_callback:
            progress_callback(completed, len(chunks))

    # Log batch success/fallback ratio
    total_batches = batch_successes + batch_fallbacks
    if total_batches > 0 and batch_fallbacks > 0:
        print(
            f"  Batch stats: {batch_successes}/{total_batches} succeeded, "
            f"{batch_fallbacks} fell back to individual extraction",
            flush=True,
        )

    return results


def _filter_topics(topics: list[str]) -> list[str]:
    """Normalize topics and filter out stoplist matches."""
    filtered = []
    for topic in topics:
        normalized = normalize_topic(topic)
        if normalized is not None:
            filtered.append(normalized)
    return filtered


def _build_batch_schema(num_chunks: int, num_topics: int) -> dict:
    """Build a JSON schema for batch extraction output."""
    properties = {}
    for i in range(1, num_chunks + 1):
        properties[str(i)] = {
            "type": "array",
            "items": {"type": "string"},
            "minItems": num_topics,
            "maxItems": num_topics,
        }
    return {
        "type": "object",
        "properties": properties,
        "required": [str(i) for i in range(1, num_chunks + 1)],
    }


def _extract_batch(
    chunks: list[str],
    context: str,
    model: str,
    num_topics: int,
    start_index: int,
) -> Optional[list[list[str]]]:
    """
    Extract topics from a batch of chunks in a single LLM call.

    Uses Ollama's structured output (format parameter) to guarantee valid JSON.

    Returns list of topic lists (one per chunk) or None on failure.
    """
    # Build numbered passages
    passages = []
    for i, chunk in enumerate(chunks, 1):
        passages.append(f"--- Passage {i} ---\n{chunk}")

    passages_text = "\n\n".join(passages)

    prompt = f"""{context}

Extract {num_topics} specific topic labels from EACH passage below. Topics should be multi-word noun phrases specific to the content, NOT generic words.

{passages_text}

Return a JSON object mapping passage numbers to topic arrays."""

    schema = _build_batch_schema(len(chunks), num_topics)

    try:
        if _is_gemini_model(model):
            output = _call_gemini(prompt, model, response_schema=schema)
        else:
            client = _get_client()
            response = client.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": schema,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_NUM_CTX},
                },
                timeout=120.0,
            )
            response.raise_for_status()
            output = response.json().get("response", "").strip()

        parsed = json.loads(output)
        if not isinstance(parsed, dict):
            return None

        # Extract topics for each chunk in order
        batch_results = []
        for i in range(1, len(chunks) + 1):
            key = str(i)
            if key in parsed and isinstance(parsed[key], list):
                topics = [str(t).strip() for t in parsed[key] if t]
                batch_results.append(topics[:num_topics])
            else:
                batch_results.append([])

        return batch_results

    except (json.JSONDecodeError, httpx.TimeoutException):
        return None
    except Exception:
        return None


def _extract_single_contextualized(
    text: str,
    context: str,
    model: str,
    num_topics: int,
) -> list[str]:
    """Extract topics from a single chunk with book context (fallback)."""
    prompt = f'''{context}

Extract {num_topics} specific topic labels from this passage. Use multi-word noun phrases, NOT generic single words.

Passage: "{text}"'''

    schema = {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": num_topics,
                "maxItems": num_topics,
            }
        },
        "required": ["topics"],
    }

    try:
        if _is_gemini_model(model):
            output = _call_gemini(prompt, model, response_schema=schema, timeout=60.0)
        else:
            client = _get_client()
            response = client.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": schema,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_NUM_CTX},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            output = response.json().get("response", "").strip()

        parsed = json.loads(output)
        if isinstance(parsed, dict) and "topics" in parsed:
            return [_unwrap_topic(t).strip() for t in parsed["topics"] if t][:num_topics]
        return _parse_topics(output)

    except (httpx.TimeoutException, Exception):
        return []


# ---------------------------------------------------------------------------
# DSPy-optimized few-shot demos (from optimized_topic_extractor.json)
# ---------------------------------------------------------------------------

_DSPY_INSTRUCTION = (
    "Extract specific topic labels from a book passage.\n\n"
    "Given a passage from a book along with the book's title, author, and\n"
    "high-level themes, extract exactly 5 topic labels that capture the\n"
    "key concepts discussed in this specific passage."
)

_DSPY_DEMOS = [
    {
        "passage": (
            "A beginning is the time for taking the most delicate care that the balances "
            "are correct. This every sister of the Bene Gesserit knows. To begin your study "
            "of the life of Muad'Dib, then, take care that you first place him in his time: "
            "born in the 57th year of the Padishah Emperor, Shaddam IV. And take the most "
            "special care that you locate Muad'Dib in his place: the planet Arrakis. Do not "
            "be deceived by the fact that he was born on Caladan and lived his first fifteen "
            "years there. Arrakis, the planet known as Dune, is forever his place."
        ),
        "book_context": (
            "Book: Dune by Frank Herbert\n"
            "Book themes: desert ecology and survival, political intrigue and feudalism, "
            "prescient abilities and fate, spice melange economy, "
            "fremen culture and resistance, messianic prophecy"
        ),
        "topics": [
            "muad'dib's planetary origin",
            "dune as muad'dib's true home",
            "temporal placement of muad'dib",
            "caladan as a temporary location",
            "bene gesserit strategic placement",
        ],
    },
    {
        "passage": (
            "Siddhartha had started to nurse discontent in himself, he had started to feel "
            "that the love of his father and the love of his mother, and also the love of his "
            "friend, Govinda, would not bring him joy for ever and ever, would not nurse him, "
            "feed him, satisfy him. He had started to suspect that his venerable father and "
            "his other teachers, that the wise Brahmans had already revealed to him the most "
            "and best of their wisdom, that they had already filled his expecting vessel with "
            "their richness, and the vessel was not full, the spirit was not content, the soul "
            "was not calm, the heart was not satisfied. The ablutions were good, but they were "
            "water, they did not wash off the sin, they did not heal the spirit's thirst, "
            "they did not relieve the fear in his heart."
        ),
        "book_context": (
            "Book: Siddhartha by Hermann Hesse\n"
            "Book themes: indian religious philosophy, spiritual self-discovery, "
            "renunciation and worldly life, brahmanical social structure, "
            "the search for enlightenment, early buddhist influences"
        ),
        "topics": [
            "dissatisfaction with conventional wisdom",
            "unfulfilled spiritual yearning",
            "water's inadequacy as a solution",
            "fear and anxiety within the soul",
            "seeking beyond brahmanical teachings",
        ],
    },
    {
        "passage": (
            "What is meditation? What is leaving one's body? What is fasting? What is holding "
            "one's breath? It is fleeing from the self, it is a short escape of the agony of "
            "being a self, it is a short numbing of the senses against the pain and the "
            "pointlessness of life. The same escape, the same short numbing is what the driver "
            "of an ox-cart finds in the inn, drinking a few bowls of rice-wine or fermented "
            "coconut-milk. Then he won't feel his self any more, then he won't feel the pains "
            "of life any more, then he finds a short numbing of the senses."
        ),
        "book_context": (
            "Book: Siddhartha by Hermann Hesse\n"
            "Book themes: indian religious philosophy, spiritual self-discovery, "
            "renunciation and worldly life, brahmanical social structure, "
            "the search for enlightenment, early buddhist influences"
        ),
        "topics": [
            "self-escape mechanisms",
            "sensory numbing",
            "temporary relief from pain",
            "ox-cart driver's distraction",
            "avoidance of self-awareness",
        ],
    },
    {
        "passage": (
            "mind to be void of all conceptions. These and other ways he learned to go, "
            "a thousand times he left his self, for hours and days he remained in the non-self. "
            "But though the ways led away from the self, their end nevertheless always led back "
            "to the self. Though Siddhartha fled from the self a thousand times, stayed in "
            "nothingness, stayed in the animal, in the stone, the return was inevitable, "
            "inescapable was the hour, when he found himself back in the sunshine or in the "
            "moonlight, in the shade or in the rain, and was once again his self and Siddhartha, "
            "and again felt the agony of the cycle which had been forced upon him."
        ),
        "book_context": (
            "Book: Siddhartha by Hermann Hesse\n"
            "Book themes: indian religious philosophy, spiritual self-discovery, "
            "renunciation and worldly life, brahmanical social structure, "
            "the search for enlightenment, early buddhist influences"
        ),
        "topics": [
            "samana meditation practices",
            "escape from self",
            "inescapable return to selfhood",
            "agony of cyclical existence",
            "non-self experience",
        ],
    },
]


def extract_topics_single_optimized(
    text: str,
    context: str,
    model: str = CHUNK_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
) -> list[str]:
    """
    Extract topics from a single chunk using DSPy-optimized instruction + few-shot demos.

    Uses the instruction and 4 demonstrations produced by MIPROv2 optimization.
    Each demo shows the model a passage + book context → 5 multi-word noun-phrase topics.

    When using Gemini, the instruction + demos are sent as a cached system message
    (context caching — 90% discount on repeated tokens). The per-chunk passage and
    book context are sent as the user message.
    """
    # Build few-shot section
    demo_parts = []
    for demo in _DSPY_DEMOS:
        demo_topics = json.dumps(demo["topics"])
        demo_parts.append(
            f"---\n"
            f"Passage: {demo['passage']}\n\n"
            f"Book Context: {demo['book_context']}\n\n"
            f"Topics: {demo_topics}"
        )
    demos_text = "\n\n".join(demo_parts)

    schema = {
        "type": "object",
        "properties": {
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": num_topics,
                "maxItems": num_topics,
            }
        },
        "required": ["topics"],
    }

    try:
        if _is_gemini_model(model):
            # Split into cached system prompt (instruction + demos) and user prompt (chunk)
            system_prompt = f"{_DSPY_INSTRUCTION}\n\n{demos_text}"
            user_prompt = f"---\nPassage: {text}\n\nBook Context: {context}\n\nTopics:"
            output = _call_gemini(
                user_prompt, model,
                response_schema=schema, timeout=60.0,
                system_prompt=system_prompt,
            )
        else:
            prompt = f"""{_DSPY_INSTRUCTION}\n\n{demos_text}\n\n---\nPassage: {text}\n\nBook Context: {context}\n\nTopics:"""
            client = _get_client()
            response = client.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": schema,
                    "stream": False,
                    "options": {"num_ctx": OLLAMA_NUM_CTX},
                },
                timeout=60.0,
            )
            response.raise_for_status()
            output = response.json().get("response", "").strip()

        parsed = json.loads(output)
        if isinstance(parsed, dict) and "topics" in parsed:
            return [_unwrap_topic(t).strip() for t in parsed["topics"] if t][:num_topics]
        return _parse_topics(output)

    except (httpx.TimeoutException, Exception):
        return []


def extract_topics_single_optimized_parallel(
    chunks: list[str],
    book_title: str,
    author: str,
    book_themes: list[str] | None = None,
    model: str = CHUNK_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
    max_workers: int = 30,
    progress_callback: Optional[callable] = None,
) -> list[list[str]]:
    """
    Extract topics from chunks in parallel using DSPy-optimized single-chunk extraction.

    Each chunk is processed independently with the DSPy instruction + few-shot demos.
    Uses ThreadPoolExecutor with configurable concurrency to stay within API rate limits.

    For Gemini paid tier (~300 RPM on Flash-Lite), max_workers=30 is safe.

    Returns list[list[str]] matching input chunk order.
    """
    results: list[list[str] | None] = [None] * len(chunks)
    completed = 0

    # Build context header
    context_parts = [f"Book: {book_title} by {author}"]
    if book_themes:
        context_parts.append(f"Book themes: {', '.join(book_themes)}")
    context = "\n".join(context_parts)

    def process_chunk(idx: int, text: str) -> tuple[int, list[str]]:
        raw = extract_topics_single_optimized(text, context, model, num_topics)
        return idx, _filter_topics(raw)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_chunk, i, chunk): i
            for i, chunk in enumerate(chunks)
        }

        for future in as_completed(futures):
            idx, topics = future.result()
            results[idx] = topics
            completed += 1
            if progress_callback:
                progress_callback(completed, len(chunks))

    # Replace any None entries (shouldn't happen, but safety)
    return [r if r is not None else [] for r in results]


def consolidate_book_topics(
    topics: list[str],
    book_title: str,
    author: str,
    book_themes: list[str],
    model: str = THEME_MODEL,
) -> dict[str, list[str]]:
    """
    Pass 3: Consolidate raw chunk topics using a larger model.

    Takes the ~400 raw topics from Pass 2 and returns a mapping of
    consolidated label → list of original topics that merge into it.

    The model merges near-duplicates, removes noise, and drops
    verbatim theme echoes. Topics not in the output are discarded.

    Returns dict like {"river as spiritual teacher": ["river as a teacher",
    "river as a source of wisdom", "the river's wisdom"]}.
    """
    if not topics:
        return {}

    # Deduplicate input list preserving order
    seen = set()
    unique_topics = []
    for t in topics:
        key = t.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_topics.append(key)

    themes_str = ", ".join(book_themes) if book_themes else "none"
    topics_str = "\n".join(f"- {t}" for t in unique_topics)
    target_min = int(len(unique_topics) * 0.55)
    target_max = int(len(unique_topics) * 0.70)

    prompt = f"""You are deduplicating a topic index for "{book_title}" by {author}.

Below are {len(unique_topics)} topics. Your job is to MERGE only obvious near-duplicates and REMOVE only clear noise. Most topics should survive unchanged.

**Book themes** (remove these verbatim if they appear):
{themes_str}

**Topics:**
{topics_str}

**Rules:**
1. MERGE topics that are clearly the same concept with different wording:
   - "river as a teacher" + "river as a source of wisdom" → pick one
   - "spiritual awakening" + "spiritual enlightenment" → pick one
   - "loss of identity" + "loss of innocence" → these are DIFFERENT concepts, do NOT merge
2. REMOVE only:
   - Exact copies of book themes listed above
   - Trivial 1-2 word plot details that are meaningless on their own
   - Sentence-length descriptions (7+ words that read like a summary)
3. KEEP everything else unchanged — when in doubt, keep it
4. Target: {target_min} to {target_max} consolidated topics (you have {len(unique_topics)} now)
5. Every output key must map to its list of original topic strings that merged into it

Return a JSON object where each key is the chosen canonical label and each value is the array of original topics it represents (including itself if kept as-is)."""

    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "array",
            "items": {"type": "string"},
        },
    }

    # Use larger context for consolidation (input + output can be large)
    consolidation_ctx = max(OLLAMA_NUM_CTX, 16384)

    try:
        if _is_gemini_model(model):
            output = _call_gemini(prompt, model, response_schema=schema, timeout=300.0)
        else:
            client = _get_client()
            response = client.post(
                OLLAMA_API_URL,
                json={
                    "model": model,
                    "prompt": prompt,
                    "format": schema,
                    "stream": False,
                    "options": {"num_ctx": consolidation_ctx},
                },
                timeout=300.0,
            )
            response.raise_for_status()
            output = response.json().get("response", "").strip()

        parsed = json.loads(output)

        if not isinstance(parsed, dict):
            return {}

        # Normalize keys and validate values
        result: dict[str, list[str]] = {}
        for label, originals in parsed.items():
            clean_label = label.strip().lower()
            if not clean_label:
                continue
            if isinstance(originals, list):
                clean_originals = [str(o).strip().lower() for o in originals if o]
                if clean_originals:
                    result[clean_label] = clean_originals

        return result

    except httpx.TimeoutException:
        return {}
    except Exception:
        return {}


def _parse_topics(output: str) -> list[str]:
    """Parse LLM output to extract topic list."""
    output = output.strip()

    # Try to parse as JSON array
    try:
        # Find JSON array in output
        match = re.search(r"\[.*?\]", output, re.DOTALL)
        if match:
            topics = json.loads(match.group())
            if isinstance(topics, list):
                return [str(t).strip() for t in topics if t]
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract quoted strings
    quoted = re.findall(r'"([^"]+)"', output)
    if quoted:
        return quoted[:TOPICS_PER_CHUNK]

    # Last resort: split by commas or newlines
    if "," in output:
        parts = [p.strip().strip("\"'") for p in output.split(",")]
        return [p for p in parts if p and len(p) < 50][:TOPICS_PER_CHUNK]

    return []


def check_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is available and model is loaded."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        return model.split(":")[0] in result.stdout
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            return [line.split()[0] for line in lines if line]
    except Exception:
        pass
    return []
