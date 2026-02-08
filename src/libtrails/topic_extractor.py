"""Topic extraction using local LLM via Ollama."""

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


def _get_client() -> httpx.Client:
    """Get or create a reusable HTTP client."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120.0)
    return _client


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
            return [str(t).strip() for t in parsed["topics"] if t][:num_topics]
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
            return [str(t).strip().lower() for t in parsed["themes"] if t]
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

    prompt = f'''{context}

Extract {num_topics} specific topic labels from EACH passage below. Topics should be multi-word noun phrases specific to the content, NOT generic words.

{passages_text}

Return a JSON object mapping passage numbers to topic arrays.'''

    schema = _build_batch_schema(len(chunks), num_topics)

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
            return [str(t).strip() for t in parsed["topics"] if t][:num_topics]
        return _parse_topics(output)

    except (httpx.TimeoutException, Exception):
        return []


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
