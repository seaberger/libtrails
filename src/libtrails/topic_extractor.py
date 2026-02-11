"""Topic extraction using local LLM via Ollama, LM Studio, or Gemini API."""

import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

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


class ContentFilterError(Exception):
    """Raised when Gemini's content filter blocks a response."""


# Ollama API endpoint
OLLAMA_API_URL = "http://localhost:11434/api/generate"

# Reusable HTTP client for connection pooling
_client: Optional[httpx.Client] = None

# Number of parallel workers for topic extraction
# Ollama queues requests, but parallelism helps with HTTP overhead
NUM_WORKERS = 4

# Per-model LM Studio API base URL overrides (e.g., for routing to a remote GPU)
# Keys are model strings like "lm_studio/google/gemma-3-12b"
# Registered at runtime by CLI from config/env/flags via set_lm_studio_api_base().
_lm_studio_api_bases: dict[str, str] = {}


def set_lm_studio_api_base(model: str, api_base: str) -> None:
    """Register a custom LM Studio API base URL for a specific model.

    Allows routing different models to different LM Studio instances,
    e.g., theme model on local Mac, chunk model on a remote GPU.

    The api_base should be the server root (e.g., http://192.168.1.36:1234).
    The /v1 suffix is added automatically.
    """
    if not api_base.endswith("/v1"):
        api_base = api_base.rstrip("/") + "/v1"
    _lm_studio_api_bases[model] = api_base


def _get_lm_studio_api_base(model: str) -> str:
    """Look up the API base URL for an LM Studio model."""
    return _lm_studio_api_bases.get(model, "http://localhost:1234/v1")


def _is_litellm_model(model: str) -> bool:
    """Check if a model string should be routed through litellm.

    Matches Gemini API models (gemini/) and LM Studio MLX models (lm_studio/).
    """
    return model.startswith("gemini/") or model.startswith("lm_studio/")


def _get_client() -> httpx.Client:
    """Get or create a reusable HTTP client."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=120.0)
    return _client


def _extract_json_from_text(text: str) -> str:
    """Extract JSON from LM Studio responses that may contain prose or markdown.

    Tries in order: direct parse, code fence extraction, brace extraction.
    Returns the JSON string or the original text if no JSON found.
    """
    # 1. Already valid JSON
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Markdown code fences: ```json\n{...}\n```
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. First JSON object in the text
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            pass

    return text


def _call_litellm(
    prompt: str,
    model: str,
    response_schema: dict | None = None,
    timeout: float = 120.0,
    system_prompt: str | None = None,
    cache_system_prompt: bool = False,
) -> str:
    """Call an LLM via litellm and return the response text.

    Supports Gemini API (gemini/...) and LM Studio (lm_studio/...) models.
    Uses response_format for JSON output when a schema is provided.

    When system_prompt is provided, it is sent as a system message
    and prompt becomes the user message.

    When cache_system_prompt=True, the system message is marked with
    cache_control for Gemini context caching (90% discount on cached tokens).
    Only use this when the system prompt exceeds 2048 tokens.
    """
    import litellm

    litellm.suppress_debug_info = True

    messages = []
    if system_prompt:
        if cache_system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
        "timeout": timeout,
    }
    is_lm_studio = model.startswith("lm_studio/")
    if is_lm_studio:
        kwargs["api_base"] = _get_lm_studio_api_base(model)
    if response_schema is not None:
        if is_lm_studio:
            # LM Studio rejects json_object; use json_schema with strict mode
            # to activate Outlines regex-based token masking for MLX models
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "strict": True, "schema": response_schema},
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}

    response = litellm.completion(**kwargs)
    content = response.choices[0].message.content

    # Gemini's content filter returns None content with finish_reason="content_filter"
    # when response_format=json_object is used on sensitive passages.
    # Retry without response_format as fallback.
    if content is None and response_schema is not None:
        del kwargs["response_format"]
        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content

    if content is None:
        raise ContentFilterError("Gemini returned empty content (content_filter)")

    content = content.strip()

    # LM Studio strict mode may still produce malformed JSON in edge cases.
    # Extract JSON from: code fences, embedded JSON, or plain-text lists.
    if is_lm_studio and response_schema is not None:
        content = _extract_json_from_text(content)

    return content


def extract_topics_batch(
    chunks: list[str],
    model: str = DEFAULT_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
    progress_callback: Optional[Callable] = None,
    save_callback: Optional[Callable[[int, list[str]], None]] = None,
) -> list[list[str]]:
    """
    Extract topics from multiple chunks in parallel (legacy single-call mode).

    Returns a list of topic lists, one per chunk.
    If save_callback is provided, calls save_callback(chunk_index, topics) as each chunk completes.
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
            if save_callback:
                save_callback(idx, topics)
            if progress_callback:
                progress_callback(completed, len(chunks))

    return results


def _unwrap_topic(t) -> str:
    """Extract a string from a topic that may be a dict or other type.

    Some models return {"topic": "..."} or {"topic label": "..."} instead
    of plain strings. Also handles string-encoded dicts like
    "{'topic extraction': 'value'}" that JSON parsing leaves as strings.
    """
    if isinstance(t, dict):
        return str(t.get("topic", t.get("topic label", t.get("name", next(iter(t.values()), t)))))
    s = str(t)
    # Handle string representations of Python dicts: "{'key': 'value'}"
    # Use regex extraction instead of eval for safety
    if s.startswith("{") and s.endswith("}"):
        import re

        # Extract value after the colon — handles apostrophes in values
        # Matches: {'any key': 'value with apostrophe's'} or {"key": "value"}
        m = re.search(r""":\s*['"](.+)['"]\s*\}$""", s)
        if m:
            return m.group(1)
    return s


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
_NOISE_TAGS = frozenset(
    {
        "general",
        "fiction",
        "unknown",
        "adult",
        "non-fiction",
        "nonfiction",
    }
)

# Patterns for compound tag variants
_PREFIX_RE = re.compile(r"^(?:fiction|general|non-fiction)\s*[-–/]\s*", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"\s*[-–/]\s*(?:general|fiction)$", re.IGNORECASE)


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

    prompt = f"""You are a librarian categorizing a book. Based on the information below, extract 5 to 8 high-level themes that describe what this book is about.

Rules:
- Use specific multi-word noun phrases (e.g., "epic fantasy worldbuilding", "algorithmic trading strategies")
- Do NOT use generic single words like "conflict", "power", "love", "death"
- Each theme should be specific enough to distinguish this book from unrelated books
- Include the book's genre/domain as the first theme

{book_info}"""

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
        if _is_litellm_model(model):
            output = _call_litellm(prompt, model, response_schema=schema)
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
            return [
                _unwrap_topic(t).strip().lower()
                for t in parsed["themes"]
                if t and _unwrap_topic(t).strip().lower() != "themes"
            ]
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
    progress_callback: Optional[Callable] = None,
    save_callback: Optional[Callable[[int, list[str]], None]] = None,
) -> list[list[str]]:
    """
    Extract topics from chunks in batches, contextualized with book metadata.

    Groups consecutive chunks into batches of `batch_size` and asks for
    per-chunk topic arrays in a single call. Falls back to individual
    extraction if JSON parsing fails for a batch.

    Topics are normalized and filtered through the stoplist before returning.

    Returns list[list[str]] matching input chunk order.
    If save_callback is provided, calls save_callback(chunk_index, topics) as each chunk completes.
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

        batch_results = _extract_batch(batch_chunks, context, model, num_topics, batch_start)

        if batch_results is not None:
            batch_successes += 1
            for i, topics in enumerate(batch_results):
                filtered = _filter_topics(topics)
                results[batch_start + i] = filtered
                if save_callback:
                    save_callback(batch_start + i, filtered)
        else:
            # Fallback: extract individually
            batch_fallbacks += 1
            for i, chunk in enumerate(batch_chunks):
                topics = _extract_single_contextualized(chunk, context, model, num_topics)
                filtered = _filter_topics(topics)
                results[batch_start + i] = filtered
                if save_callback:
                    save_callback(batch_start + i, filtered)

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
        if _is_litellm_model(model):
            output = _call_litellm(prompt, model, response_schema=schema)
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
                return None  # incomplete batch — trigger per-chunk fallback

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
        if _is_litellm_model(model):
            output = _call_litellm(prompt, model, response_schema=schema, timeout=60.0)
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

# Extended instruction with quality guidelines for the cached prompt variant.
_DSPY_INSTRUCTION_EXTENDED = (
    "Extract specific topic labels from a book passage.\n\n"
    "Given a passage from a book along with the book's title, author, and\n"
    "high-level themes, extract exactly 5 topic labels that capture the\n"
    "key concepts discussed in this specific passage.\n\n"
    "Guidelines:\n"
    "- Each topic must be a multi-word noun phrase (3-6 words ideal).\n"
    "- Topics should be specific to this passage, not generic labels.\n"
    "- Avoid single-word topics or vague terms like 'conflict' or 'emotion'.\n"
    "- Ground topics in the passage's concrete details: names, places, events.\n"
    '- Return topics as a JSON object: {"topics": ["topic1", "topic2", ...]}.\n'
    "- Each topic must be a plain string, never a dict or nested object.\n"
    "- Use the book context to disambiguate: 'river journey' in Siddhartha\n"
    "  is spiritual, in Huckleberry Finn is geographical.\n"
    '- Do not wrap topics in dicts like {"topic": "value"}. Return plain strings only.\n'
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

# Extended demos for Gemini context caching (need 2048+ tokens in system prompt).
# These supplement _DSPY_DEMOS with 5 more examples from diverse genres.
_DSPY_DEMOS_EXTENDED = [
    {
        "passage": (
            "Economists said it was the boll weevil that tore through the cotton fields "
            "and left them without work and in even greater misery, which likely gave "
            "hard-bitten sharecroppers just one more reason to go. Still, many of them "
            "picked cotton not because it was their preference but because it was the "
            "only work allowed them in the cotton-growing states. In South Carolina, "
            "colored people had to apply for a permit to do any work other than "
            "agriculture after Reconstruction. It would not likely have been their "
            "choice had there been an alternative."
        ),
        "book_context": (
            "Book: The Warmth of Other Suns by Isabel Wilkerson\n"
            "Book themes: the great migration of african americans, jim crow era "
            "social history, demographic shifts in american cities"
        ),
        "topics": [
            "boll weevil's impact on cotton sharecroppers",
            "restrictions on black labor post-reconstruction",
            "great migration as a long-term demographic shift",
            "migrant motivations for leaving the south",
            "post-world war i acceleration of migration",
        ],
    },
    {
        "passage": (
            "He stopped. I was about to demand that he be more specific, but he said, "
            "'Goodbye, Mr. Ai,' turned, and left. I stood benumbed. The man was like "
            "an electric shock — nothing to hold on to and you don't know what hit you. "
            "He had certainly spoiled the mood of peaceful self-congratulation in which "
            "I had eaten breakfast. I went to the narrow window and looked out. The snow "
            "had thinned a little. It was bitterly cold; the window, facing south, was "
            "iced over on the outside."
        ),
        "book_context": (
            "Book: The Left Hand of Darkness by Ursula K. le Guin\n"
            "Book themes: anthropological science fiction, androgynous alien biology, "
            "interplanetary diplomatic relations, subarctic planetary survival"
        ),
        "topics": [
            "feelings of isolation and distrust",
            "harsh winter conditions",
            "nostalgia for home planet",
            "political espionage and unknown factions",
            "sudden departure and confusion",
        ],
    },
    {
        "passage": (
            "Startups must attempt to tune the engine from the baseline toward the "
            "ideal. This may take many attempts. After the startup has made all the "
            "micro changes and product optimizations it can to move its baseline "
            "toward the ideal, the company reaches a decision point. That is the "
            "third step: pivot or persevere. If the company is making good progress "
            "toward the ideal, that means it is learning appropriately and using that "
            "learning effectively, in which case it makes sense to continue."
        ),
        "book_context": (
            "Book: The Lean Startup by Eric Ries\n"
            "Book themes: startup business methodology, preventing startup failure, "
            "scientific entrepreneurial process, validated learning and customer feedback"
        ),
        "topics": [
            "establishing a baseline with mvps",
            "pivot or persevere decision point",
            "testing risky assumptions first",
            "tuning the engine toward an ideal",
            "validated learning through experimentation",
        ],
    },
    {
        "passage": (
            "Then I went to the bedroom to look for Mackerel. The cat was curled up "
            "under the quilt, sound asleep. I peeled back the quilt and took the cat's "
            "tail in my hand to study its shape. I ran my fingers over it, trying to "
            "recall the exact angle of the bent tip, when the cat gave an annoyed "
            "stretch and went back to sleep. I could no longer say for sure that this "
            "was the same exact tail the cat had had when we first got it."
        ),
        "book_context": (
            "Book: The Wind-Up Bird Chronicle by Haruki Murakami\n"
            "Book themes: japanese magical realism, disintegrating marital "
            "relationships, subterranean metaphysical exploration"
        ),
        "topics": [
            "cat tail as symbol of identity",
            "dream logic and symbolism",
            "letter from lieutenant mamiya and delayed communication",
            "memory and perception of physical objects",
            "supermarket and domestic routines",
        ],
    },
    {
        "passage": (
            "The book is divided into five parts. Part 1 presents the basic elements "
            "of a two-systems approach to judgment and choice. It elaborates the "
            "distinction between the automatic operations of System 1 and the "
            "controlled operations of System 2, and shows how associative memory, "
            "the core of System 1, continually constructs a coherent interpretation "
            "of what is going on in our world at any instant."
        ),
        "book_context": (
            "Book: Thinking, Fast and Slow by Daniel Kahneman\n"
            "Book themes: behavioral economics and cognitive psychology, "
            "dual-process model of human cognition, cognitive heuristics "
            "and systematic biases, critique of rational choice theory"
        ),
        "topics": [
            "deviations from rationality in decision-making",
            "dual-process model of cognition",
            "heuristics and biases in judgment",
            "overconfidence and uncertainty",
            "system 1 vs system 2 thinking",
        ],
    },
]


def extract_topics_single_optimized(
    text: str,
    context: str,
    model: str = CHUNK_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
    use_extended_prompt: bool = False,
) -> list[str]:
    """
    Extract topics from a single chunk using DSPy-optimized instruction + few-shot demos.

    Uses the instruction and demonstrations produced by MIPROv2 optimization.
    Each demo shows the model a passage + book context -> 5 multi-word noun-phrase topics.

    When use_extended_prompt=True, uses 9 demos + detailed instruction (2090+ tokens)
    which enables Gemini context caching for a 90% discount on the system prompt.
    """
    # Select instruction and demos based on prompt variant
    if use_extended_prompt:
        instruction = _DSPY_INSTRUCTION_EXTENDED
        demos = _DSPY_DEMOS + _DSPY_DEMOS_EXTENDED
    else:
        instruction = _DSPY_INSTRUCTION
        demos = _DSPY_DEMOS

    # Build few-shot section
    demo_parts = []
    for demo in demos:
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

    max_retries = 3
    for attempt in range(max_retries):
        try:
            if _is_litellm_model(model):
                system_prompt = f"{instruction}\n\n{demos_text}"
                user_prompt = f"---\nPassage: {text}\n\nBook Context: {context}\n\nTopics:"
                output = _call_litellm(
                    user_prompt,
                    model,
                    response_schema=schema,
                    timeout=60.0,
                    system_prompt=system_prompt,
                    cache_system_prompt=use_extended_prompt,
                )
            else:
                prompt = f"""{instruction}\n\n{demos_text}\n\n---\nPassage: {text}\n\nBook Context: {context}\n\nTopics:"""
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

            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict) and "topics" in parsed:
                    return [_unwrap_topic(t).strip() for t in parsed["topics"] if t][:num_topics]
            except json.JSONDecodeError:
                pass
            # Fallback: parse plain text response (e.g., from content_filter retry)
            return _parse_topics(output)

        except ContentFilterError:
            return []  # Don't retry — content filter is deterministic
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2**attempt)  # 1s, 2s backoff
                continue
            return []


def extract_topics_single_optimized_parallel(
    chunks: list[str],
    book_title: str,
    author: str,
    book_themes: list[str] | None = None,
    model: str = CHUNK_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
    max_workers: int = 30,
    progress_callback: Optional[Callable] = None,
    save_callback: Optional[Callable[[int, list[str]], None]] = None,
    use_extended_prompt: bool = False,
) -> list[list[str]]:
    """
    Extract topics from chunks in parallel using DSPy-optimized single-chunk extraction.

    Each chunk is processed independently with the DSPy instruction + few-shot demos.
    Uses ThreadPoolExecutor with configurable concurrency to stay within API rate limits.

    When use_extended_prompt=True, uses 9 demos + detailed instruction (2090+ tokens)
    enabling Gemini context caching for a 90% discount on the system prompt.

    Returns list[list[str]] matching input chunk order.
    If save_callback is provided, calls save_callback(chunk_index, topics) as each chunk completes.
    """
    results: list[list[str] | None] = [None] * len(chunks)
    completed = 0

    # Build context header
    context_parts = [f"Book: {book_title} by {author}"]
    if book_themes:
        context_parts.append(f"Book themes: {', '.join(book_themes)}")
    context = "\n".join(context_parts)

    def process_chunk(idx: int, text: str) -> tuple[int, list[str]]:
        raw = extract_topics_single_optimized(
            text,
            context,
            model,
            num_topics,
            use_extended_prompt=use_extended_prompt,
        )
        return idx, _filter_topics(raw)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_chunk, i, chunk): i for i, chunk in enumerate(chunks)}

        for future in as_completed(futures):
            idx, topics = future.result()
            results[idx] = topics
            completed += 1
            if save_callback:
                save_callback(idx, topics)
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
        if _is_litellm_model(model):
            output = _call_litellm(prompt, model, response_schema=schema, timeout=300.0)
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
