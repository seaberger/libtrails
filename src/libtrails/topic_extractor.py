"""Topic extraction using local LLM via Ollama."""

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import httpx

from .config import DEFAULT_MODEL, TOPICS_PER_CHUNK

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
        _client = httpx.Client(timeout=60.0)
    return _client


def extract_topics_batch(
    chunks: list[str],
    model: str = DEFAULT_MODEL,
    num_topics: int = TOPICS_PER_CHUNK,
    progress_callback: Optional[callable] = None
) -> list[list[str]]:
    """
    Extract topics from multiple chunks in parallel.

    Returns a list of topic lists, one per chunk.
    """
    results = [None] * len(chunks)
    completed = 0

    def process_chunk(idx: int, text: str) -> tuple[int, list[str]]:
        topics = extract_topics(text, model, num_topics)
        return idx, topics

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
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

    return results



def normalize_topic(topic: str) -> str:
    """
    Normalize a topic label for deduplication.

    - Strips whitespace
    - Converts to lowercase
    - Replaces underscores with spaces
    - Collapses multiple spaces
    """
    normalized = topic.strip().lower().replace("_", " ")
    # Collapse multiple spaces
    while "  " in normalized:
        normalized = normalized.replace("  ", " ")
    return normalized


def extract_topics(
    text: str,
    model: str = DEFAULT_MODEL,
    num_topics: int = TOPICS_PER_CHUNK
) -> list[str]:
    """
    Extract topics from a text chunk using Ollama HTTP API.

    Returns a list of topic strings.
    """
    prompt = f'''Extract {num_topics} topic labels from this book passage.
Return ONLY a JSON array of topic strings, no other text.

Passage: "{text[:2000]}"

Topics:'''

    try:
        client = _get_client()
        response = client.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
        )
        response.raise_for_status()
        result = response.json()
        return _parse_topics(result.get("response", ""))

    except httpx.TimeoutException:
        return []
    except Exception:
        return []


def _parse_topics(output: str) -> list[str]:
    """Parse LLM output to extract topic list."""
    output = output.strip()

    # Try to parse as JSON array
    try:
        # Find JSON array in output
        match = re.search(r'\[.*?\]', output, re.DOTALL)
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
    if ',' in output:
        parts = [p.strip().strip('"\'') for p in output.split(',')]
        return [p for p in parts if p and len(p) < 50][:TOPICS_PER_CHUNK]

    return []


def check_ollama_available(model: str = DEFAULT_MODEL) -> bool:
    """Check if Ollama is available and model is loaded."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return model.split(':')[0] in result.stdout
    except Exception:
        return False


def get_available_models() -> list[str]:
    """Get list of available Ollama models."""
    try:
        result = subprocess.run(
            ['ollama', 'list'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')[1:]  # Skip header
            return [line.split()[0] for line in lines if line]
    except Exception:
        pass
    return []
