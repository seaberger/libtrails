"""Topic extraction using local LLM via Ollama."""

import subprocess
import json
import re
from typing import Optional

from .config import DEFAULT_MODEL, TOPICS_PER_CHUNK



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
    Extract topics from a text chunk using Ollama.

    Returns a list of topic strings.
    """
    prompt = f'''Extract {num_topics} topic labels from this book passage.
Return ONLY a JSON array of topic strings, no other text.

Passage: "{text[:2000]}"

Topics:'''

    try:
        result = subprocess.run(
            ['ollama', 'run', model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return []

        return _parse_topics(result.stdout)

    except subprocess.TimeoutExpired:
        return []
    except Exception as e:
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
    except:
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
    except:
        pass
    return []
