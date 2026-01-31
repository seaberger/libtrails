"""Text chunking for topic extraction."""

import re
from .config import CHUNK_TARGET_WORDS, CHUNK_MIN_WORDS


def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """
    Split text into chunks of approximately target_words.

    Respects sentence boundaries to keep chunks coherent.
    Filters out chunks that are too short (likely artifacts).
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = len(sentence.split())

        # If adding this sentence would exceed target and we have content, start new chunk
        if current_words + words > target_words and current_chunk:
            chunk_text = ' '.join(current_chunk)
            if len(chunk_text.split()) >= CHUNK_MIN_WORDS:
                chunks.append(chunk_text)
            current_chunk = [sentence]
            current_words = words
        else:
            current_chunk.append(sentence)
            current_words += words

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = ' '.join(current_chunk)
        if len(chunk_text.split()) >= CHUNK_MIN_WORDS:
            chunks.append(chunk_text)

    return chunks


def chunk_text_by_paragraphs(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """
    Split text into chunks, respecting paragraph boundaries.

    This produces more natural chunks but may have more size variance.
    """
    # Split by double newlines (paragraphs)
    paragraphs = re.split(r'\n\n+', text)

    chunks = []
    current_chunk = []
    current_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        words = len(para.split())

        if current_words + words > target_words and current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            if len(chunk_text.split()) >= CHUNK_MIN_WORDS:
                chunks.append(chunk_text)
            current_chunk = [para]
            current_words = words
        else:
            current_chunk.append(para)
            current_words += words

    if current_chunk:
        chunk_text = '\n\n'.join(current_chunk)
        if len(chunk_text.split()) >= CHUNK_MIN_WORDS:
            chunks.append(chunk_text)

    return chunks
