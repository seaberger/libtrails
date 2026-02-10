"""Recursive text chunking for topic extraction.

Splits text using a hierarchy of natural boundaries:
  1. Paragraphs (\\n\\n) — preferred, keeps complete thoughts together
  2. Sentences (. ! ?) — fallback for long paragraphs
  3. Words (spaces) — last resort for very long sentences

Each level is tried in order. If a unit fits within the target chunk
size, it's kept whole. If it exceeds the target, the next level down
is used to split it further.
"""

import re

from .config import CHUNK_MIN_WORDS, CHUNK_TARGET_WORDS

# Common abbreviations that end with a period but aren't sentence endings
_ABBREVIATIONS = frozenset(
    {
        "mr",
        "mrs",
        "ms",
        "dr",
        "prof",
        "sr",
        "jr",
        "st",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "corp",
        "dept",
        "gen",
        "gov",
        "sgt",
        "cpl",
        "pvt",
        "capt",
        "col",
        "maj",
        "rev",
        "hon",
        "pres",
        "approx",
        "vol",
        "no",
        "fig",
        "ed",
        "trans",
        "repr",
    }
)


def chunk_text(text: str, target_words: int = CHUNK_TARGET_WORDS) -> list[str]:
    """
    Split text into chunks of approximately target_words.

    Uses a recursive splitting hierarchy:
      1. Paragraph breaks — keeps paragraphs intact when possible
      2. Sentence endings — splits long paragraphs at sentence boundaries
      3. Word boundaries — last resort for very long sentences

    Chunks shorter than CHUNK_MIN_WORDS are merged with neighbors
    or discarded if they appear at the end.
    """
    paragraphs = _split_paragraphs(text)

    chunks = []
    current_parts = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words == 0:
            continue

        # If this single paragraph exceeds target, split it further
        if para_words > target_words:
            # Emit any accumulated content first
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_words = 0

            # Split the long paragraph into sentence-level chunks
            sub_chunks = _chunk_by_sentences(para, target_words)
            chunks.extend(sub_chunks)
            continue

        # If adding this paragraph would exceed target, emit current chunk
        if current_words + para_words > target_words and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = []
            current_words = 0

        current_parts.append(para)
        current_words += para_words

    # Emit last chunk
    if current_parts:
        chunks.append("\n\n".join(current_parts))

    # Filter out chunks below minimum word count
    return [c for c in chunks if len(c.split()) >= CHUNK_MIN_WORDS]


def _split_paragraphs(text: str) -> list[str]:
    """Split text on paragraph boundaries (double newlines)."""
    paragraphs = re.split(r"\n\n+", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences, handling abbreviations and edge cases.

    Uses a split-then-rejoin strategy: aggressively split on [.!?]
    followed by whitespace, then rejoin false splits at abbreviations
    and other non-boundary periods.
    """
    # Split on sentence-ending punctuation followed by whitespace
    raw_parts = re.split(r"(?<=[.!?])\s+", text)

    if len(raw_parts) <= 1:
        return [text.strip()] if text.strip() else []

    # Rejoin parts that were falsely split at abbreviations
    sentences = []
    current = raw_parts[0]

    for next_part in raw_parts[1:]:
        if _is_false_sentence_break(current, next_part):
            current = current + " " + next_part
        else:
            sentences.append(current.strip())
            current = next_part

    if current.strip():
        sentences.append(current.strip())

    return [s for s in sentences if s]


def _is_false_sentence_break(before: str, after: str) -> bool:
    """
    Determine if a split between 'before' and 'after' is a false
    sentence boundary (e.g., an abbreviation, initial, or decimal).
    """
    if not before or not after:
        return False

    before = before.rstrip()

    # Get the last "word" before the period
    tokens = before.rsplit(None, 1)
    if not tokens:
        return False

    last_token = tokens[-1]
    # Strip the trailing punctuation to get the word
    word = last_token.rstrip(".!?").lower()

    # Single letter followed by period (initials: "J. K. Rowling")
    if len(word) == 1 and word.isalpha():
        return True

    # Multi-period abbreviation (U.S., U.K., D.C., Ph.D., M.I.T., etc.)
    if re.match(r"^[a-z](?:\.[a-z])+$", word):
        return True

    # Known abbreviation
    if word in _ABBREVIATIONS:
        return True

    # Ends with a digit followed by period (decimals, version numbers)
    if last_token.rstrip(".") and last_token.rstrip(".")[-1:].isdigit():
        # But only if next part starts lowercase (not a new sentence)
        if after[:1].islower():
            return True

    # Next part starts with lowercase — very likely a false split
    # (real sentences start with uppercase in book text)
    if after[:1].islower():
        return True

    return False


def _chunk_by_sentences(text: str, target_words: int) -> list[str]:
    """
    Chunk a long paragraph by grouping sentences.

    Falls back to word-level splitting for individual sentences
    that exceed the target size.
    """
    sentences = _split_sentences(text)

    chunks = []
    current_parts = []
    current_words = 0

    for sentence in sentences:
        sent_words = len(sentence.split())

        if sent_words == 0:
            continue

        # If a single sentence exceeds target, split on words
        if sent_words > target_words:
            if current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_words = 0

            word_chunks = _chunk_by_words(sentence, target_words)
            chunks.extend(word_chunks)
            continue

        if current_words + sent_words > target_words and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = []
            current_words = 0

        current_parts.append(sentence)
        current_words += sent_words

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


def _chunk_by_words(text: str, target_words: int) -> list[str]:
    """Last resort: split on word boundaries."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), target_words):
        chunk = " ".join(words[i : i + target_words])
        chunks.append(chunk)
    return chunks
