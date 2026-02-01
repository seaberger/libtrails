"""Book indexing functionality - parse, chunk, extract topics."""

import time
from typing import Callable, Optional

from .chunker import chunk_text
from .config import CHUNK_TARGET_WORDS, DEFAULT_MODEL
from .database import (
    get_book,
    get_book_path,
    get_db,
    init_chunks_table,
    save_chunk_topics,
    save_chunks,
)
from .document_parser import extract_text
from .topic_extractor import check_ollama_available, extract_topics_batch


class IndexingError(Exception):
    """Raised when indexing fails."""
    pass


class IndexingResult:
    """Result of indexing a book."""

    def __init__(
        self,
        book_id: int,
        word_count: int = 0,
        chunk_count: int = 0,
        unique_topics: int = 0,
        all_topics: list[str] = None,
        skipped: bool = False,
        skip_reason: str = None,
        error: str = None
    ):
        self.book_id = book_id
        self.word_count = word_count
        self.chunk_count = chunk_count
        self.unique_topics = unique_topics
        self.all_topics = all_topics or []
        self.skipped = skipped
        self.skip_reason = skip_reason
        self.error = error

    @property
    def success(self) -> bool:
        return not self.error and not self.skipped

    def __repr__(self):
        if self.error:
            return f"IndexingResult(book_id={self.book_id}, error={self.error!r})"
        if self.skipped:
            return f"IndexingResult(book_id={self.book_id}, skipped={self.skip_reason!r})"
        return f"IndexingResult(book_id={self.book_id}, chunks={self.chunk_count}, topics={self.unique_topics})"


def index_book(
    book_id: int,
    model: str = DEFAULT_MODEL,
    max_words: Optional[int] = None,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    topic_progress_callback: Optional[Callable[[int, int], None]] = None,
) -> IndexingResult:
    """
    Index a single book - extract text, chunk, and extract topics.

    Args:
        book_id: The book ID from our database
        model: Ollama model to use for topic extraction
        max_words: Skip books with more than this many words
        dry_run: If True, parse and chunk but skip topic extraction
        progress_callback: Called with status messages
        topic_progress_callback: Called with (completed, total) during topic extraction

    Returns:
        IndexingResult with details about the indexing

    Raises:
        IndexingError: If a fatal error occurs
    """
    # Ensure chunks table exists
    init_chunks_table()

    # Get book from database
    book = get_book(book_id)
    if not book:
        raise IndexingError(f"Book {book_id} not found in database")

    if progress_callback:
        progress_callback(f"Indexing: {book['title'][:60]}")

    # Check for Calibre match
    if not book.get('calibre_id'):
        return IndexingResult(
            book_id=book_id,
            error="No Calibre match for this book"
        )

    # Get book file path
    book_path = get_book_path(book['calibre_id'])
    if not book_path:
        return IndexingResult(
            book_id=book_id,
            error="No EPUB or PDF found in Calibre library"
        )

    file_format = book_path.suffix.upper().lstrip('.')
    if progress_callback:
        progress_callback(f"  {file_format}: {book_path.name}")

    # Extract text
    if progress_callback:
        progress_callback(f"  Extracting text from {file_format}...")

    try:
        text = extract_text(book_path)
    except Exception as e:
        return IndexingResult(
            book_id=book_id,
            error=f"Text extraction failed: {e}"
        )

    word_count = len(text.split())

    # Check minimum content
    if word_count < 100:
        return IndexingResult(
            book_id=book_id,
            word_count=word_count,
            error=f"Insufficient content: only {word_count} words extracted"
        )

    # Check maximum content
    if max_words and word_count > max_words:
        return IndexingResult(
            book_id=book_id,
            word_count=word_count,
            skipped=True,
            skip_reason=f"{word_count:,} words exceeds max {max_words:,}"
        )

    if progress_callback:
        progress_callback(f"  Extracted {word_count:,} words")

    # Chunk text
    chunks = chunk_text(text, CHUNK_TARGET_WORDS)
    if progress_callback:
        progress_callback(f"  Created {len(chunks)} chunks")

    # Save chunks to database
    save_chunks(book_id, chunks)

    if dry_run:
        return IndexingResult(
            book_id=book_id,
            word_count=word_count,
            chunk_count=len(chunks),
            skipped=True,
            skip_reason="Dry run - skipped topic extraction"
        )

    # Check Ollama availability
    if not check_ollama_available(model):
        return IndexingResult(
            book_id=book_id,
            word_count=word_count,
            chunk_count=len(chunks),
            error=f"Model {model} not available in Ollama"
        )

    if progress_callback:
        progress_callback(f"  Extracting topics with {model}...")

    # Get chunk IDs for saving topics
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id FROM chunks WHERE book_id = ? ORDER BY chunk_index",
            (book_id,)
        )
        chunk_ids = [row[0] for row in cursor.fetchall()]

    # Extract topics in parallel
    topics_per_chunk = extract_topics_batch(
        chunks, model, progress_callback=topic_progress_callback
    )

    # Save topics
    all_topics = []
    for chunk_id, topics in zip(chunk_ids, topics_per_chunk):
        if topics:
            save_chunk_topics(chunk_id, topics)
            all_topics.extend(topics)

    unique_topics = set(all_topics)

    if progress_callback:
        progress_callback(f"  Extracted {len(unique_topics)} unique topics")

    return IndexingResult(
        book_id=book_id,
        word_count=word_count,
        chunk_count=len(chunks),
        unique_topics=len(unique_topics),
        all_topics=all_topics
    )


def index_books_batch(
    book_ids: list[int],
    model: str = DEFAULT_MODEL,
    max_words: Optional[int] = None,
    dry_run: bool = False,
    progress_callback: Optional[Callable[[str], None]] = None,
    book_callback: Optional[Callable[[int, int, IndexingResult], None]] = None,
    battery_check: Optional[Callable[[], Optional[int]]] = None,
    min_battery: int = 15,
    resume_battery: int = 50,
) -> dict:
    """
    Index multiple books with progress tracking and battery management.

    Args:
        book_ids: List of book IDs to index
        model: Ollama model to use
        max_words: Skip books over this word count
        dry_run: Parse/chunk only, no topic extraction
        progress_callback: Called with status messages
        book_callback: Called after each book with (current, total, result)
        battery_check: Function returning current battery % (or None)
        min_battery: Pause when battery drops below this
        resume_battery: Resume when battery reaches this level

    Returns:
        Summary dict with success/failure counts
    """
    successful = 0
    skipped = 0
    failed = []
    start_time = time.time()

    for i, book_id in enumerate(book_ids, 1):
        # Check battery
        if battery_check:
            battery = battery_check()
            if battery is not None and battery < min_battery:
                if progress_callback:
                    progress_callback(f"Battery at {battery}% - pausing until {resume_battery}%...")

                while True:
                    time.sleep(300)  # Check every 5 minutes
                    battery = battery_check()
                    if battery is None or battery >= resume_battery:
                        if progress_callback:
                            progress_callback(f"Battery at {battery or 'unknown'}% - resuming")
                        break

        # Index book
        try:
            result = index_book(
                book_id,
                model=model,
                max_words=max_words,
                dry_run=dry_run,
                progress_callback=progress_callback
            )

            if result.success:
                successful += 1
            elif result.skipped:
                skipped += 1
            else:
                failed.append((book_id, result.error))

            if book_callback:
                book_callback(i, len(book_ids), result)

        except KeyboardInterrupt:
            if progress_callback:
                progress_callback("Interrupted - progress saved")
            break
        except Exception as e:
            failed.append((book_id, str(e)))
            if progress_callback:
                progress_callback(f"Error indexing book {book_id}: {e}")

    elapsed = time.time() - start_time

    return {
        "successful": successful,
        "skipped": skipped,
        "failed": len(failed),
        "failed_details": failed,
        "total_processed": i,
        "elapsed_seconds": elapsed
    }
