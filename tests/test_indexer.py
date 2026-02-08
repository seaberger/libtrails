"""Tests for book indexer functionality."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libtrails.indexer import (
    IndexingError,
    IndexingResult,
    index_book,
    index_books_batch,
)


class TestIndexingResult:
    """Tests for IndexingResult class."""

    def test_success_result(self):
        """Test successful indexing result."""
        result = IndexingResult(
            book_id=1,
            word_count=5000,
            chunk_count=10,
            unique_topics=25,
            all_topics=["topic1", "topic2"],
        )

        assert result.success is True
        assert result.book_id == 1
        assert result.word_count == 5000
        assert result.chunk_count == 10
        assert result.unique_topics == 25

    def test_skipped_result(self):
        """Test skipped result."""
        result = IndexingResult(book_id=1, skipped=True, skip_reason="Too large")

        assert result.success is False
        assert result.skipped is True
        assert result.skip_reason == "Too large"

    def test_error_result(self):
        """Test error result."""
        result = IndexingResult(book_id=1, error="No file found")

        assert result.success is False
        assert result.error == "No file found"

    def test_repr(self):
        """Test string representation."""
        result = IndexingResult(book_id=1, chunk_count=5, unique_topics=10)
        assert "book_id=1" in repr(result)

        error_result = IndexingResult(book_id=2, error="Test error")
        assert "error" in repr(error_result)


class TestIndexBook:
    """Tests for index_book function."""

    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_book_not_found(self, mock_init, mock_get_book):
        """Test indexing a book that doesn't exist."""
        mock_get_book.return_value = None

        with pytest.raises(IndexingError) as exc:
            index_book(999)

        assert "not found" in str(exc.value)

    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_no_calibre_match(self, mock_init, mock_get_book):
        """Test indexing a book without Calibre match."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": None}

        result = index_book(1)

        assert result.success is False
        assert "No Calibre match" in result.error

    @patch("libtrails.indexer.get_book_path")
    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_no_epub_file(self, mock_init, mock_get_book, mock_get_path):
        """Test indexing when no EPUB/PDF file exists."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": 123}
        mock_get_path.return_value = None

        result = index_book(1)

        assert result.success is False
        assert "No EPUB or PDF" in result.error

    @patch("libtrails.indexer.extract_text")
    @patch("libtrails.indexer.get_book_path")
    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_insufficient_content(self, mock_init, mock_get_book, mock_get_path, mock_extract):
        """Test indexing when book has too little content."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": 123}
        mock_get_path.return_value = Path("/fake/path/book.epub")
        mock_extract.return_value = "Too short"  # Only 2 words

        result = index_book(1)

        assert result.success is False
        assert "Insufficient content" in result.error

    @patch("libtrails.indexer.extract_text")
    @patch("libtrails.indexer.get_book_path")
    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_max_words_exceeded(self, mock_init, mock_get_book, mock_get_path, mock_extract):
        """Test skipping when book exceeds max_words."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": 123}
        mock_get_path.return_value = Path("/fake/path/book.epub")
        # Create text with 1000 words
        mock_extract.return_value = " ".join(["word"] * 1000)

        result = index_book(1, max_words=500)

        assert result.skipped is True
        assert "exceeds max" in result.skip_reason

    @patch("libtrails.indexer.save_chunks")
    @patch("libtrails.indexer.chunk_text")
    @patch("libtrails.indexer.extract_text")
    @patch("libtrails.indexer.get_book_path")
    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_dry_run(
        self, mock_init, mock_get_book, mock_get_path, mock_extract, mock_chunk, mock_save
    ):
        """Test dry run mode."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": 123}
        mock_get_path.return_value = Path("/fake/path/book.epub")
        mock_extract.return_value = " ".join(["word"] * 500)
        mock_chunk.return_value = ["chunk1", "chunk2", "chunk3"]

        result = index_book(1, dry_run=True)

        assert result.skipped is True
        assert "Dry run" in result.skip_reason
        assert result.chunk_count == 3
        mock_save.assert_called_once()

    @patch("libtrails.indexer.save_chunk_topics")
    @patch("libtrails.indexer.get_db")
    @patch("libtrails.indexer.extract_topics_batch")
    @patch("libtrails.indexer.check_ollama_available")
    @patch("libtrails.indexer.save_chunks")
    @patch("libtrails.indexer.chunk_text")
    @patch("libtrails.indexer.extract_text")
    @patch("libtrails.indexer.get_book_path")
    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_full_indexing(
        self,
        mock_init,
        mock_get_book,
        mock_get_path,
        mock_extract,
        mock_chunk,
        mock_save_chunks,
        mock_ollama,
        mock_extract_topics,
        mock_db,
        mock_save_topics,
    ):
        """Test full indexing pipeline."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": 123}
        mock_get_path.return_value = Path("/fake/path/book.epub")
        mock_extract.return_value = " ".join(["word"] * 500)
        mock_chunk.return_value = ["chunk1", "chunk2"]
        mock_ollama.return_value = True
        mock_extract_topics.return_value = [["topic1", "topic2"], ["topic2", "topic3"]]

        # Mock database context manager
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [(1,), (2,)]  # chunk IDs
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        result = index_book(1, legacy=True)

        assert result.success is True
        assert result.chunk_count == 2
        assert result.unique_topics == 3  # topic1, topic2, topic3
        assert len(result.all_topics) == 4  # All topic occurrences

    @patch("libtrails.indexer.check_ollama_available")
    @patch("libtrails.indexer.save_chunks")
    @patch("libtrails.indexer.chunk_text")
    @patch("libtrails.indexer.extract_text")
    @patch("libtrails.indexer.get_book_path")
    @patch("libtrails.indexer.get_book")
    @patch("libtrails.indexer.init_chunks_table")
    def test_ollama_unavailable(
        self,
        mock_init,
        mock_get_book,
        mock_get_path,
        mock_extract,
        mock_chunk,
        mock_save,
        mock_ollama,
    ):
        """Test when Ollama model is not available."""
        mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": 123}
        mock_get_path.return_value = Path("/fake/path/book.epub")
        mock_extract.return_value = " ".join(["word"] * 500)
        mock_chunk.return_value = ["chunk1", "chunk2"]
        mock_ollama.return_value = False

        result = index_book(1, chunk_model="nonexistent:model")

        assert result.success is False
        assert "not available" in result.error

    def test_progress_callback(self):
        """Test that progress callback is called."""
        messages = []

        def callback(msg):
            messages.append(msg)

        with (
            patch("libtrails.indexer.get_book") as mock_get_book,
            patch("libtrails.indexer.init_chunks_table"),
        ):
            mock_get_book.return_value = {"id": 1, "title": "Test Book", "calibre_id": None}

            index_book(1, progress_callback=callback)

        assert len(messages) > 0
        assert any("Indexing" in m for m in messages)


class TestIndexBooksBatch:
    """Tests for index_books_batch function."""

    @patch("libtrails.indexer.index_book")
    def test_batch_processing(self, mock_index):
        """Test batch processing of multiple books."""
        mock_index.side_effect = [
            IndexingResult(book_id=1, chunk_count=5, unique_topics=10),
            IndexingResult(book_id=2, chunk_count=8, unique_topics=15),
            IndexingResult(book_id=3, skipped=True, skip_reason="Too large"),
        ]

        result = index_books_batch([1, 2, 3])

        assert result["successful"] == 2
        assert result["skipped"] == 1
        assert result["failed"] == 0

    @patch("libtrails.indexer.index_book")
    def test_batch_with_failures(self, mock_index):
        """Test batch handling failures."""
        mock_index.side_effect = [
            IndexingResult(book_id=1, chunk_count=5, unique_topics=10),
            IndexingResult(book_id=2, error="File not found"),
            IndexingResult(book_id=3, chunk_count=3, unique_topics=5),
        ]

        result = index_books_batch([1, 2, 3])

        assert result["successful"] == 2
        assert result["failed"] == 1

    @patch("libtrails.indexer.index_book")
    def test_book_callback(self, mock_index):
        """Test that book callback is called after each book."""
        mock_index.return_value = IndexingResult(book_id=1, chunk_count=5, unique_topics=10)

        callbacks = []

        def book_callback(current, total, result):
            callbacks.append((current, total, result.book_id))

        index_books_batch([1, 2, 3], book_callback=book_callback)

        assert len(callbacks) == 3
        assert callbacks[0] == (1, 3, 1)
        assert callbacks[-1] == (3, 3, 1)

    @patch("time.sleep")  # Patch at module level
    @patch("libtrails.indexer.index_book")
    def test_battery_management(self, mock_index, mock_sleep):
        """Test battery pause/resume behavior."""
        mock_index.return_value = IndexingResult(book_id=1, chunk_count=5, unique_topics=10)

        # Simulate battery dropping then recovering
        battery_levels = [10, 20, 30, 50, 60]  # Start low, then recover
        battery_iter = iter(battery_levels)

        def mock_battery_check():
            return next(battery_iter, 100)

        messages = []

        def progress_callback(msg):
            messages.append(msg)

        index_books_batch(
            [1],
            battery_check=mock_battery_check,
            min_battery=15,
            resume_battery=50,
            progress_callback=progress_callback,
        )

        # Should have paused and resumed
        assert any("pausing" in m.lower() for m in messages)

    @patch("libtrails.indexer.index_book")
    def test_dry_run_batch(self, mock_index):
        """Test batch processing with dry run."""
        mock_index.return_value = IndexingResult(
            book_id=1, word_count=500, chunk_count=5, skipped=True, skip_reason="Dry run"
        )

        result = index_books_batch([1, 2], dry_run=True)

        assert result["skipped"] == 2
        assert result["successful"] == 0
