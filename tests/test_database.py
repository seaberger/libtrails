"""Tests for database operations."""

import sqlite3
from contextlib import contextmanager

import pytest

from libtrails.database import (
    get_all_books,
    get_book,
    get_book_by_title,
    get_indexing_status,
    get_topics_without_embeddings,
    save_chunk_topics,
    save_chunks,
    save_topic_embedding,
)


@pytest.fixture
def test_db(tmp_path):
    """Create a test database with schema."""
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Create books table
    conn.execute("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            ipad_id TEXT,
            title TEXT NOT NULL,
            author TEXT,
            calibre_id INTEGER,
            has_calibre_match INTEGER DEFAULT 0
        )
    """)

    # Create chunks table (with all required columns)
    conn.execute("""
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            book_id INTEGER,
            chunk_index INTEGER,
            text TEXT,
            word_count INTEGER,
            topics_json TEXT
        )
    """)

    # Create chunk_topics table
    conn.execute("""
        CREATE TABLE chunk_topics (
            id INTEGER PRIMARY KEY,
            chunk_id INTEGER,
            topic TEXT
        )
    """)

    # Create topics table
    conn.execute("""
        CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            label TEXT UNIQUE,
            embedding BLOB,
            cluster_id INTEGER,
            occurrence_count INTEGER DEFAULT 0
        )
    """)

    # Create chunk_topic_links table
    conn.execute("""
        CREATE TABLE chunk_topic_links (
            chunk_id INTEGER,
            topic_id INTEGER,
            PRIMARY KEY (chunk_id, topic_id)
        )
    """)

    # Create topic_cooccurrences table
    conn.execute("""
        CREATE TABLE topic_cooccurrences (
            topic1_id INTEGER,
            topic2_id INTEGER,
            count INTEGER DEFAULT 0,
            pmi REAL,
            PRIMARY KEY (topic1_id, topic2_id)
        )
    """)

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture
def mock_db(test_db):
    """Provide a mock get_db that uses the test database."""
    @contextmanager
    def _get_db(db_path=None):
        conn = sqlite3.connect(test_db)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    return _get_db


class TestGetBook:
    """Tests for book retrieval."""

    def test_get_book_by_id(self, test_db, mock_db, monkeypatch):
        """Test retrieving a book by ID."""
        # Add a test book
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (id, title, author) VALUES (1, 'Test Book', 'Test Author')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        book = get_book(1)

        assert book is not None
        assert book['title'] == 'Test Book'
        assert book['author'] == 'Test Author'

    def test_get_book_not_found(self, test_db, mock_db, monkeypatch):
        """Test retrieving a non-existent book."""
        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        book = get_book(999)

        assert book is None


class TestGetBookByTitle:
    """Tests for book retrieval by title."""

    def test_exact_match(self, test_db, mock_db, monkeypatch):
        """Test finding a book by exact title."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (title, author) VALUES ('Siddhartha', 'Hermann Hesse')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        book = get_book_by_title('Siddhartha')

        assert book is not None
        assert book['author'] == 'Hermann Hesse'

    def test_partial_match(self, test_db, mock_db, monkeypatch):
        """Test finding a book by partial title."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (title, author) VALUES ('The Great Gatsby', 'F. Scott Fitzgerald')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        book = get_book_by_title('Gatsby')

        assert book is not None
        assert 'Gatsby' in book['title']


class TestGetAllBooks:
    """Tests for retrieving all books."""

    def test_get_all_books(self, test_db, mock_db, monkeypatch):
        """Test retrieving all books."""
        conn = sqlite3.connect(test_db)
        # Note: get_all_books() defaults to with_calibre_match=True, which filters for calibre_id IS NOT NULL
        conn.execute("INSERT INTO books (title, author, calibre_id) VALUES ('Book 1', 'Author 1', 100)")
        conn.execute("INSERT INTO books (title, author, calibre_id) VALUES ('Book 2', 'Author 2', 200)")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        books = get_all_books()

        assert len(books) == 2

    def test_filter_with_calibre_match(self, test_db, mock_db, monkeypatch):
        """Test filtering books with Calibre match."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (title, calibre_id, has_calibre_match) VALUES ('Book 1', 123, 1)")
        conn.execute("INSERT INTO books (title, calibre_id, has_calibre_match) VALUES ('Book 2', NULL, 0)")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        books = get_all_books(with_calibre_match=True)

        assert len(books) == 1
        assert books[0]['calibre_id'] == 123


class TestSaveChunks:
    """Tests for saving chunks."""

    def test_save_chunks(self, test_db, mock_db, monkeypatch):
        """Test saving chunks to database."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (id, title) VALUES (1, 'Test Book')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        chunks = ["Chunk one text", "Chunk two text", "Chunk three text"]
        save_chunks(1, chunks)

        # Verify chunks were saved
        conn = sqlite3.connect(test_db)
        cursor = conn.execute("SELECT COUNT(*) FROM chunks WHERE book_id = 1")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3


class TestSaveChunkTopics:
    """Tests for saving chunk topics."""

    def test_save_chunk_topics(self, test_db, mock_db, monkeypatch):
        """Test saving topics for a chunk."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (id, title) VALUES (1, 'Test Book')")
        conn.execute("INSERT INTO chunks (id, book_id, chunk_index, text) VALUES (1, 1, 0, 'text')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        topics = ["Philosophy", "Ethics", "Morality"]
        save_chunk_topics(1, topics)

        # Verify topics were saved
        conn = sqlite3.connect(test_db)
        cursor = conn.execute("SELECT COUNT(*) FROM chunk_topics WHERE chunk_id = 1")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 3


class TestGetIndexingStatus:
    """Tests for indexing status retrieval."""

    def test_empty_database(self, test_db, mock_db, monkeypatch):
        """Test status with empty database."""
        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        status = get_indexing_status()

        assert status['total_books'] == 0
        assert status['indexed_books'] == 0
        assert status['total_chunks'] == 0

    def test_with_data(self, test_db, mock_db, monkeypatch):
        """Test status with data in database."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO books (id, title, calibre_id) VALUES (1, 'Book 1', 100)")
        conn.execute("INSERT INTO books (id, title, calibre_id) VALUES (2, 'Book 2', 200)")
        conn.execute("INSERT INTO chunks (book_id, chunk_index, text) VALUES (1, 0, 'text')")
        conn.execute("INSERT INTO chunks (book_id, chunk_index, text) VALUES (1, 1, 'text')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        status = get_indexing_status()

        assert status['total_books'] == 2
        assert status['indexed_books'] == 1
        assert status['total_chunks'] == 2


class TestTopicOperations:
    """Tests for topic-related operations."""

    def test_get_topics_without_embeddings(self, test_db, mock_db, monkeypatch):
        """Test getting topics that need embeddings."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO topics (id, label, embedding) VALUES (1, 'topic1', NULL)")
        conn.execute("INSERT INTO topics (id, label, embedding) VALUES (2, 'topic2', X'00')")
        conn.execute("INSERT INTO topics (id, label, embedding) VALUES (3, 'topic3', NULL)")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        topics = get_topics_without_embeddings()

        assert len(topics) == 2
        labels = [t['label'] for t in topics]
        assert 'topic1' in labels
        assert 'topic3' in labels

    def test_save_topic_embedding(self, test_db, mock_db, monkeypatch):
        """Test saving an embedding for a topic."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO topics (id, label) VALUES (1, 'test_topic')")
        conn.commit()
        conn.close()

        from libtrails import database
        monkeypatch.setattr(database, 'get_db', mock_db)

        embedding_bytes = b'\x00\x01\x02\x03'
        save_topic_embedding(1, embedding_bytes)

        # Verify embedding was saved
        conn = sqlite3.connect(test_db)
        cursor = conn.execute("SELECT embedding FROM topics WHERE id = 1")
        result = cursor.fetchone()[0]
        conn.close()

        assert result == embedding_bytes
