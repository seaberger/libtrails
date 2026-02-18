"""Shared fixtures for libtrails tests."""

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Sample iPad HTML responses for mocking
SAMPLE_TAGS_HTML = """
<html>
<body>
<a class='title' href='/tags?set=0&amp;tag=Fiction'>Fiction</a><br /><span class='content'>Total 5 books</span>
<a class='title' href='/tags?set=0&amp;tag=Science%20Fiction'>Science Fiction</a><br /><span class='content'>Total 3 books</span>
<a class='title' href='/tags?set=0&amp;tag=Philosophy'>Philosophy</a><br /><span class='content'>Total 2 books</span>
</body>
</html>
"""

SAMPLE_TAG_BOOKS_HTML = """
<html>
<body>
<a class='title' href='/book?id=abc123.epub'>Test Book One</a><br /><span class='author'>John Smith</span>
<a class='title' href='/book?id=def456.epub'>Test Book Two</a><br /><span class='author'>Jane Doe</span>
</body>
</html>
"""

SAMPLE_TITLE_SECTION_HTML = """
<html>
<body>
<a class='title' href='/book?id=ghi789.pdf'>Another Book</a><br /><span class='author'>Bob Wilson</span>
</body>
</html>
"""


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database with the books schema."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ipad_id TEXT UNIQUE NOT NULL,
            calibre_id INTEGER,
            title TEXT NOT NULL,
            author TEXT,
            format TEXT,
            series TEXT,
            series_index REAL,
            publisher TEXT,
            pubdate TEXT,
            description TEXT,
            has_calibre_match BOOLEAN DEFAULT 0
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            book_id INTEGER REFERENCES books(id),
            chunk_index INTEGER,
            text TEXT NOT NULL,
            word_count INTEGER,
            topics_json TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(book_id, chunk_index)
        );

        CREATE TABLE chunk_topics (
            chunk_id INTEGER REFERENCES chunks(id),
            topic TEXT NOT NULL,
            PRIMARY KEY (chunk_id, topic)
        );

        CREATE TABLE topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT UNIQUE NOT NULL,
            embedding BLOB,
            cluster_id INTEGER,
            parent_topic_id INTEGER,
            occurrence_count INTEGER DEFAULT 0
        );
    """)
    conn.close()

    yield db_path

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.fixture
def temp_calibre_db():
    """Create a temporary mock Calibre database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author_sort TEXT,
            path TEXT,
            has_cover BOOLEAN DEFAULT 0,
            pubdate TEXT,
            series_index REAL
        );

        CREATE TABLE authors (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE books_authors_link (
            book INTEGER,
            author INTEGER
        );

        CREATE TABLE tags (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE books_tags_link (
            book INTEGER,
            tag INTEGER
        );

        CREATE TABLE series (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE books_series_link (
            book INTEGER,
            series INTEGER
        );

        CREATE TABLE comments (
            book INTEGER PRIMARY KEY,
            text TEXT
        );

        CREATE TABLE identifiers (
            book INTEGER,
            type TEXT,
            val TEXT
        );

        CREATE TABLE publishers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );

        CREATE TABLE books_publishers_link (
            book INTEGER,
            publisher INTEGER
        );

        CREATE TABLE data (
            book INTEGER,
            format TEXT
        );

        -- Insert test data
        INSERT INTO books (id, title, author_sort, path, has_cover)
        VALUES
            (1, 'Test Book One', 'Smith, John', 'John Smith/Test Book One (1)', 1),
            (2, 'Test Book Two', 'Doe, Jane', 'Jane Doe/Test Book Two (2)', 1),
            (3, 'Another Book', 'Wilson, Bob', 'Bob Wilson/Another Book (3)', 0);

        INSERT INTO authors (id, name) VALUES (1, 'John Smith'), (2, 'Jane Doe'), (3, 'Bob Wilson');
        INSERT INTO books_authors_link VALUES (1, 1), (2, 2), (3, 3);

        INSERT INTO tags (id, name) VALUES (1, 'Fiction'), (2, 'Science Fiction'), (3, 'Philosophy');
        INSERT INTO books_tags_link VALUES (1, 1), (1, 2), (2, 1), (3, 3);

        INSERT INTO series (id, name) VALUES (1, 'Test Series');
        INSERT INTO books_series_link VALUES (1, 1);

        INSERT INTO comments (book, text) VALUES
            (1, 'A great test book about testing.'),
            (2, 'Another excellent test book.');

        INSERT INTO identifiers VALUES (1, 'isbn', '9781234567890'), (2, 'isbn', '9780987654321');

        INSERT INTO data VALUES (1, 'EPUB'), (2, 'EPUB'), (3, 'PDF');
    """)
    conn.close()

    yield db_path

    db_path.unlink(missing_ok=True)


@pytest.fixture
def mock_ipad_server(monkeypatch):
    """Mock urllib.request.urlopen to simulate iPad server responses."""

    def mock_urlopen(url, timeout=None):
        mock_response = MagicMock()

        if "/tags?set=0" in url and "tag=" not in url:
            # Tags list page
            mock_response.read.return_value = SAMPLE_TAGS_HTML.encode("utf-8")
        elif "/tags?set=0&tag=" in url:
            # Individual tag page
            mock_response.read.return_value = SAMPLE_TAG_BOOKS_HTML.encode("utf-8")
        elif "/?set=0&sort=title&sec=" in url:
            # Title section page
            mock_response.read.return_value = SAMPLE_TITLE_SECTION_HTML.encode("utf-8")
        else:
            mock_response.read.return_value = b"<html></html>"

        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = lambda s, *args: None
        return mock_response

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)


@pytest.fixture
def temp_config_dir(tmp_path, monkeypatch):
    """Create a temporary config directory."""
    config_dir = tmp_path / ".libtrails"
    config_dir.mkdir()

    # Patch the config module to use temp directory
    monkeypatch.setattr("libtrails.config.USER_CONFIG_DIR", config_dir)
    monkeypatch.setattr("libtrails.config.USER_CONFIG_FILE", config_dir / "config.yaml")

    return config_dir


@pytest.fixture
def stats_db(tmp_path):
    """Create a database with full schema and seeded data for stats refresh tests.

    Seed data layout:
    - 2 books (B1 with 100 total topics, B2 with 50 total topics)
    - 2 clusters (C0, C1)
    - 2 domains (D0, D1)
    - 2 communities (Comm0, Comm1)
    - cluster_books pre-populated (B1→C0:60, B1→C1:40, B2→C0:20, B2→C1:30)
    - cluster_domains: C0→D0, C1→D1
    - cluster_communities: C0→Comm0, C1→Comm1
    - topics with community_id assignments for community_stats queries
    """
    db_path = tmp_path / "stats_test.db"
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            author TEXT,
            calibre_id INTEGER,
            book_themes TEXT
        );

        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY,
            book_id INTEGER REFERENCES books(id),
            chunk_index INTEGER,
            text TEXT,
            word_count INTEGER,
            topics_json TEXT
        );

        CREATE TABLE topics (
            id INTEGER PRIMARY KEY,
            label TEXT UNIQUE NOT NULL,
            embedding BLOB,
            cluster_id INTEGER,
            community_id INTEGER,
            occurrence_count INTEGER DEFAULT 0
        );

        CREATE TABLE chunk_topic_links (
            chunk_id INTEGER,
            topic_id INTEGER,
            PRIMARY KEY (chunk_id, topic_id)
        );

        CREATE TABLE domains (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL UNIQUE,
            cluster_count INTEGER DEFAULT 0
        );

        CREATE TABLE cluster_domains (
            cluster_id INTEGER PRIMARY KEY,
            domain_id INTEGER NOT NULL REFERENCES domains(id)
        );

        CREATE TABLE communities (
            id INTEGER PRIMARY KEY,
            label TEXT NOT NULL,
            domain_id INTEGER REFERENCES domains(id),
            topic_count INTEGER DEFAULT 0
        );

        CREATE TABLE cluster_communities (
            cluster_id INTEGER PRIMARY KEY,
            community_id INTEGER NOT NULL REFERENCES communities(id)
        );

        CREATE TABLE cluster_books (
            cluster_id INTEGER NOT NULL,
            book_id INTEGER NOT NULL,
            topic_count INTEGER DEFAULT 0,
            book_total_topics INTEGER DEFAULT 0,
            PRIMARY KEY (cluster_id, book_id)
        );

        CREATE TABLE cluster_stats (
            cluster_id INTEGER PRIMARY KEY,
            size INTEGER NOT NULL DEFAULT 0,
            book_count INTEGER NOT NULL DEFAULT 0,
            top_label TEXT,
            top_topics_json TEXT,
            sample_books_json TEXT
        );

        CREATE TABLE book_domains (
            book_id INTEGER NOT NULL,
            domain_id INTEGER NOT NULL,
            topic_count INTEGER NOT NULL,
            book_total_topics INTEGER NOT NULL,
            concentration REAL NOT NULL,
            relevance_score REAL NOT NULL,
            is_primary INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (book_id, domain_id)
        );

        CREATE TABLE domain_stats (
            domain_id INTEGER PRIMARY KEY,
            book_count INTEGER NOT NULL DEFAULT 0,
            primary_book_count INTEGER NOT NULL DEFAULT 0,
            sample_books_json TEXT,
            top_clusters_json TEXT
        );

        CREATE TABLE book_communities (
            book_id INTEGER NOT NULL,
            community_id INTEGER NOT NULL,
            topic_count INTEGER NOT NULL,
            book_total_topics INTEGER NOT NULL,
            concentration REAL NOT NULL,
            relevance_score REAL NOT NULL,
            is_primary INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (book_id, community_id)
        );

        CREATE TABLE community_stats (
            community_id INTEGER PRIMARY KEY,
            topic_count INTEGER NOT NULL DEFAULT 0,
            book_count INTEGER NOT NULL DEFAULT 0,
            primary_book_count INTEGER NOT NULL DEFAULT 0,
            top_label TEXT,
            top_topics_json TEXT,
            sample_books_json TEXT,
            domain_id INTEGER,
            domain_label TEXT,
            refreshed_at TIMESTAMP
        );

        -- Seed: 2 books
        INSERT INTO books (id, title, author, calibre_id) VALUES
            (1, 'Book Alpha', 'Author A', 100),
            (2, 'Book Beta', 'Author B', 200);

        -- Seed: 2 domains
        INSERT INTO domains (id, label, cluster_count) VALUES
            (0, 'Science & Tech', 1),
            (1, 'Literature & Arts', 1);

        -- Seed: cluster→domain mapping
        INSERT INTO cluster_domains (cluster_id, domain_id) VALUES
            (0, 0),
            (1, 1);

        -- Seed: 2 communities
        INSERT INTO communities (id, label, domain_id, topic_count) VALUES
            (0, 'Physics & Math', 0, 3),
            (1, 'Poetry & Fiction', 1, 2);

        -- Seed: cluster→community mapping
        INSERT INTO cluster_communities (cluster_id, community_id) VALUES
            (0, 0),
            (1, 1);

        -- Seed: cluster_books (B1 has 100 total topics, B2 has 50)
        INSERT INTO cluster_books (cluster_id, book_id, topic_count, book_total_topics) VALUES
            (0, 1, 60, 100),
            (1, 1, 40, 100),
            (0, 2, 20, 50),
            (1, 2, 30, 50);

        -- Seed: topics with cluster and community assignments
        INSERT INTO topics (id, label, cluster_id, community_id, occurrence_count) VALUES
            (1, 'quantum mechanics', 0, 0, 15),
            (2, 'linear algebra', 0, 0, 10),
            (3, 'thermodynamics', 0, 0, 8),
            (4, 'romantic poetry', 1, 1, 12),
            (5, 'narrative structure', 1, 1, 9);

        -- Seed: cluster_stats (needed by refresh_domain_stats)
        INSERT INTO cluster_stats (cluster_id, size, book_count, top_label) VALUES
            (0, 3, 2, 'quantum mechanics'),
            (1, 2, 2, 'romantic poetry');
    """)
    conn.commit()
    yield conn
    conn.close()


@pytest.fixture
def sample_chunks():
    """Sample text chunks for testing."""
    return [
        "This is the first chunk of text about philosophy and meaning. " * 50,
        "The second chunk discusses technology and artificial intelligence. " * 50,
        "Chapter three explores relationships and human connection. " * 50,
    ]


@pytest.fixture
def sample_topics():
    """Sample topics for testing."""
    return [
        "Philosophy",
        "Meaning of Life",
        "Technology",
        "Artificial Intelligence",
        "Relationships",
        "Human Connection",
    ]
