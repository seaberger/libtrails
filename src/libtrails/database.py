"""Database operations for libtrails."""

import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

from .config import IPAD_DB_PATH, CALIBRE_DB_PATH, CALIBRE_LIBRARY_PATH


@contextmanager
def get_db(db_path: Path = IPAD_DB_PATH):
    """Get a database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def get_calibre_db():
    """Get a read-only connection to Calibre database."""
    conn = sqlite3.connect(f"file:{CALIBRE_DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_book(book_id: int) -> Optional[dict]:
    """Get a book by ID from the iPad library."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_book_by_title(title: str) -> Optional[dict]:
    """Get a book by title (partial match)."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM books WHERE title LIKE ? LIMIT 1",
            (f"%{title}%",)
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def get_all_books(with_calibre_match: bool = True) -> list[dict]:
    """Get all books from the iPad library."""
    with get_db() as conn:
        cursor = conn.cursor()
        if with_calibre_match:
            cursor.execute("SELECT * FROM books WHERE calibre_id IS NOT NULL")
        else:
            cursor.execute("SELECT * FROM books")
        return [dict(row) for row in cursor.fetchall()]


def get_epub_path(calibre_id: int) -> Optional[Path]:
    """Get the path to a book's EPUB file in Calibre library."""
    with get_calibre_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM books WHERE id = ?", (calibre_id,))
        row = cursor.fetchone()
        if not row:
            return None

        book_dir = CALIBRE_LIBRARY_PATH / row['path']

        # Find EPUB file
        for epub in book_dir.glob("*.epub"):
            return epub

        return None


def get_book_path(calibre_id: int) -> Optional[Path]:
    """
    Get the path to a book file in Calibre library.

    Prefers EPUB over PDF if both exist.
    Returns None if no supported format found.
    """
    with get_calibre_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT path FROM books WHERE id = ?", (calibre_id,))
        row = cursor.fetchone()
        if not row:
            return None

        book_dir = CALIBRE_LIBRARY_PATH / row['path']

        # Prefer EPUB
        for epub in book_dir.glob("*.epub"):
            return epub

        # Fall back to PDF
        for pdf in book_dir.glob("*.pdf"):
            return pdf

        return None


def get_book_format(calibre_id: int) -> Optional[str]:
    """Get the format of the book file (epub or pdf)."""
    path = get_book_path(calibre_id)
    if path:
        return path.suffix.lower().lstrip('.')
    return None


def init_chunks_table():
    """Create the chunks and topics tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
            -- Original chunks table
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                book_id INTEGER REFERENCES books(id),
                chunk_index INTEGER,
                text TEXT NOT NULL,
                word_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(book_id, chunk_index)
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_book ON chunks(book_id);

            -- Original chunk_topics table (raw topics per chunk)
            CREATE TABLE IF NOT EXISTS chunk_topics (
                chunk_id INTEGER REFERENCES chunks(id),
                topic TEXT NOT NULL,
                PRIMARY KEY (chunk_id, topic)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_topics_topic ON chunk_topics(topic);

            -- Normalized topics table with embeddings
            CREATE TABLE IF NOT EXISTS topics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                label TEXT UNIQUE NOT NULL,
                embedding BLOB,
                cluster_id INTEGER,
                parent_topic_id INTEGER REFERENCES topics(id),
                occurrence_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_topics_label ON topics(label);
            CREATE INDEX IF NOT EXISTS idx_topics_cluster ON topics(cluster_id);

            -- Link table for normalized topics to chunks
            CREATE TABLE IF NOT EXISTS chunk_topic_links (
                chunk_id INTEGER REFERENCES chunks(id),
                topic_id INTEGER REFERENCES topics(id),
                PRIMARY KEY (chunk_id, topic_id)
            );

            CREATE INDEX IF NOT EXISTS idx_chunk_topic_links_topic ON chunk_topic_links(topic_id);
            CREATE INDEX IF NOT EXISTS idx_chunk_topic_links_chunk ON chunk_topic_links(chunk_id);

            -- Topic co-occurrence for graph building
            CREATE TABLE IF NOT EXISTS topic_cooccurrences (
                topic1_id INTEGER REFERENCES topics(id),
                topic2_id INTEGER REFERENCES topics(id),
                count INTEGER DEFAULT 0,
                pmi REAL,
                PRIMARY KEY (topic1_id, topic2_id)
            );

            CREATE INDEX IF NOT EXISTS idx_cooccur_topic1 ON topic_cooccurrences(topic1_id);
            CREATE INDEX IF NOT EXISTS idx_cooccur_topic2 ON topic_cooccurrences(topic2_id);
        """)
        conn.commit()


def save_chunks(book_id: int, chunks: list[str]):
    """Save chunks for a book."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Delete existing chunks for this book
        cursor.execute("DELETE FROM chunk_topics WHERE chunk_id IN (SELECT id FROM chunks WHERE book_id = ?)", (book_id,))
        cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))

        # Insert new chunks
        for i, text in enumerate(chunks):
            cursor.execute(
                "INSERT INTO chunks (book_id, chunk_index, text, word_count) VALUES (?, ?, ?, ?)",
                (book_id, i, text, len(text.split()))
            )

        conn.commit()


def save_chunk_topics(chunk_id: int, topics: list[str]):
    """Save topics for a chunk."""
    with get_db() as conn:
        cursor = conn.cursor()
        for topic in topics:
            cursor.execute(
                "INSERT OR IGNORE INTO chunk_topics (chunk_id, topic) VALUES (?, ?)",
                (chunk_id, topic.strip())
            )
        conn.commit()


def get_indexing_status() -> dict:
    """Get the current indexing status."""
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM books WHERE calibre_id IS NOT NULL")
        total_books = cursor.fetchone()[0]

        # Check if chunks table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(DISTINCT book_id) FROM chunks")
            indexed_books = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM chunks")
            total_chunks = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(DISTINCT topic) FROM chunk_topics")
            unique_topics = cursor.fetchone()[0]
        else:
            indexed_books = 0
            total_chunks = 0
            unique_topics = 0

        # Get normalized topic stats
        topic_stats = get_topic_stats()

        return {
            "total_books": total_books,
            "indexed_books": indexed_books,
            "total_chunks": total_chunks,
            "unique_topics": unique_topics,
            "normalized_topics": topic_stats["total_topics"],
            "topics_with_embeddings": topic_stats["with_embeddings"],
            "clustered_topics": topic_stats["clustered"],
        }


def get_or_create_topic(label: str) -> int:
    """Get or create a normalized topic and return its ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM topics WHERE label = ?", (label,))
        row = cursor.fetchone()
        if row:
            return row[0]
        
        cursor.execute(
            "INSERT INTO topics (label, occurrence_count) VALUES (?, 1)",
            (label,)
        )
        conn.commit()
        return cursor.lastrowid


def increment_topic_count(topic_id: int):
    """Increment the occurrence count for a topic."""
    with get_db() as conn:
        conn.execute(
            "UPDATE topics SET occurrence_count = occurrence_count + 1 WHERE id = ?",
            (topic_id,)
        )
        conn.commit()


def link_chunk_to_topic(chunk_id: int, topic_id: int):
    """Link a chunk to a normalized topic."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chunk_topic_links (chunk_id, topic_id) VALUES (?, ?)",
            (chunk_id, topic_id)
        )
        conn.commit()


def get_all_topics() -> list[dict]:
    """Get all topics with their metadata."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, label, embedding IS NOT NULL as has_embedding,
                   cluster_id, parent_topic_id, occurrence_count
            FROM topics
            ORDER BY occurrence_count DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_topics_without_embeddings() -> list[dict]:
    """Get topics that don't have embeddings yet."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, label FROM topics WHERE embedding IS NULL
            ORDER BY id
        """)
        return [dict(row) for row in cursor.fetchall()]


def save_topic_embedding(topic_id: int, embedding: bytes):
    """Save an embedding for a topic."""
    with get_db() as conn:
        conn.execute(
            "UPDATE topics SET embedding = ? WHERE id = ?",
            (embedding, topic_id)
        )
        conn.commit()


def get_topic_embeddings() -> list[tuple[int, bytes]]:
    """Get all topic embeddings."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, embedding FROM topics WHERE embedding IS NOT NULL
        """)
        return [(row[0], row[1]) for row in cursor.fetchall()]


def update_topic_cluster(topic_id: int, cluster_id: int, parent_id: Optional[int] = None):
    """Update the cluster assignment for a topic."""
    with get_db() as conn:
        conn.execute(
            "UPDATE topics SET cluster_id = ?, parent_topic_id = ? WHERE id = ?",
            (cluster_id, parent_id, topic_id)
        )
        conn.commit()


def save_cooccurrence(topic1_id: int, topic2_id: int, count: int, pmi: Optional[float] = None):
    """Save or update topic co-occurrence data."""
    # Ensure consistent ordering
    if topic1_id > topic2_id:
        topic1_id, topic2_id = topic2_id, topic1_id
    
    with get_db() as conn:
        conn.execute("""
            INSERT INTO topic_cooccurrences (topic1_id, topic2_id, count, pmi)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(topic1_id, topic2_id) DO UPDATE SET
                count = excluded.count,
                pmi = excluded.pmi
        """, (topic1_id, topic2_id, count, pmi))
        conn.commit()


def get_cooccurrences() -> list[dict]:
    """Get all topic co-occurrences."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT tc.topic1_id, tc.topic2_id, tc.count, tc.pmi,
                   t1.label as label1, t2.label as label2
            FROM topic_cooccurrences tc
            JOIN topics t1 ON tc.topic1_id = t1.id
            JOIN topics t2 ON tc.topic2_id = t2.id
            ORDER BY tc.count DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def migrate_raw_topics_to_normalized():
    """Migrate existing raw topics from chunk_topics to normalized topics table."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Get all unique raw topics with their counts
        cursor.execute("""
            SELECT topic, COUNT(*) as count
            FROM chunk_topics
            GROUP BY topic
        """)
        raw_topics = cursor.fetchall()
        
        migrated = 0
        for topic, count in raw_topics:
            # Normalize and insert
            from .topic_extractor import normalize_topic
            normalized = normalize_topic(topic)
            
            # Get or create the normalized topic
            cursor.execute("SELECT id FROM topics WHERE label = ?", (normalized,))
            row = cursor.fetchone()
            if row:
                topic_id = row[0]
                cursor.execute(
                    "UPDATE topics SET occurrence_count = occurrence_count + ? WHERE id = ?",
                    (count, topic_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO topics (label, occurrence_count) VALUES (?, ?)",
                    (normalized, count)
                )
                topic_id = cursor.lastrowid
            
            # Link chunks to the normalized topic
            cursor.execute("""
                INSERT OR IGNORE INTO chunk_topic_links (chunk_id, topic_id)
                SELECT chunk_id, ? FROM chunk_topics WHERE topic = ?
            """, (topic_id, topic))
            
            migrated += 1
        
        conn.commit()
        return migrated


def get_topic_stats() -> dict:
    """Get statistics about the topics table."""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if topics table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='topics'")
        if not cursor.fetchone():
            return {
                "total_topics": 0,
                "with_embeddings": 0,
                "clustered": 0,
                "total_cooccurrences": 0,
            }
        
        cursor.execute("SELECT COUNT(*) FROM topics")
        total = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM topics WHERE embedding IS NOT NULL")
        with_embeddings = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM topics WHERE cluster_id IS NOT NULL")
        clustered = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM topic_cooccurrences")
        cooccurrences = cursor.fetchone()[0]
        
        return {
            "total_topics": total,
            "with_embeddings": with_embeddings,
            "clustered": clustered,
            "total_cooccurrences": cooccurrences,
        }
