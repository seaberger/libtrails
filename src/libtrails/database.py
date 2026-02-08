"""Database operations for libtrails."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .config import CALIBRE_DB_PATH, CALIBRE_LIBRARY_PATH, IPAD_DB_PATH


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
        cursor.execute("SELECT * FROM books WHERE title LIKE ? LIMIT 1", (f"%{title}%",))
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

        book_dir = CALIBRE_LIBRARY_PATH / row["path"]

        # Find EPUB file
        epub_files = list(book_dir.glob("*.epub"))
        if epub_files:
            return epub_files[0]

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

        book_dir = CALIBRE_LIBRARY_PATH / row["path"]

        # Prefer EPUB
        epub_files = list(book_dir.glob("*.epub"))
        if epub_files:
            return epub_files[0]

        # Fall back to PDF
        pdf_files = list(book_dir.glob("*.pdf"))
        if pdf_files:
            return pdf_files[0]

        return None


def get_book_format(calibre_id: int) -> Optional[str]:
    """Get the format of the book file (epub or pdf)."""
    path = get_book_path(calibre_id)
    if path:
        return path.suffix.lower().lstrip(".")
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
                topics_json TEXT,
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

            -- Multi-cluster membership for hub topics
            -- Allows topics (especially hubs) to belong to multiple clusters with strength scores
            CREATE TABLE IF NOT EXISTS topic_cluster_memberships (
                topic_id INTEGER NOT NULL REFERENCES topics(id),
                cluster_id INTEGER NOT NULL,
                strength REAL NOT NULL,  -- 0.0 to 1.0, proportion of connections
                is_primary BOOLEAN DEFAULT FALSE,
                PRIMARY KEY (topic_id, cluster_id)
            );

            CREATE INDEX IF NOT EXISTS idx_tcm_cluster ON topic_cluster_memberships(cluster_id);
            CREATE INDEX IF NOT EXISTS idx_tcm_topic ON topic_cluster_memberships(topic_id);

            -- Cluster labels (LLM-generated)
            CREATE TABLE IF NOT EXISTS cluster_labels (
                cluster_id INTEGER PRIMARY KEY,
                label TEXT NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Domains (super-clusters) - high-level groupings of Leiden clusters
            CREATE TABLE IF NOT EXISTS domains (
                id INTEGER PRIMARY KEY,
                label TEXT NOT NULL UNIQUE,
                cluster_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Mapping of Leiden clusters to domains
            CREATE TABLE IF NOT EXISTS cluster_domains (
                cluster_id INTEGER PRIMARY KEY,
                domain_id INTEGER NOT NULL REFERENCES domains(id)
            );

            CREATE INDEX IF NOT EXISTS idx_cluster_domains_domain ON cluster_domains(domain_id);

            -- Materialized stats: bridge table eliminates 4-table join for book lookups
            CREATE TABLE IF NOT EXISTS cluster_books (
                cluster_id INTEGER NOT NULL,
                book_id INTEGER NOT NULL,
                topic_count INTEGER DEFAULT 0,
                PRIMARY KEY (cluster_id, book_id)
            );

            CREATE INDEX IF NOT EXISTS idx_cluster_books_book ON cluster_books(book_id);

            -- Materialized stats: per-cluster cached stats with JSON blobs
            CREATE TABLE IF NOT EXISTS cluster_stats (
                cluster_id INTEGER PRIMARY KEY,
                size INTEGER NOT NULL DEFAULT 0,
                book_count INTEGER NOT NULL DEFAULT 0,
                top_label TEXT,
                top_topics_json TEXT,
                sample_books_json TEXT,
                refreshed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            -- Materialized stats: per-domain cached stats
            CREATE TABLE IF NOT EXISTS domain_stats (
                domain_id INTEGER PRIMARY KEY,
                book_count INTEGER NOT NULL DEFAULT 0,
                sample_books_json TEXT,
                top_clusters_json TEXT,
                refreshed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Migration: add topics_json column if it doesn't exist
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(chunks)")
        columns = [row[1] for row in cursor.fetchall()]
        if "topics_json" not in columns:
            conn.execute("ALTER TABLE chunks ADD COLUMN topics_json TEXT")

        conn.commit()


def save_chunks(book_id: int, chunks: list[str]):
    """Save chunks for a book."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Delete existing chunks for this book
        cursor.execute(
            "DELETE FROM chunk_topics WHERE chunk_id IN (SELECT id FROM chunks WHERE book_id = ?)",
            (book_id,),
        )
        cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))

        # Insert new chunks
        for i, text in enumerate(chunks):
            cursor.execute(
                "INSERT INTO chunks (book_id, chunk_index, text, word_count) VALUES (?, ?, ?, ?)",
                (book_id, i, text, len(text.split())),
            )

        conn.commit()


def save_chunk_topics(chunk_id: int, topics: list[str]):
    """Save topics for a chunk (both normalized table and JSON column)."""
    import json

    cleaned_topics = [t.strip() for t in topics if t.strip()]

    with get_db() as conn:
        cursor = conn.cursor()

        # Save to chunk_topics table (normalized)
        for topic in cleaned_topics:
            cursor.execute(
                "INSERT OR IGNORE INTO chunk_topics (chunk_id, topic) VALUES (?, ?)",
                (chunk_id, topic),
            )

        # Save JSON to chunks table (denormalized for RAG)
        cursor.execute(
            "UPDATE chunks SET topics_json = ? WHERE id = ?", (json.dumps(cleaned_topics), chunk_id)
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

        cursor.execute("INSERT INTO topics (label, occurrence_count) VALUES (?, 1)", (label,))
        conn.commit()
        return cursor.lastrowid


def increment_topic_count(topic_id: int):
    """Increment the occurrence count for a topic."""
    with get_db() as conn:
        conn.execute(
            "UPDATE topics SET occurrence_count = occurrence_count + 1 WHERE id = ?", (topic_id,)
        )
        conn.commit()


def link_chunk_to_topic(chunk_id: int, topic_id: int):
    """Link a chunk to a normalized topic."""
    with get_db() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO chunk_topic_links (chunk_id, topic_id) VALUES (?, ?)",
            (chunk_id, topic_id),
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
        conn.execute("UPDATE topics SET embedding = ? WHERE id = ?", (embedding, topic_id))
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
            (cluster_id, parent_id, topic_id),
        )
        conn.commit()


def save_cooccurrence(topic1_id: int, topic2_id: int, count: int, pmi: Optional[float] = None):
    """Save or update topic co-occurrence data."""
    # Ensure consistent ordering
    if topic1_id > topic2_id:
        topic1_id, topic2_id = topic2_id, topic1_id

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO topic_cooccurrences (topic1_id, topic2_id, count, pmi)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(topic1_id, topic2_id) DO UPDATE SET
                count = excluded.count,
                pmi = excluded.pmi
        """,
            (topic1_id, topic2_id, count, pmi),
        )
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
    """
    Migrate existing raw topics from chunk_topics to normalized topics table.

    This function is idempotent - running it multiple times produces the same result.
    It creates normalized topics, links chunks to them, and recalculates occurrence
    counts based on actual links (not by summing, which would cause double-counting).
    """
    with get_db() as conn:
        cursor = conn.cursor()

        # Get all unique raw topics
        cursor.execute("SELECT DISTINCT topic FROM chunk_topics")
        raw_topics = cursor.fetchall()

        from .topic_extractor import normalize_topic

        migrated = 0
        for (topic,) in raw_topics:
            normalized = normalize_topic(topic)

            # Get or create the normalized topic (with initial count of 0)
            cursor.execute("SELECT id FROM topics WHERE label = ?", (normalized,))
            row = cursor.fetchone()
            if row:
                topic_id = row[0]
            else:
                cursor.execute(
                    "INSERT INTO topics (label, occurrence_count) VALUES (?, 0)", (normalized,)
                )
                topic_id = cursor.lastrowid
                migrated += 1

            # Link chunks to the normalized topic (INSERT OR IGNORE is idempotent)
            cursor.execute(
                """
                INSERT OR IGNORE INTO chunk_topic_links (chunk_id, topic_id)
                SELECT chunk_id, ? FROM chunk_topics WHERE topic = ?
            """,
                (topic_id, topic),
            )

        # Recalculate all occurrence_counts from actual links (idempotent)
        cursor.execute("""
            UPDATE topics
            SET occurrence_count = (
                SELECT COUNT(*) FROM chunk_topic_links WHERE topic_id = topics.id
            )
        """)

        conn.commit()
        return migrated


def load_domains_from_json(json_path: Path):
    """
    Load domains from the final labels JSON file into the database.

    This clears existing domain data and reloads from the JSON.
    """
    import json

    with open(json_path) as f:
        domains = json.load(f)

    with get_db() as conn:
        cursor = conn.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM cluster_domains")
        cursor.execute("DELETE FROM domains")

        # Insert domains
        for d in domains:
            cursor.execute(
                "INSERT INTO domains (id, label, cluster_count) VALUES (?, ?, ?)",
                (d["domain_id"], d["label"], d["cluster_count"]),
            )

            # Insert cluster mappings
            for cluster_id in d["leiden_cluster_ids"]:
                cursor.execute(
                    "INSERT OR REPLACE INTO cluster_domains (cluster_id, domain_id) VALUES (?, ?)",
                    (cluster_id, d["domain_id"]),
                )

        conn.commit()

    return len(domains)


def get_all_domains() -> list[dict]:
    """Get all domains with their metadata."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.id, d.label, d.cluster_count,
                   COUNT(DISTINCT cd.cluster_id) as actual_clusters
            FROM domains d
            LEFT JOIN cluster_domains cd ON cd.domain_id = d.id
            GROUP BY d.id
            ORDER BY d.cluster_count DESC
        """)
        return [dict(row) for row in cursor.fetchall()]


def get_domain(domain_id: int) -> Optional[dict]:
    """Get a single domain by ID."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM domains WHERE id = ?", (domain_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def get_clusters_for_domain(domain_id: int) -> list[int]:
    """Get all Leiden cluster IDs belonging to a domain."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT cluster_id FROM cluster_domains WHERE domain_id = ?", (domain_id,))
        return [row[0] for row in cursor.fetchall()]


def get_domain_for_cluster(cluster_id: int) -> Optional[dict]:
    """Get the domain for a given Leiden cluster."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT d.* FROM domains d
            JOIN cluster_domains cd ON cd.domain_id = d.id
            WHERE cd.cluster_id = ?
        """,
            (cluster_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


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
