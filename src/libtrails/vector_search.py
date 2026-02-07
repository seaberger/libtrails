"""Vector search using sqlite-vec for semantic topic search."""

import sqlite3
from pathlib import Path

from .config import IPAD_DB_PATH
from .embeddings import embed_text, embedding_to_bytes, get_embedding_dimension


def init_vector_search(conn: sqlite3.Connection, force_recreate: bool = False):
    """
    Initialize sqlite-vec extension and create vector table.

    Uses cosine distance metric for normalized embeddings.
    """
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    dim = get_embedding_dimension()

    if force_recreate:
        # Drop existing table to recreate with new settings
        conn.execute("DROP TABLE IF EXISTS topic_vectors")

    # Create with cosine distance metric for semantic similarity
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS topic_vectors USING vec0(
            topic_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}] distance_metric=cosine
        )
    """)
    conn.commit()


def get_vec_db(db_path: Path = IPAD_DB_PATH) -> sqlite3.Connection:
    """Get a database connection with sqlite-vec loaded."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    init_vector_search(conn)
    return conn


def index_topic_vector(conn: sqlite3.Connection, topic_id: int, embedding: bytes):
    """Add or update a topic's embedding in the vector index."""
    # Delete existing entry if present
    conn.execute("DELETE FROM topic_vectors WHERE topic_id = ?", (topic_id,))
    # Insert new embedding
    conn.execute(
        "INSERT INTO topic_vectors (topic_id, embedding) VALUES (?, ?)", (topic_id, embedding)
    )


def rebuild_vector_index(conn: sqlite3.Connection, force_recreate: bool = False):
    """
    Rebuild the entire vector index from the topics table.

    Args:
        conn: Database connection
        force_recreate: If True, drop and recreate the table (needed for schema changes)
    """
    cursor = conn.cursor()

    if force_recreate:
        # Recreate table with correct distance metric
        init_vector_search(conn, force_recreate=True)

    # Clear existing vectors
    conn.execute("DELETE FROM topic_vectors")

    # Get all topics with embeddings
    cursor.execute("SELECT id, embedding FROM topics WHERE embedding IS NOT NULL")

    count = 0
    for row in cursor.fetchall():
        conn.execute(
            "INSERT INTO topic_vectors (topic_id, embedding) VALUES (?, ?)",
            (row["id"], row["embedding"]),
        )
        count += 1

    conn.commit()
    return count


def search_topics_semantic(query: str, limit: int = 20, db_path: Path = IPAD_DB_PATH) -> list[dict]:
    """
    Search for topics semantically using vector similarity.

    Uses cosine distance - lower distance = more similar.
    For normalized embeddings: similarity = 1 - cosine_distance

    Args:
        query: The search query text
        limit: Maximum number of results

    Returns:
        List of dicts with topic_id, label, similarity, and occurrence_count
    """
    conn = get_vec_db(db_path)
    cursor = conn.cursor()

    # Generate query embedding
    query_embedding = embed_text(query)
    query_bytes = embedding_to_bytes(query_embedding)

    # Vector similarity search - sqlite-vec requires k=? in WHERE clause
    cursor.execute(
        """
        SELECT
            tv.topic_id,
            tv.distance,
            t.label,
            t.occurrence_count,
            t.cluster_id
        FROM topic_vectors tv
        JOIN topics t ON tv.topic_id = t.id
        WHERE tv.embedding MATCH ? AND k = ?
        ORDER BY tv.distance
    """,
        (query_bytes, limit),
    )

    results = []
    for row in cursor.fetchall():
        # Cosine distance ranges from 0 (identical) to 2 (opposite)
        # Convert to similarity: 1 - (distance / 2) gives range [0, 1]
        # Or simpler: similarity = 1 - distance for normalized vectors
        distance = row["distance"]
        similarity = 1.0 - distance

        results.append(
            {
                "topic_id": row["topic_id"],
                "label": row["label"],
                "distance": distance,
                "similarity": similarity,
                "occurrence_count": row["occurrence_count"],
                "cluster_id": row["cluster_id"],
            }
        )

    conn.close()
    return results


def search_books_by_topic_semantic(
    query: str, limit: int = 20, db_path: Path = IPAD_DB_PATH
) -> list[dict]:
    """
    Search for books that contain topics semantically similar to the query.

    Args:
        query: The search query text
        limit: Maximum number of books to return

    Returns:
        List of dicts with book info and matching topics
    """
    conn = get_vec_db(db_path)
    cursor = conn.cursor()

    # Generate query embedding
    query_embedding = embed_text(query)
    query_bytes = embedding_to_bytes(query_embedding)

    # Find matching topics first - sqlite-vec requires k=? in WHERE clause
    cursor.execute(
        """
        SELECT tv.topic_id, tv.distance, t.label
        FROM topic_vectors tv
        JOIN topics t ON tv.topic_id = t.id
        WHERE tv.embedding MATCH ? AND k = 50
        ORDER BY tv.distance
    """,
        (query_bytes,),
    )

    matching_topics = cursor.fetchall()
    if not matching_topics:
        conn.close()
        return []

    # Build a mapping of topic_id to distance for scoring
    topic_distances = {row["topic_id"]: row["distance"] for row in matching_topics}
    topic_ids = list(topic_distances.keys())
    placeholders = ",".join("?" * len(topic_ids))

    cursor.execute(
        f"""
        SELECT
            b.id, b.title, b.author,
            GROUP_CONCAT(DISTINCT t.label) as matching_topics,
            GROUP_CONCAT(DISTINCT ctl.topic_id) as topic_id_list,
            COUNT(DISTINCT ctl.topic_id) as match_count
        FROM books b
        JOIN chunks c ON b.id = c.book_id
        JOIN chunk_topic_links ctl ON c.id = ctl.chunk_id
        JOIN topics t ON ctl.topic_id = t.id
        WHERE ctl.topic_id IN ({placeholders})
        GROUP BY b.id
        ORDER BY match_count DESC
        LIMIT ?
    """,
        (*topic_ids, limit),
    )

    results = []
    for row in cursor.fetchall():
        # Calculate best distance from the matched topics
        matched_ids = (
            [int(x) for x in row["topic_id_list"].split(",")] if row["topic_id_list"] else []
        )
        best_distance = min((topic_distances.get(tid, 1.0) for tid in matched_ids), default=1.0)

        results.append(
            {
                "id": row["id"],
                "title": row["title"],
                "author": row["author"],
                "matching_topics": row["matching_topics"].split(",")
                if row["matching_topics"]
                else [],
                "match_count": row["match_count"],
                "relevance": 1.0 - best_distance,
            }
        )

    conn.close()
    return results


def get_vector_index_stats(db_path: Path = IPAD_DB_PATH) -> dict:
    """Get statistics about the vector index."""
    try:
        conn = get_vec_db(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM topic_vectors")
        indexed = cursor.fetchone()[0]

        conn.close()
        return {"indexed_vectors": indexed}
    except Exception as e:
        return {"indexed_vectors": 0, "error": str(e)}
