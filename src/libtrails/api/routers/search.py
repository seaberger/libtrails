"""Search API endpoints."""

from fastapi import APIRouter, Query

from ..dependencies import DBConnection
from ..schemas import BookSummary, SearchResult

router = APIRouter()


@router.get("/search", response_model=list[SearchResult])
def search_books(
    db: DBConnection,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search books by title, author, or topic."""
    cursor = db.cursor()
    results = []
    seen_ids = set()

    # 1. Title matches (highest priority)
    cursor.execute("""
        SELECT id, title, author, calibre_id
        FROM books
        WHERE title LIKE ?
        ORDER BY title
        LIMIT ?
    """, (f"%{q}%", limit))

    for row in cursor.fetchall():
        if row["id"] not in seen_ids:
            results.append(SearchResult(
                book=BookSummary(**dict(row)),
                score=1.0,
                match_type="keyword",
            ))
            seen_ids.add(row["id"])

    # 2. Author matches
    cursor.execute("""
        SELECT id, title, author, calibre_id
        FROM books
        WHERE author LIKE ?
        ORDER BY title
        LIMIT ?
    """, (f"%{q}%", limit))

    for row in cursor.fetchall():
        if row["id"] not in seen_ids:
            results.append(SearchResult(
                book=BookSummary(**dict(row)),
                score=0.8,
                match_type="keyword",
            ))
            seen_ids.add(row["id"])

    # 3. Topic matches - find books containing matching topics
    cursor.execute("""
        SELECT DISTINCT b.id, b.title, b.author, b.calibre_id, t.label
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.label LIKE ?
        ORDER BY b.title
        LIMIT ?
    """, (f"%{q}%", limit))

    for row in cursor.fetchall():
        if row["id"] not in seen_ids:
            results.append(SearchResult(
                book=BookSummary(
                    id=row["id"],
                    title=row["title"],
                    author=row["author"],
                    calibre_id=row["calibre_id"],
                ),
                score=0.6,
                match_type="keyword",
            ))
            seen_ids.add(row["id"])

    # Sort by score descending, limit results
    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]


@router.get("/search/semantic", response_model=list[SearchResult])
def semantic_search(
    db: DBConnection,
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(20, ge=1, le=100),
):
    """Semantic search using embeddings (requires topic_vectors table)."""
    cursor = db.cursor()

    # Check if vector table exists
    cursor.execute("""
        SELECT name FROM sqlite_master
        WHERE type='table' AND name='topic_vectors'
    """)
    if not cursor.fetchone():
        return []  # Fall back to empty if no embeddings

    # Import embedding function lazily
    try:
        from ...embeddings import embed_text
    except ImportError:
        return []

    # Embed the query
    query_embedding = embed_text(q)

    # Search using sqlite-vec
    cursor.execute("""
        SELECT topic_id, distance
        FROM topic_vectors
        WHERE embedding MATCH ? AND k = ?
        ORDER BY distance
    """, (query_embedding.tobytes(), limit * 3))

    topic_results = cursor.fetchall()
    if not topic_results:
        return []

    # Get books for these topics
    topic_ids = [r["topic_id"] for r in topic_results]
    topic_distances = {r["topic_id"]: r["distance"] for r in topic_results}

    placeholders = ",".join("?" * len(topic_ids))
    cursor.execute(f"""
        SELECT DISTINCT b.id, b.title, b.author, b.calibre_id, t.id as topic_id
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.id IN ({placeholders})
    """, topic_ids)

    seen_ids = set()
    results = []

    for row in cursor.fetchall():
        if row["id"] not in seen_ids:
            distance = topic_distances.get(row["topic_id"], 1.0)
            similarity = max(0, 1 - distance)  # Convert distance to similarity

            results.append(SearchResult(
                book=BookSummary(
                    id=row["id"],
                    title=row["title"],
                    author=row["author"],
                    calibre_id=row["calibre_id"],
                ),
                score=round(similarity, 3),
                match_type="semantic",
            ))
            seen_ids.add(row["id"])

    results.sort(key=lambda x: x.score, reverse=True)
    return results[:limit]
