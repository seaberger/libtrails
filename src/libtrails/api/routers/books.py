"""Book API endpoints."""

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import DBConnection
from ..schemas import BookDetail, BookSummary, RelatedBook, TopicInfo

router = APIRouter()


@router.get("/books", response_model=list[BookSummary])
def list_books(
    db: DBConnection,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    indexed_only: bool = True,
):
    """List books with pagination."""
    cursor = db.cursor()
    offset = (page - 1) * page_size

    if indexed_only:
        # Only books that have been indexed (have chunks)
        cursor.execute("""
            SELECT DISTINCT b.id, b.title, b.author, b.calibre_id
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            ORDER BY b.title
            LIMIT ? OFFSET ?
        """, (page_size, offset))
    else:
        cursor.execute("""
            SELECT id, title, author, calibre_id
            FROM books
            WHERE calibre_id IS NOT NULL
            ORDER BY title
            LIMIT ? OFFSET ?
        """, (page_size, offset))

    return [BookSummary(**dict(row)) for row in cursor.fetchall()]


@router.get("/books/{book_id}", response_model=BookDetail)
def get_book(db: DBConnection, book_id: int):
    """Get book detail with topics."""
    cursor = db.cursor()

    # Get book
    cursor.execute("""
        SELECT id, title, author, calibre_id, description
        FROM books
        WHERE id = ?
    """, (book_id,))
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Book not found")

    book = dict(row)

    # Get chunk count
    cursor.execute("SELECT COUNT(*) as cnt FROM chunks WHERE book_id = ?", (book_id,))
    chunk_count = cursor.fetchone()["cnt"]

    # Get topics for this book with occurrence counts
    cursor.execute("""
        SELECT t.id, t.label, COUNT(*) as count, t.cluster_id
        FROM topics t
        JOIN chunk_topic_links ctl ON ctl.topic_id = t.id
        JOIN chunks c ON c.id = ctl.chunk_id
        WHERE c.book_id = ?
        GROUP BY t.id
        ORDER BY count DESC
        LIMIT 30
    """, (book_id,))
    topics = [TopicInfo(**dict(r)) for r in cursor.fetchall()]

    # Get unique theme IDs
    theme_ids = list({t.cluster_id for t in topics if t.cluster_id is not None})

    return BookDetail(
        id=book["id"],
        title=book["title"],
        author=book["author"],
        calibre_id=book["calibre_id"],
        description=book.get("description"),
        topics=topics,
        theme_ids=theme_ids,
        chunk_count=chunk_count,
    )


@router.get("/books/{book_id}/related", response_model=list[RelatedBook])
def get_related_books(
    db: DBConnection,
    book_id: int,
    limit: int = Query(10, ge=1, le=50),
):
    """Get related books by topic overlap."""
    cursor = db.cursor()

    # Verify book exists
    cursor.execute("SELECT id FROM books WHERE id = ?", (book_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Book not found")

    # Get topics for source book
    cursor.execute("""
        SELECT DISTINCT t.id
        FROM topics t
        JOIN chunk_topic_links ctl ON ctl.topic_id = t.id
        JOIN chunks c ON c.id = ctl.chunk_id
        WHERE c.book_id = ?
    """, (book_id,))
    source_topics = {row["id"] for row in cursor.fetchall()}

    if not source_topics:
        return []

    # Find books sharing topics
    placeholders = ",".join("?" * len(source_topics))
    cursor.execute(f"""
        SELECT
            b.id, b.title, b.author, b.calibre_id,
            COUNT(DISTINCT t.id) as shared_topics
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.id IN ({placeholders})
        AND b.id != ?
        GROUP BY b.id
        ORDER BY shared_topics DESC
        LIMIT ?
    """, (*source_topics, book_id, limit))

    results = []
    for row in cursor.fetchall():
        shared = row["shared_topics"]
        similarity = shared / len(source_topics) if source_topics else 0
        results.append(RelatedBook(
            id=row["id"],
            title=row["title"],
            author=row["author"],
            calibre_id=row["calibre_id"],
            shared_topics=shared,
            similarity=round(similarity, 3),
        ))

    return results
