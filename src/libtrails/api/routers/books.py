"""Book API endpoints."""

from fastapi import APIRouter, Body, HTTPException, Query

from ...hybrid_search import find_related_books
from ..dependencies import DBConnection
from ..schemas import BookDetail, BookSummary, RelatedBook, ThemeRef, TopicInfo

router = APIRouter()


@router.post("/books/batch", response_model=list[BookSummary])
def get_books_batch(db: DBConnection, book_ids: list[int] = Body(...)):
    """Get multiple books by ID for multi-select sidebar."""
    if not book_ids or len(book_ids) > 50:
        return []
    placeholders = ",".join("?" * len(book_ids))
    cursor = db.cursor()
    cursor.execute(
        f"SELECT id, title, author, calibre_id FROM books WHERE id IN ({placeholders})",
        book_ids,
    )
    return [BookSummary(**dict(row)) for row in cursor.fetchall()]


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
        cursor.execute(
            """
            SELECT DISTINCT b.id, b.title, b.author, b.calibre_id
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            ORDER BY b.title
            LIMIT ? OFFSET ?
        """,
            (page_size, offset),
        )
    else:
        cursor.execute(
            """
            SELECT id, title, author, calibre_id
            FROM books
            WHERE calibre_id IS NOT NULL
            ORDER BY title
            LIMIT ? OFFSET ?
        """,
            (page_size, offset),
        )

    return [BookSummary(**dict(row)) for row in cursor.fetchall()]


@router.get("/books/{book_id}", response_model=BookDetail)
def get_book(db: DBConnection, book_id: int):
    """Get book detail with topics."""
    cursor = db.cursor()

    # Get book
    cursor.execute(
        """
        SELECT id, title, author, calibre_id, description, ipad_id
        FROM books
        WHERE id = ?
    """,
        (book_id,),
    )
    row = cursor.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Book not found")

    book = dict(row)

    # Build Gutenberg URL from ipad_id (format: "gutenberg:1342")
    gutenberg_url = None
    ipad_id = book.pop("ipad_id", None)
    if ipad_id and str(ipad_id).startswith("gutenberg:"):
        gid = str(ipad_id).split(":", 1)[1]
        gutenberg_url = f"https://www.gutenberg.org/ebooks/{gid}"

    # Get chunk count
    cursor.execute("SELECT COUNT(*) as cnt FROM chunks WHERE book_id = ?", (book_id,))
    chunk_count = cursor.fetchone()["cnt"]

    # Get topics for this book with occurrence counts
    cursor.execute(
        """
        SELECT t.id, t.label, COUNT(*) as count, t.cluster_id, cc.community_id
        FROM topics t
        JOIN chunk_topic_links ctl ON ctl.topic_id = t.id
        JOIN chunks c ON c.id = ctl.chunk_id
        LEFT JOIN cluster_communities cc ON cc.cluster_id = t.cluster_id
        WHERE c.book_id = ?
        GROUP BY t.id
        ORDER BY count DESC
        LIMIT 30
    """,
        (book_id,),
    )
    topics = [TopicInfo(**dict(r)) for r in cursor.fetchall()]

    # Get unique themes with labels, mapped to communities
    cluster_ids = list({t.cluster_id for t in topics if t.cluster_id is not None})
    themes = []
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        cursor.execute(
            f"""
            SELECT cs.cluster_id, cs.top_label, cc.community_id
            FROM cluster_stats cs
            LEFT JOIN cluster_communities cc ON cc.cluster_id = cs.cluster_id
            WHERE cs.cluster_id IN ({placeholders})
            """,
            cluster_ids,
        )
        themes = [
            ThemeRef(
                cluster_id=r["cluster_id"],
                community_id=r["community_id"],
                label=r["top_label"],
            )
            for r in cursor.fetchall()
        ]

    return BookDetail(
        id=book["id"],
        title=book["title"],
        author=book["author"],
        calibre_id=book["calibre_id"],
        description=book.get("description"),
        gutenberg_url=gutenberg_url,
        topics=topics,
        themes=themes,
        chunk_count=chunk_count,
    )


@router.get("/books/{book_id}/related", response_model=list[RelatedBook])
def get_related_books(
    db: DBConnection,
    book_id: int,
    limit: int = Query(10, ge=1, le=50),
):
    """Get related books via hybrid search on the source book's metadata."""
    cursor = db.cursor()

    # Verify book exists
    cursor.execute("SELECT id FROM books WHERE id = ?", (book_id,))
    if not cursor.fetchone():
        raise HTTPException(status_code=404, detail="Book not found")

    results = find_related_books(db, book_id, limit=limit)

    return [
        RelatedBook(
            id=r["book_id"],
            title=r["title"],
            author=r["author"],
            calibre_id=r["calibre_id"],
            similarity=r["similarity"],
            match_type=r.get("match_type", ""),
        )
        for r in results
    ]
