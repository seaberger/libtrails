"""Theme/cluster API endpoints."""

from fastapi import APIRouter, HTTPException

from ..dependencies import DBConnection
from ..schemas import BookSummary, ThemeDetail, ThemeSummary, TopicInfo

router = APIRouter()


def _generate_cluster_label(topics: list[dict], max_labels: int = 1) -> str:
    """Generate a label from top topics in cluster.

    Args:
        topics: List of topic dicts with 'label' key
        max_labels: Number of labels to include (1 for focused, 3 for descriptive)
    """
    if not topics:
        return "Miscellaneous"
    top_labels = [t["label"] for t in topics[:max_labels]]
    return " / ".join(top_labels)


@router.get("/themes", response_model=list[ThemeSummary])
def list_themes(
    db: DBConnection,
    limit: int = 100,
    min_books: int = 2,
    max_topics: int = 500,
):
    """List theme clusters with sample books.

    Sample books are selected by "concentration" - books with the highest
    percentage of their topics in this cluster are shown first.

    Clusters with >max_topics are filtered out as they're too broad to be useful.
    """
    cursor = db.cursor()

    # Get clusters with topic counts, filtering out mega-clusters
    cursor.execute("""
        SELECT
            cluster_id,
            COUNT(*) as size
        FROM topics
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        HAVING COUNT(*) <= ?
        ORDER BY size DESC
        LIMIT ?
    """, (max_topics, limit * 2,))  # Fetch extra to filter by min_books

    clusters = cursor.fetchall()
    themes = []

    for row in clusters:
        cluster_id = row["cluster_id"]

        # Get book count for this cluster
        cursor.execute("""
            SELECT COUNT(DISTINCT b.id) as book_count
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE t.cluster_id = ?
        """, (cluster_id,))
        book_count = cursor.fetchone()["book_count"]

        if book_count < min_books:
            continue

        # Get top topics for label
        cursor.execute("""
            SELECT id, label, occurrence_count as count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
            LIMIT 3
        """, (cluster_id,))
        top_topics = [dict(r) for r in cursor.fetchall()]

        # Get sample books with most topics in this cluster
        # (Simple and fast - books deeply covering this theme appear first)
        cursor.execute("""
            SELECT
                b.id,
                b.title,
                b.author,
                b.calibre_id,
                COUNT(DISTINCT t.id) as topics_in_cluster
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE t.cluster_id = ? AND b.calibre_id IS NOT NULL
            GROUP BY b.id
            ORDER BY topics_in_cluster DESC
            LIMIT 5
        """, (cluster_id,))
        sample_books = [BookSummary(**dict(r)) for r in cursor.fetchall()]

        themes.append(ThemeSummary(
            cluster_id=cluster_id,
            label=_generate_cluster_label(top_topics),
            size=row["size"],
            book_count=book_count,
            sample_books=sample_books,
        ))

        if len(themes) >= limit:
            break

    return themes


@router.get("/themes/{cluster_id}", response_model=ThemeDetail)
def get_theme(db: DBConnection, cluster_id: int):
    """Get theme detail with all books."""
    cursor = db.cursor()

    # Verify cluster exists
    cursor.execute("""
        SELECT COUNT(*) as size FROM topics WHERE cluster_id = ?
    """, (cluster_id,))
    result = cursor.fetchone()
    if result["size"] == 0:
        raise HTTPException(status_code=404, detail="Theme not found")

    # Get all topics in cluster
    cursor.execute("""
        SELECT id, label, occurrence_count as count, cluster_id
        FROM topics
        WHERE cluster_id = ?
        ORDER BY occurrence_count DESC
        LIMIT 50
    """, (cluster_id,))
    topics = [TopicInfo(**dict(r)) for r in cursor.fetchall()]

    # Get all books in this cluster
    cursor.execute("""
        SELECT DISTINCT b.id, b.title, b.author, b.calibre_id
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.cluster_id = ?
        ORDER BY b.title
    """, (cluster_id,))
    books = [BookSummary(**dict(r)) for r in cursor.fetchall()]

    return ThemeDetail(
        cluster_id=cluster_id,
        label=_generate_cluster_label([{"label": t.label} for t in topics]),
        size=result["size"],
        topics=topics,
        books=books,
    )
