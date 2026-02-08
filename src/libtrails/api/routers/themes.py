"""Theme/cluster API endpoints.

Uses materialized stats tables (cluster_stats, cluster_books) for fast
responses. Run `libtrails refresh-stats` to populate after clustering.
"""

import json

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import DBConnection
from ..schemas import BookSummary, ThemeDetail, ThemeSummary, TopicInfo

router = APIRouter()


@router.get("/themes/search", response_model=list[ThemeSummary])
def search_themes(
    db: DBConnection,
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = 20,
):
    """Search for themes using semantic similarity.

    Finds topics matching the query, then returns their clusters
    ranked by how many matching topics each cluster contains.
    """
    from libtrails.vector_search import search_topics_semantic

    cursor = db.cursor()

    # Find matching topics via semantic search
    matching_topics = search_topics_semantic(q, limit=100)

    if not matching_topics:
        return []

    # Group by cluster and count matches
    cluster_matches = {}
    for topic in matching_topics:
        cluster_id = topic.get("cluster_id")
        if cluster_id is not None and cluster_id >= 0:
            if cluster_id not in cluster_matches:
                cluster_matches[cluster_id] = {
                    "count": 0,
                    "best_score": 0,
                    "best_topic": None,
                }
            cluster_matches[cluster_id]["count"] += 1
            distance = topic.get("distance") or 0
            score = 1 - float(distance)
            if score > cluster_matches[cluster_id]["best_score"]:
                cluster_matches[cluster_id]["best_score"] = score
                cluster_matches[cluster_id]["best_topic"] = topic.get("label")

    # Sort clusters by match count, then by best score
    sorted_clusters = sorted(
        cluster_matches.items(), key=lambda x: (x[1]["count"], x[1]["best_score"]), reverse=True
    )[:limit]

    if not sorted_clusters:
        return []

    # Batch-fetch cluster_stats for all matching cluster IDs
    cluster_ids = [cid for cid, _ in sorted_clusters]
    placeholders = ",".join("?" * len(cluster_ids))
    cursor.execute(
        f"""
        SELECT cluster_id, size, book_count, top_label, top_topics_json, sample_books_json
        FROM cluster_stats
        WHERE cluster_id IN ({placeholders})
    """,
        cluster_ids,
    )
    stats_by_id = {r["cluster_id"]: r for r in cursor.fetchall()}

    # Collect cluster IDs missing from cluster_stats for fallback
    missing_ids = [cid for cid, _ in sorted_clusters if cid not in stats_by_id]
    if missing_ids:
        missing_ph = ",".join("?" * len(missing_ids))
        cursor.execute(
            f"""
            SELECT cluster_id, COUNT(*) as size,
                   (SELECT label FROM topics t2
                    WHERE t2.cluster_id = t.cluster_id AND LENGTH(t2.label) >= 4
                    ORDER BY t2.occurrence_count DESC LIMIT 1) as top_label
            FROM topics t
            WHERE cluster_id IN ({missing_ph})
            GROUP BY cluster_id
        """,
            missing_ids,
        )
        for row in cursor.fetchall():
            cid = row["cluster_id"]
            stats_by_id[cid] = {
                "cluster_id": cid,
                "size": row["size"],
                "book_count": 0,
                "top_label": row["top_label"] or f"cluster_{cid}",
                "top_topics_json": "[]",
                "sample_books_json": "[]",
            }

    themes = []
    for cluster_id, match_info in sorted_clusters:
        cs = stats_by_id.get(cluster_id)
        if not cs:
            continue

        top_topics = json.loads(cs["top_topics_json"] or "[]")
        sample_books = [BookSummary(**b) for b in json.loads(cs["sample_books_json"] or "[]")]

        themes.append(
            ThemeSummary(
                cluster_id=cluster_id,
                label=_generate_cluster_label(top_topics),
                size=cs["size"],
                book_count=cs["book_count"],
                sample_books=sample_books,
            )
        )

    return themes


def _generate_cluster_label(topics: list[dict], max_labels: int = 1) -> str:
    """Generate a label from top topics in cluster."""
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

    Clusters with >max_topics are filtered out as they're too broad to be useful.
    """
    cursor = db.cursor()

    cursor.execute(
        """
        SELECT cluster_id, size, book_count, top_label, top_topics_json, sample_books_json
        FROM cluster_stats
        WHERE size <= ? AND book_count >= ?
        ORDER BY size DESC
        LIMIT ?
    """,
        (max_topics, min_books, limit),
    )
    rows = cursor.fetchall()

    themes = []
    for row in rows:
        top_topics = json.loads(row["top_topics_json"] or "[]")
        sample_books = [BookSummary(**b) for b in json.loads(row["sample_books_json"] or "[]")]

        themes.append(
            ThemeSummary(
                cluster_id=row["cluster_id"],
                label=_generate_cluster_label(top_topics),
                size=row["size"],
                book_count=row["book_count"],
                sample_books=sample_books,
            )
        )

    return themes


@router.get("/themes/{cluster_id}", response_model=ThemeDetail)
def get_theme(db: DBConnection, cluster_id: int):
    """Get theme detail with all books."""
    cursor = db.cursor()

    # Verify cluster exists via cluster_stats
    cursor.execute(
        "SELECT size FROM cluster_stats WHERE cluster_id = ?",
        (cluster_id,),
    )
    stats_row = cursor.fetchone()
    if not stats_row:
        # Fall back to checking topics directly (stats may not be populated)
        cursor.execute(
            "SELECT COUNT(*) as size FROM topics WHERE cluster_id = ?",
            (cluster_id,),
        )
        result = cursor.fetchone()
        if result["size"] == 0:
            raise HTTPException(status_code=404, detail="Theme not found")
        size = result["size"]
    else:
        size = stats_row["size"]

    # Get all topics in cluster (already fast — single table scan)
    cursor.execute(
        """
        SELECT id, label, occurrence_count as count, cluster_id
        FROM topics
        WHERE cluster_id = ?
        ORDER BY occurrence_count DESC
        LIMIT 50
    """,
        (cluster_id,),
    )
    topics = [TopicInfo(**dict(r)) for r in cursor.fetchall()]

    # Get all books from cluster_books bridge table
    cursor.execute(
        """
        SELECT DISTINCT b.id, b.title, b.author, b.calibre_id
        FROM cluster_books cb
        JOIN books b ON b.id = cb.book_id
        WHERE cb.cluster_id = ?
        ORDER BY b.title
    """,
        (cluster_id,),
    )
    books = [BookSummary(**dict(r)) for r in cursor.fetchall()]

    return ThemeDetail(
        cluster_id=cluster_id,
        label=_generate_cluster_label([{"label": t.label} for t in topics]),
        size=size,
        topics=topics,
        books=books,
    )
