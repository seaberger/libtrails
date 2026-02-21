"""Community (mid-tier grouping) API endpoints.

Uses materialized stats tables (community_stats, book_communities, cluster_communities)
for fast responses. Run `libtrails refresh-stats` to populate after clustering.
"""

import json

from fastapi import APIRouter, HTTPException, Query

from ..dependencies import DBConnection
from ..schemas import BookSummary, ClusterInfo, CommunityBook, CommunityDetail, CommunitySummary

router = APIRouter()


@router.get("/communities", response_model=list[CommunitySummary])
def list_communities(
    db: DBConnection,
    domain_id: int | None = Query(None, description="Filter by parent domain"),
):
    """List all communities with stats and sample books."""
    cursor = db.cursor()

    if domain_id is not None:
        cursor.execute(
            """
            SELECT cs.community_id, c.label, cs.topic_count,
                   cs.book_count, cs.primary_book_count,
                   cs.sample_books_json, cs.domain_id, cs.domain_label
            FROM community_stats cs
            JOIN communities c ON c.id = cs.community_id
            WHERE cs.domain_id = ?
            ORDER BY cs.book_count DESC
        """,
            (domain_id,),
        )
    else:
        cursor.execute("""
            SELECT cs.community_id, c.label, cs.topic_count,
                   cs.book_count, cs.primary_book_count,
                   cs.sample_books_json, cs.domain_id, cs.domain_label
            FROM community_stats cs
            JOIN communities c ON c.id = cs.community_id
            ORDER BY cs.book_count DESC
        """)

    rows = cursor.fetchall()

    # Pre-fetch cluster counts per community
    cursor.execute("""
        SELECT community_id, COUNT(*) as cluster_count
        FROM cluster_communities
        GROUP BY community_id
    """)
    cluster_counts = {r["community_id"]: r["cluster_count"] for r in cursor.fetchall()}

    result = []
    for row in rows:
        sample_books_raw = json.loads(row["sample_books_json"] or "[]")
        sample_books = [BookSummary(**b) for b in sample_books_raw]

        result.append(
            CommunitySummary(
                community_id=row["community_id"],
                label=row["label"],
                topic_count=row["topic_count"],
                cluster_count=cluster_counts.get(row["community_id"], 0),
                book_count=row["book_count"] or 0,
                primary_book_count=row["primary_book_count"] or 0,
                domain_id=row["domain_id"],
                domain_label=row["domain_label"] or "",
                sample_books=sample_books,
            )
        )

    return result


@router.get("/communities/{community_id}", response_model=CommunityDetail)
def get_community(db: DBConnection, community_id: int):
    """Get community detail with clusters and books."""
    cursor = db.cursor()

    # Get community
    cursor.execute("SELECT * FROM communities WHERE id = ?", (community_id,))
    community = cursor.fetchone()
    if not community:
        raise HTTPException(status_code=404, detail="Community not found")

    # Get domain label
    domain_label = ""
    if community["domain_id"] is not None:
        cursor.execute("SELECT label FROM domains WHERE id = ?", (community["domain_id"],))
        domain_row = cursor.fetchone()
        if domain_row:
            domain_label = domain_row["label"]

    # Get constituent clusters from cluster_communities
    cursor.execute(
        """
        SELECT cc.cluster_id,
               COALESCE(cs.size, 0) as size,
               COALESCE(cs.top_label, 'cluster_' || cc.cluster_id) as label,
               COALESCE(cs.book_count, 0) as book_count
        FROM cluster_communities cc
        LEFT JOIN cluster_stats cs ON cs.cluster_id = cc.cluster_id
        WHERE cc.community_id = ?
        ORDER BY size DESC
    """,
        (community_id,),
    )
    clusters = [
        ClusterInfo(
            cluster_id=r["cluster_id"],
            label=r["label"],
            size=r["size"],
            book_count=r["book_count"],
        )
        for r in cursor.fetchall()
    ]

    # Get books from book_communities with concentration threshold
    cursor.execute(
        """
        SELECT b.id, b.title, b.author, b.calibre_id,
               bc.concentration, bc.is_primary
        FROM book_communities bc
        JOIN books b ON b.id = bc.book_id
        WHERE bc.community_id = ? AND bc.concentration >= 0.01
        ORDER BY bc.relevance_score DESC, b.title
    """,
        (community_id,),
    )
    books = [
        CommunityBook(
            id=r["id"],
            title=r["title"],
            author=r["author"],
            calibre_id=r["calibre_id"],
            concentration=round(r["concentration"], 4),
            is_primary=bool(r["is_primary"]),
        )
        for r in cursor.fetchall()
    ]

    return CommunityDetail(
        community_id=community_id,
        label=community["label"],
        topic_count=community["topic_count"],
        domain_id=community["domain_id"],
        domain_label=domain_label,
        clusters=clusters,
        books=books,
    )
