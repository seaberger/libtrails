"""Domain (super-cluster) API endpoints.

Uses materialized stats tables (domain_stats, cluster_stats, cluster_books)
for fast responses. Run `libtrails refresh-stats` to populate after clustering.
"""

import json

from fastapi import APIRouter, HTTPException

from ..dependencies import DBConnection
from ..schemas import BookSummary, CommunityRef, DomainBook, DomainDetail, DomainSummary

router = APIRouter()


@router.get("/domains", response_model=list[DomainSummary])
def list_domains(db: DBConnection):
    """List all domains with cluster counts and sample books."""
    cursor = db.cursor()

    cursor.execute("""
        SELECT d.id, d.label, d.cluster_count,
               ds.book_count, ds.primary_book_count,
               ds.sample_books_json, ds.top_clusters_json,
               ds.community_count, ds.top_communities_json
        FROM domains d
        LEFT JOIN domain_stats ds ON ds.domain_id = d.id
        ORDER BY d.cluster_count DESC
    """)
    rows = cursor.fetchall()

    result = []
    for row in rows:
        book_count = row["book_count"] or 0
        primary_book_count = row["primary_book_count"] or 0
        community_count = row["community_count"] or 0
        sample_books_raw = json.loads(row["sample_books_json"] or "[]")
        top_clusters = json.loads(row["top_clusters_json"] or "[]")
        top_communities_raw = json.loads(row["top_communities_json"] or "[]")

        sample_books = [BookSummary(**b) for b in sample_books_raw]
        top_communities = [CommunityRef(**c) for c in top_communities_raw]

        result.append(
            DomainSummary(
                domain_id=row["id"],
                label=row["label"],
                cluster_count=row["cluster_count"],
                book_count=book_count,
                primary_book_count=primary_book_count,
                community_count=community_count,
                sample_books=sample_books,
                top_clusters=top_clusters,
                top_communities=top_communities,
            )
        )

    return result


@router.get("/domains/{domain_id}", response_model=DomainDetail)
def get_domain(db: DBConnection, domain_id: int):
    """Get domain detail with all clusters."""
    cursor = db.cursor()

    # Get domain
    cursor.execute("SELECT * FROM domains WHERE id = ?", (domain_id,))
    domain = cursor.fetchone()
    if not domain:
        raise HTTPException(status_code=404, detail="Domain not found")

    # Get cluster details from cluster_stats (LEFT JOIN for clusters without stats)
    cursor.execute(
        """
        SELECT cd.cluster_id,
               COALESCE(cs.size, 0) as size,
               COALESCE(cs.top_label, 'cluster_' || cd.cluster_id) as label,
               COALESCE(cs.book_count, 0) as book_count
        FROM cluster_domains cd
        LEFT JOIN cluster_stats cs ON cs.cluster_id = cd.cluster_id
        WHERE cd.domain_id = ?
        ORDER BY size DESC
    """,
        (domain_id,),
    )
    clusters = [dict(r) for r in cursor.fetchall()]

    # Get books in domain from book_domains, with concentration threshold
    cursor.execute(
        """
        SELECT b.id, b.title, b.author, b.calibre_id,
               bd.concentration, bd.is_primary
        FROM book_domains bd
        JOIN books b ON b.id = bd.book_id
        WHERE bd.domain_id = ? AND bd.concentration >= 0.01
        ORDER BY bd.relevance_score DESC, b.title
    """,
        (domain_id,),
    )
    books = [
        DomainBook(
            id=r["id"],
            title=r["title"],
            author=r["author"],
            calibre_id=r["calibre_id"],
            concentration=round(r["concentration"], 4),
            is_primary=bool(r["is_primary"]),
        )
        for r in cursor.fetchall()
    ]

    return DomainDetail(
        domain_id=domain_id,
        label=domain["label"],
        cluster_count=domain["cluster_count"],
        clusters=clusters,
        books=books,
    )
