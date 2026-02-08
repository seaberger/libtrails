"""Domain (super-cluster) API endpoints.

Uses materialized stats tables (domain_stats, cluster_stats, cluster_books)
for fast responses. Run `libtrails refresh-stats` to populate after clustering.
"""

import json

from fastapi import APIRouter, HTTPException

from ..dependencies import DBConnection
from ..schemas import BookSummary, DomainDetail, DomainSummary

router = APIRouter()


@router.get("/domains", response_model=list[DomainSummary])
def list_domains(db: DBConnection):
    """List all domains with cluster counts and sample books."""
    cursor = db.cursor()

    cursor.execute("""
        SELECT d.id, d.label, d.cluster_count,
               ds.book_count, ds.sample_books_json, ds.top_clusters_json
        FROM domains d
        LEFT JOIN domain_stats ds ON ds.domain_id = d.id
        ORDER BY d.cluster_count DESC
    """)
    rows = cursor.fetchall()

    result = []
    for row in rows:
        book_count = row["book_count"] or 0
        sample_books_raw = json.loads(row["sample_books_json"] or "[]")
        top_clusters = json.loads(row["top_clusters_json"] or "[]")

        sample_books = [BookSummary(**b) for b in sample_books_raw]

        result.append(
            DomainSummary(
                domain_id=row["id"],
                label=row["label"],
                cluster_count=row["cluster_count"],
                book_count=book_count,
                sample_books=sample_books,
                top_clusters=top_clusters,
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

    # Get cluster details from cluster_stats
    cursor.execute(
        """
        SELECT cs.cluster_id, cs.size, cs.top_label as label, cs.book_count
        FROM cluster_domains cd
        JOIN cluster_stats cs ON cs.cluster_id = cd.cluster_id
        WHERE cd.domain_id = ?
        ORDER BY cs.size DESC
    """,
        (domain_id,),
    )
    clusters = [dict(r) for r in cursor.fetchall()]

    # Get all books in domain from cluster_books bridge table
    cursor.execute(
        """
        SELECT DISTINCT b.id, b.title, b.author, b.calibre_id
        FROM cluster_domains cd
        JOIN cluster_books cb ON cb.cluster_id = cd.cluster_id
        JOIN books b ON b.id = cb.book_id
        WHERE cd.domain_id = ?
        ORDER BY b.title
    """,
        (domain_id,),
    )
    books = [BookSummary(**dict(r)) for r in cursor.fetchall()]

    return DomainDetail(
        domain_id=domain_id,
        label=domain["label"],
        cluster_count=domain["cluster_count"],
        clusters=clusters,
        books=books,
    )
