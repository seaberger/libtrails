"""Domain (super-cluster) API endpoints."""

from fastapi import APIRouter, HTTPException

from ..dependencies import DBConnection
from ..schemas import BookSummary, DomainDetail, DomainSummary

router = APIRouter()


@router.get("/domains", response_model=list[DomainSummary])
def list_domains(db: DBConnection):
    """List all domains with cluster counts and sample books."""
    cursor = db.cursor()

    cursor.execute("""
        SELECT id, label, cluster_count FROM domains ORDER BY cluster_count DESC
    """)
    domains = cursor.fetchall()

    result = []
    for d in domains:
        domain_id = d["id"]

        # Get cluster IDs for this domain
        cursor.execute(
            "SELECT cluster_id FROM cluster_domains WHERE domain_id = ?",
            (domain_id,)
        )
        cluster_ids = [r["cluster_id"] for r in cursor.fetchall()]

        if not cluster_ids:
            continue

        # Get book count across all clusters in domain
        placeholders = ",".join("?" * len(cluster_ids))
        cursor.execute(f"""
            SELECT COUNT(DISTINCT b.id) as book_count
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE t.cluster_id IN ({placeholders})
        """, cluster_ids)
        book_count = cursor.fetchone()["book_count"]

        # Get sample books (top 5 by topic coverage)
        cursor.execute(f"""
            SELECT b.id, b.title, b.author, b.calibre_id,
                   COUNT(DISTINCT t.id) as topic_count
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE t.cluster_id IN ({placeholders}) AND b.calibre_id IS NOT NULL
            GROUP BY b.id
            ORDER BY topic_count DESC
            LIMIT 5
        """, cluster_ids)
        sample_books = [BookSummary(**dict(r)) for r in cursor.fetchall()]

        # Get top clusters in domain (by size)
        cursor.execute(f"""
            SELECT t.cluster_id, COUNT(*) as size
            FROM topics t
            WHERE t.cluster_id IN ({placeholders})
            GROUP BY t.cluster_id
            ORDER BY size DESC
            LIMIT 5
        """, cluster_ids)
        top_clusters = []
        for r in cursor.fetchall():
            cid = r["cluster_id"]
            # Get cluster label
            cursor.execute("""
                SELECT label FROM topics
                WHERE cluster_id = ? AND LENGTH(label) >= 4
                ORDER BY occurrence_count DESC
                LIMIT 1
            """, (cid,))
            label_row = cursor.fetchone()
            top_clusters.append({
                "cluster_id": cid,
                "label": label_row["label"] if label_row else f"cluster_{cid}",
                "size": r["size"]
            })

        result.append(DomainSummary(
            domain_id=domain_id,
            label=d["label"],
            cluster_count=d["cluster_count"],
            book_count=book_count,
            sample_books=sample_books,
            top_clusters=top_clusters,
        ))

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

    # Get all cluster IDs
    cursor.execute(
        "SELECT cluster_id FROM cluster_domains WHERE domain_id = ?",
        (domain_id,)
    )
    cluster_ids = [r["cluster_id"] for r in cursor.fetchall()]

    # Get cluster details
    clusters = []
    for cid in cluster_ids:
        cursor.execute("""
            SELECT COUNT(*) as size FROM topics WHERE cluster_id = ?
        """, (cid,))
        size = cursor.fetchone()["size"]

        cursor.execute("""
            SELECT label FROM topics
            WHERE cluster_id = ? AND LENGTH(label) >= 4
            ORDER BY occurrence_count DESC
            LIMIT 1
        """, (cid,))
        label_row = cursor.fetchone()

        cursor.execute("""
            SELECT COUNT(DISTINCT b.id) as book_count
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE t.cluster_id = ?
        """, (cid,))
        book_count = cursor.fetchone()["book_count"]

        clusters.append({
            "cluster_id": cid,
            "label": label_row["label"] if label_row else f"cluster_{cid}",
            "size": size,
            "book_count": book_count,
        })

    # Sort by size descending
    clusters.sort(key=lambda x: x["size"], reverse=True)

    # Get all books in domain
    if cluster_ids:
        placeholders = ",".join("?" * len(cluster_ids))
        cursor.execute(f"""
            SELECT DISTINCT b.id, b.title, b.author, b.calibre_id
            FROM books b
            JOIN chunks c ON c.book_id = b.id
            JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
            JOIN topics t ON t.id = ctl.topic_id
            WHERE t.cluster_id IN ({placeholders})
            ORDER BY b.title
        """, cluster_ids)
        books = [BookSummary(**dict(r)) for r in cursor.fetchall()]
    else:
        books = []

    return DomainDetail(
        domain_id=domain_id,
        label=domain["label"],
        cluster_count=domain["cluster_count"],
        clusters=clusters,
        books=books,
    )
