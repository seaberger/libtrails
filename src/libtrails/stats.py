"""Materialized statistics for fast API responses.

Pre-computes expensive aggregations (cluster→book mappings, per-cluster stats,
per-domain stats) into denormalized tables. Refreshed after clustering or
domain loading — not on every API request.
"""

import json
import sqlite3
import time

from .database import get_db, init_chunks_table


def refresh_cluster_books(conn: sqlite3.Connection) -> int:
    """Rebuild the cluster_books bridge table from the canonical 4-table join.

    This runs the expensive books→chunks→chunk_topic_links→topics join once
    so that API endpoints can look up books-per-cluster via a single table.

    Returns the number of rows inserted.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cluster_books")
    cursor.execute("""
        INSERT INTO cluster_books (cluster_id, book_id, topic_count)
        SELECT t.cluster_id, b.id, COUNT(DISTINCT t.id)
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.cluster_id IS NOT NULL
        GROUP BY t.cluster_id, b.id
    """)
    count = cursor.rowcount
    conn.commit()
    return count


def refresh_cluster_stats(conn: sqlite3.Connection) -> int:
    """Rebuild cluster_stats from topics + cluster_books.

    Computes per-cluster: size, book_count, top_label, top_topics_json,
    sample_books_json. Requires cluster_books to be populated first.

    Returns the number of clusters with stats.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cluster_stats")

    # Get all clusters with their sizes
    cursor.execute("""
        SELECT cluster_id, COUNT(*) as size
        FROM topics
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
    """)
    clusters = cursor.fetchall()

    for row in clusters:
        cluster_id = row["cluster_id"]
        size = row["size"]

        # Book count from bridge table
        cursor.execute(
            "SELECT COUNT(*) as cnt FROM cluster_books WHERE cluster_id = ?",
            (cluster_id,),
        )
        book_count = cursor.fetchone()["cnt"]

        # Top label (highest-occurrence topic with length >= 4)
        cursor.execute(
            """
            SELECT label FROM topics
            WHERE cluster_id = ? AND LENGTH(label) >= 4
            ORDER BY occurrence_count DESC
            LIMIT 1
        """,
            (cluster_id,),
        )
        label_row = cursor.fetchone()
        top_label = label_row["label"] if label_row else f"cluster_{cluster_id}"

        # Top 3 topics
        cursor.execute(
            """
            SELECT id, label, occurrence_count as count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
            LIMIT 3
        """,
            (cluster_id,),
        )
        top_topics = [dict(r) for r in cursor.fetchall()]

        # Sample books: top 5 by topic_count in this cluster (with calibre_id)
        cursor.execute(
            """
            SELECT b.id, b.title, b.author, b.calibre_id
            FROM cluster_books cb
            JOIN books b ON b.id = cb.book_id
            WHERE cb.cluster_id = ? AND b.calibre_id IS NOT NULL
            ORDER BY cb.topic_count DESC
            LIMIT 5
        """,
            (cluster_id,),
        )
        sample_books = [dict(r) for r in cursor.fetchall()]

        cursor.execute(
            """
            INSERT INTO cluster_stats
                (cluster_id, size, book_count, top_label, top_topics_json, sample_books_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                cluster_id,
                size,
                book_count,
                top_label,
                json.dumps(top_topics),
                json.dumps(sample_books),
            ),
        )

    conn.commit()
    return len(clusters)


def refresh_domain_stats(conn: sqlite3.Connection) -> int:
    """Rebuild domain_stats from cluster_stats + cluster_books.

    Computes per-domain: book_count, sample_books_json, top_clusters_json.
    Requires cluster_stats to be populated first.

    Returns the number of domains with stats.
    """
    cursor = conn.cursor()
    cursor.execute("DELETE FROM domain_stats")

    cursor.execute("SELECT id FROM domains")
    domains = cursor.fetchall()

    for domain_row in domains:
        domain_id = domain_row["id"]

        # Get cluster IDs in this domain
        cursor.execute(
            "SELECT cluster_id FROM cluster_domains WHERE domain_id = ?",
            (domain_id,),
        )
        cluster_ids = [r["cluster_id"] for r in cursor.fetchall()]

        if not cluster_ids:
            cursor.execute(
                """
                INSERT INTO domain_stats (domain_id, book_count, sample_books_json, top_clusters_json)
                VALUES (?, 0, '[]', '[]')
            """,
                (domain_id,),
            )
            continue

        placeholders = ",".join("?" * len(cluster_ids))

        # Book count: distinct books across all clusters in domain
        cursor.execute(
            f"""
            SELECT COUNT(DISTINCT book_id) as cnt
            FROM cluster_books
            WHERE cluster_id IN ({placeholders})
        """,
            cluster_ids,
        )
        book_count = cursor.fetchone()["cnt"]

        # Sample books: top 5 by total topic coverage across domain clusters
        cursor.execute(
            f"""
            SELECT b.id, b.title, b.author, b.calibre_id,
                   SUM(cb.topic_count) as total_topics
            FROM cluster_books cb
            JOIN books b ON b.id = cb.book_id
            WHERE cb.cluster_id IN ({placeholders}) AND b.calibre_id IS NOT NULL
            GROUP BY b.id
            ORDER BY total_topics DESC
            LIMIT 5
        """,
            cluster_ids,
        )
        sample_books = [
            {
                "id": r["id"],
                "title": r["title"],
                "author": r["author"],
                "calibre_id": r["calibre_id"],
            }
            for r in cursor.fetchall()
        ]

        # Top 5 clusters by size
        cursor.execute(
            f"""
            SELECT cluster_id, size, top_label as label
            FROM cluster_stats
            WHERE cluster_id IN ({placeholders})
            ORDER BY size DESC
            LIMIT 5
        """,
            cluster_ids,
        )
        top_clusters = [
            {"cluster_id": r["cluster_id"], "label": r["label"], "size": r["size"]}
            for r in cursor.fetchall()
        ]

        cursor.execute(
            """
            INSERT INTO domain_stats (domain_id, book_count, sample_books_json, top_clusters_json)
            VALUES (?, ?, ?, ?)
        """,
            (domain_id, book_count, json.dumps(sample_books), json.dumps(top_clusters)),
        )

    conn.commit()
    return len(domains)


def refresh_all_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Refresh all materialized stats tables in order.

    If no connection is provided, opens one to the default database.

    Returns a summary dict with counts and timing.
    """
    init_chunks_table()

    if conn is None:
        with get_db() as conn:
            return _refresh_all_stats_impl(conn)
    else:
        return _refresh_all_stats_impl(conn)


def _refresh_all_stats_impl(conn: sqlite3.Connection) -> dict:
    start = time.time()

    cluster_book_rows = refresh_cluster_books(conn)
    cluster_count = refresh_cluster_stats(conn)
    domain_count = refresh_domain_stats(conn)

    elapsed = time.time() - start

    return {
        "cluster_book_rows": cluster_book_rows,
        "clusters_with_stats": cluster_count,
        "domains_with_stats": domain_count,
        "elapsed_seconds": round(elapsed, 2),
    }
