"""Materialized statistics for fast API responses.

Pre-computes expensive aggregations (cluster→book mappings, per-cluster stats,
per-domain stats) into denormalized tables. Refreshed after clustering or
domain loading — not on every API request.
"""

import json
import math
import sqlite3
import time

from .database import get_db, init_chunks_table


def book_cluster_relevance(
    topics_in_cluster: int,
    total_topics_book: int,
    total_topics_cluster: int,
    total_corpus: int,
    k1: float = 1.5,
    min_topics: int = 3,
) -> float:
    """Score a book's relevance to a cluster using concentration + BM25 + PPMI.

    Eliminates length bias by normalizing for book size, applies BM25 saturation
    to prevent tiny books from dominating, and uses PPMI to reward above-chance
    associations.
    """
    if topics_in_cluster < min_topics:
        return 0.0

    # Concentration: what fraction of this book's topics are in this cluster
    concentration = topics_in_cluster / total_topics_book

    # BM25-style saturation: diminishing returns on concentration
    saturated = concentration * (k1 + 1) / (concentration + k1)

    # PPMI: is this association above random chance?
    expected = (total_topics_book * total_topics_cluster) / total_corpus
    if expected > 0 and topics_in_cluster > 0:
        pmi = math.log2(topics_in_cluster / expected)
        ppmi = max(pmi, 0)
    else:
        ppmi = 0

    return saturated * (1 + ppmi)


def refresh_cluster_books(conn: sqlite3.Connection) -> int:
    """Rebuild the cluster_books bridge table from the canonical 4-table join.

    This runs the expensive books→chunks→chunk_topic_links→topics join once
    so that API endpoints can look up books-per-cluster via a single table.

    Returns the number of rows inserted.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cluster_books")

        # Insert cluster-book pairs with topic counts
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

        # Populate book_total_topics: each book's total distinct topics across all clusters
        cursor.execute("""
            UPDATE cluster_books
            SET book_total_topics = (
                SELECT SUM(cb2.topic_count)
                FROM cluster_books cb2
                WHERE cb2.book_id = cluster_books.book_id
            )
        """)

        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise


def refresh_cluster_stats(conn: sqlite3.Connection) -> int:
    """Rebuild cluster_stats from topics + cluster_books.

    Computes per-cluster: size, book_count, top_label, top_topics_json,
    sample_books_json. Requires cluster_books to be populated first.

    Returns the number of clusters with stats.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cluster_stats")

        # Total corpus size for PPMI calculation (sum of all topic counts)
        total_corpus_row = cursor.execute(
            "SELECT SUM(topic_count) FROM cluster_books"
        ).fetchone()
        total_corpus = total_corpus_row[0] if total_corpus_row[0] else 1

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

            # Sample books: top 5 by relevance score (not raw topic_count)
            cursor.execute(
                """
                SELECT b.id, b.title, b.author, b.calibre_id,
                       cb.topic_count, cb.book_total_topics
                FROM cluster_books cb
                JOIN books b ON b.id = cb.book_id
                WHERE cb.cluster_id = ? AND b.calibre_id IS NOT NULL
            """,
                (cluster_id,),
            )
            book_rows = cursor.fetchall()

            scored = []
            for r in book_rows:
                score = book_cluster_relevance(
                    topics_in_cluster=r["topic_count"],
                    total_topics_book=r["book_total_topics"],
                    total_topics_cluster=size,
                    total_corpus=total_corpus,
                )
                scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            sample_books = [
                {"id": r["id"], "title": r["title"], "author": r["author"],
                 "calibre_id": r["calibre_id"]}
                for _, r in scored[:5]
            ]

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
    except Exception:
        conn.rollback()
        raise


def refresh_domain_stats(conn: sqlite3.Connection) -> int:
    """Rebuild domain_stats from cluster_stats + cluster_books.

    Computes per-domain: book_count, sample_books_json, top_clusters_json.
    Requires cluster_stats to be populated first.

    Returns the number of domains with stats.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM domain_stats")

        # Total corpus for PPMI (computed once)
        total_corpus_row = cursor.execute(
            "SELECT SUM(topic_count) FROM cluster_books"
        ).fetchone()
        total_corpus = total_corpus_row[0] if total_corpus_row[0] else 1

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

            # Sample books: top 5 by relevance score across domain clusters
            cursor.execute(
                f"""
                SELECT b.id, b.title, b.author, b.calibre_id,
                       SUM(cb.topic_count) as domain_topics,
                       cb.book_total_topics
                FROM cluster_books cb
                JOIN books b ON b.id = cb.book_id
                WHERE cb.cluster_id IN ({placeholders}) AND b.calibre_id IS NOT NULL
                GROUP BY b.id
            """,
                cluster_ids,
            )
            domain_book_rows = cursor.fetchall()

            # Total topics across all clusters in this domain
            cursor.execute(
                f"""
                SELECT SUM(topic_count) FROM cluster_books
                WHERE cluster_id IN ({placeholders})
            """,
                cluster_ids,
            )
            domain_total = cursor.fetchone()[0] or 1

            scored = []
            for r in domain_book_rows:
                score = book_cluster_relevance(
                    topics_in_cluster=r["domain_topics"],
                    total_topics_book=r["book_total_topics"],
                    total_topics_cluster=domain_total,
                    total_corpus=total_corpus,
                )
                scored.append((score, r))
            scored.sort(key=lambda x: x[0], reverse=True)
            sample_books = [
                {"id": r["id"], "title": r["title"], "author": r["author"],
                 "calibre_id": r["calibre_id"]}
                for _, r in scored[:5]
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
    except Exception:
        conn.rollback()
        raise


def refresh_all_stats(conn: sqlite3.Connection | None = None) -> dict:
    """Refresh all materialized stats tables in order.

    If no connection is provided, opens one to the default database.

    Returns a summary dict with counts and timing.
    """
    init_chunks_table()

    if conn is None:
        with get_db() as db_conn:
            return _refresh_all_stats_impl(db_conn)
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
