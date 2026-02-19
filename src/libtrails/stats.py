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
        total_corpus_row = cursor.execute("SELECT SUM(topic_count) FROM cluster_books").fetchone()
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
                {
                    "id": r["id"],
                    "title": r["title"],
                    "author": r["author"],
                    "calibre_id": r["calibre_id"],
                }
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


def refresh_book_domains(conn: sqlite3.Connection) -> int:
    """Rebuild book_domains from cluster_books + cluster_domains.

    For each (book, domain) pair, aggregates topic counts across all clusters
    in the domain, computes concentration and relevance scores, and marks
    each book's highest-scoring domain as primary.

    Requires cluster_books to be populated first.
    Returns the number of rows inserted.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM book_domains")

        # Total corpus for PPMI
        total_corpus_row = cursor.execute("SELECT SUM(topic_count) FROM cluster_books").fetchone()
        total_corpus = total_corpus_row[0] if total_corpus_row[0] else 1

        # Aggregate cluster_books by domain: (book_id, domain_id) -> topic_count
        cursor.execute("""
            INSERT INTO book_domains (book_id, domain_id, topic_count, book_total_topics,
                                      concentration, relevance_score, is_primary)
            SELECT
                cb.book_id,
                cd.domain_id,
                SUM(cb.topic_count) as topic_count,
                MAX(cb.book_total_topics) as book_total_topics,
                CAST(SUM(cb.topic_count) AS REAL) / MAX(cb.book_total_topics) as concentration,
                0.0,  -- placeholder, scored below
                0
            FROM cluster_books cb
            JOIN cluster_domains cd ON cd.cluster_id = cb.cluster_id
            WHERE cb.book_total_topics > 0
            GROUP BY cb.book_id, cd.domain_id
        """)
        count = cursor.rowcount

        # Compute relevance scores using book_cluster_relevance()
        # We need domain-level total topics for the PPMI component
        cursor.execute("""
            SELECT bd.book_id, bd.domain_id, bd.topic_count, bd.book_total_topics
            FROM book_domains bd
        """)
        rows = cursor.fetchall()

        # Cache domain total topics (sum of all topic_count in that domain's clusters)
        domain_totals: dict[int, int] = {}
        cursor.execute("""
            SELECT cd.domain_id, SUM(cb.topic_count) as total
            FROM cluster_books cb
            JOIN cluster_domains cd ON cd.cluster_id = cb.cluster_id
            GROUP BY cd.domain_id
        """)
        for r in cursor.fetchall():
            domain_totals[r["domain_id"]] = r["total"]

        # Batch update relevance scores
        updates = []
        for r in rows:
            domain_total = domain_totals.get(r["domain_id"], 1)
            score = book_cluster_relevance(
                topics_in_cluster=r["topic_count"],
                total_topics_book=r["book_total_topics"],
                total_topics_cluster=domain_total,
                total_corpus=total_corpus,
                min_topics=1,  # lower threshold for domains (aggregated)
            )
            updates.append((score, r["book_id"], r["domain_id"]))

        cursor.executemany(
            "UPDATE book_domains SET relevance_score = ? WHERE book_id = ? AND domain_id = ?",
            updates,
        )

        # Mark is_primary: each book's single highest-scoring domain
        cursor.execute("""
            UPDATE book_domains SET is_primary = 1
            WHERE rowid IN (
                SELECT rowid FROM (
                    SELECT rowid,
                           ROW_NUMBER() OVER (
                               PARTITION BY book_id
                               ORDER BY relevance_score DESC, rowid ASC
                           ) as rn
                    FROM book_domains
                    WHERE relevance_score > 0
                )
                WHERE rn = 1
            )
        """)

        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise


def refresh_domain_stats(conn: sqlite3.Connection) -> int:
    """Rebuild domain_stats from book_domains + cluster_stats.

    Computes per-domain: book_count (≥1% concentration), primary_book_count,
    sample_books_json (top 5 by relevance), top_clusters_json.
    Requires book_domains and cluster_stats to be populated first.

    Returns the number of domains with stats.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM domain_stats")

        cursor.execute("SELECT id FROM domains")
        domains = cursor.fetchall()

        for domain_row in domains:
            domain_id = domain_row["id"]

            # Book count: books with ≥1% concentration in this domain
            cursor.execute(
                """
                SELECT COUNT(*) as cnt FROM book_domains
                WHERE domain_id = ? AND concentration >= 0.01
            """,
                (domain_id,),
            )
            book_count = cursor.fetchone()["cnt"]

            # Primary book count: books where this is their top domain
            cursor.execute(
                """
                SELECT COUNT(*) as cnt FROM book_domains
                WHERE domain_id = ? AND is_primary = 1
            """,
                (domain_id,),
            )
            primary_book_count = cursor.fetchone()["cnt"]

            # Sample books: top 5 by relevance_score from book_domains
            cursor.execute(
                """
                SELECT b.id, b.title, b.author, b.calibre_id
                FROM book_domains bd
                JOIN books b ON b.id = bd.book_id
                WHERE bd.domain_id = ? AND b.calibre_id IS NOT NULL
                ORDER BY bd.relevance_score DESC
                LIMIT 5
            """,
                (domain_id,),
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
                """
                SELECT cs.cluster_id, cs.size, cs.top_label as label
                FROM cluster_stats cs
                JOIN cluster_domains cd ON cd.cluster_id = cs.cluster_id
                WHERE cd.domain_id = ?
                ORDER BY cs.size DESC
                LIMIT 5
            """,
                (domain_id,),
            )
            top_clusters = [
                {"cluster_id": r["cluster_id"], "label": r["label"], "size": r["size"]}
                for r in cursor.fetchall()
            ]

            # Community count and top communities for this domain
            cursor.execute(
                "SELECT COUNT(*) as cnt FROM communities WHERE domain_id = ?",
                (domain_id,),
            )
            community_count = cursor.fetchone()["cnt"]

            cursor.execute(
                """
                SELECT c.id as community_id, cs.top_label as label,
                       cs.topic_count, cs.book_count
                FROM communities c
                JOIN community_stats cs ON cs.community_id = c.id
                WHERE c.domain_id = ?
                ORDER BY cs.topic_count DESC
                LIMIT 8
            """,
                (domain_id,),
            )
            top_communities = [
                {
                    "community_id": r["community_id"],
                    "label": r["label"],
                    "topic_count": r["topic_count"],
                    "book_count": r["book_count"],
                }
                for r in cursor.fetchall()
            ]

            cursor.execute(
                """
                INSERT INTO domain_stats
                    (domain_id, book_count, primary_book_count, sample_books_json,
                     top_clusters_json, community_count, top_communities_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    domain_id,
                    book_count,
                    primary_book_count,
                    json.dumps(sample_books),
                    json.dumps(top_clusters),
                    community_count,
                    json.dumps(top_communities),
                ),
            )

        conn.commit()
        return len(domains)
    except Exception:
        conn.rollback()
        raise


def refresh_book_communities(conn: sqlite3.Connection) -> int:
    """Rebuild book_communities from cluster_books + cluster_communities.

    For each (book, community) pair, aggregates topic counts across all clusters
    in the community, computes concentration and relevance scores, and marks
    each book's highest-scoring community as primary.

    Requires cluster_books and cluster_communities to be populated first.
    Returns the number of rows inserted.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM book_communities")

        # Check if cluster_communities has data
        cursor.execute("SELECT COUNT(*) FROM cluster_communities")
        if cursor.fetchone()[0] == 0:
            conn.commit()
            return 0

        # Total corpus for PPMI
        total_corpus_row = cursor.execute("SELECT SUM(topic_count) FROM cluster_books").fetchone()
        total_corpus = total_corpus_row[0] if total_corpus_row[0] else 1

        # Aggregate cluster_books by community
        cursor.execute("""
            INSERT INTO book_communities (book_id, community_id, topic_count, book_total_topics,
                                          concentration, relevance_score, is_primary)
            SELECT
                cb.book_id,
                cc.community_id,
                SUM(cb.topic_count) as topic_count,
                MAX(cb.book_total_topics) as book_total_topics,
                CAST(SUM(cb.topic_count) AS REAL) / MAX(cb.book_total_topics) as concentration,
                0.0,
                0
            FROM cluster_books cb
            JOIN cluster_communities cc ON cc.cluster_id = cb.cluster_id
            WHERE cb.book_total_topics > 0
            GROUP BY cb.book_id, cc.community_id
        """)
        count = cursor.rowcount

        # Compute relevance scores
        cursor.execute("""
            SELECT bc.book_id, bc.community_id, bc.topic_count, bc.book_total_topics
            FROM book_communities bc
        """)
        rows = cursor.fetchall()

        # Cache community total topics
        community_totals: dict[int, int] = {}
        cursor.execute("""
            SELECT cc.community_id, SUM(cb.topic_count) as total
            FROM cluster_books cb
            JOIN cluster_communities cc ON cc.cluster_id = cb.cluster_id
            GROUP BY cc.community_id
        """)
        for r in cursor.fetchall():
            community_totals[r["community_id"]] = r["total"]

        # Batch update relevance scores
        updates = []
        for r in rows:
            community_total = community_totals.get(r["community_id"], 1)
            score = book_cluster_relevance(
                topics_in_cluster=r["topic_count"],
                total_topics_book=r["book_total_topics"],
                total_topics_cluster=community_total,
                total_corpus=total_corpus,
                min_topics=1,
            )
            updates.append((score, r["book_id"], r["community_id"]))

        cursor.executemany(
            "UPDATE book_communities SET relevance_score = ? WHERE book_id = ? AND community_id = ?",
            updates,
        )

        # Mark is_primary: each book's single highest-scoring community
        cursor.execute("""
            UPDATE book_communities SET is_primary = 1
            WHERE rowid IN (
                SELECT rowid FROM (
                    SELECT rowid,
                           ROW_NUMBER() OVER (
                               PARTITION BY book_id
                               ORDER BY relevance_score DESC, rowid ASC
                           ) as rn
                    FROM book_communities
                    WHERE relevance_score > 0
                )
                WHERE rn = 1
            )
        """)

        conn.commit()
        return count
    except Exception:
        conn.rollback()
        raise


def refresh_community_stats(conn: sqlite3.Connection) -> int:
    """Rebuild community_stats from communities + book_communities.

    Computes per-community: topic_count, book_count (>=1% concentration),
    primary_book_count, sample_books_json, top_topics_json, domain info.
    Requires book_communities to be populated first.

    Returns the number of communities with stats.
    """
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM community_stats")

        cursor.execute("SELECT id, label, domain_id, topic_count FROM communities")
        communities = cursor.fetchall()

        # Cache domain labels
        domain_labels: dict[int, str] = {}
        cursor.execute("SELECT id, label FROM domains")
        for r in cursor.fetchall():
            domain_labels[r["id"]] = r["label"]

        for row in communities:
            community_id = row["id"]
            topic_count = row["topic_count"]
            domain_id = row["domain_id"]
            domain_label = domain_labels.get(domain_id, "") if domain_id is not None else ""

            # Book count: books with >=1% concentration
            cursor.execute(
                """
                SELECT COUNT(*) as cnt FROM book_communities
                WHERE community_id = ? AND concentration >= 0.01
            """,
                (community_id,),
            )
            book_count = cursor.fetchone()["cnt"]

            # Primary book count
            cursor.execute(
                """
                SELECT COUNT(*) as cnt FROM book_communities
                WHERE community_id = ? AND is_primary = 1
            """,
                (community_id,),
            )
            primary_book_count = cursor.fetchone()["cnt"]

            # Top label (highest-occurrence topic in community)
            cursor.execute(
                """
                SELECT label FROM topics
                WHERE community_id = ? AND LENGTH(label) >= 4
                ORDER BY occurrence_count DESC
                LIMIT 1
            """,
                (community_id,),
            )
            label_row = cursor.fetchone()
            top_label = label_row["label"] if label_row else row["label"]

            # Top 5 topics
            cursor.execute(
                """
                SELECT id, label, occurrence_count as count
                FROM topics
                WHERE community_id = ?
                ORDER BY occurrence_count DESC
                LIMIT 5
            """,
                (community_id,),
            )
            top_topics = [dict(r) for r in cursor.fetchall()]

            # Sample books: top 5 by relevance_score
            cursor.execute(
                """
                SELECT b.id, b.title, b.author, b.calibre_id
                FROM book_communities bc
                JOIN books b ON b.id = bc.book_id
                WHERE bc.community_id = ? AND b.calibre_id IS NOT NULL
                ORDER BY bc.relevance_score DESC
                LIMIT 5
            """,
                (community_id,),
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

            cursor.execute(
                """
                INSERT INTO community_stats
                    (community_id, topic_count, book_count, primary_book_count,
                     top_label, top_topics_json, sample_books_json,
                     domain_id, domain_label, refreshed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
                (
                    community_id,
                    topic_count,
                    book_count,
                    primary_book_count,
                    top_label,
                    json.dumps(top_topics),
                    json.dumps(sample_books),
                    domain_id,
                    domain_label,
                ),
            )

        conn.commit()
        return len(communities)
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
    book_community_rows = refresh_book_communities(conn)
    community_count = refresh_community_stats(conn)
    book_domain_rows = refresh_book_domains(conn)
    domain_count = refresh_domain_stats(conn)

    elapsed = time.time() - start

    return {
        "cluster_book_rows": cluster_book_rows,
        "clusters_with_stats": cluster_count,
        "book_community_rows": book_community_rows,
        "communities_with_stats": community_count,
        "book_domain_rows": book_domain_rows,
        "domains_with_stats": domain_count,
        "elapsed_seconds": round(elapsed, 2),
    }
