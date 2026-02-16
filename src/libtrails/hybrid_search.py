"""Hybrid search: FTS5 + sqlite-vec + Reciprocal Rank Fusion."""

import re
import sqlite3

from .embeddings import embed_text, embedding_to_bytes


def rrf_fuse(ranked_lists: list[list[tuple[int, float]]], k: int = 60) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion over multiple ranked lists.

    Each ranked list is [(id, score)] sorted by score descending.
    Returns fused [(id, rrf_score)] sorted by rrf_score descending.
    """
    scores: dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, (item_id, _score) in enumerate(ranked):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def _fts5_safe_query(query: str) -> str:
    """
    Sanitize user input for FTS5 MATCH syntax.

    Strips special characters, splits into tokens, joins with OR.
    """
    # Remove FTS5 operators and special chars
    cleaned = re.sub(r"[^\w\s]", " ", query)
    tokens = cleaned.split()
    if not tokens:
        return '""'
    # Quote each token and join with OR
    return " OR ".join(f'"{t}"' for t in tokens)


def _fts_search_books(
    conn: sqlite3.Connection, query: str, limit: int = 50
) -> list[tuple[int, float]]:
    """FTS5 search on books_fts. Returns [(book_id, bm25_score)]."""
    safe_q = _fts5_safe_query(query)
    try:
        rows = conn.execute(
            """
            SELECT rowid, bm25(books_fts) as score
            FROM books_fts
            WHERE books_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_q, limit),
        ).fetchall()
    except Exception:
        return []
    # bm25() returns negative values (lower = better match), negate for ranking
    return [(row[0], -row[1]) for row in rows]


def _fts_search_topics(
    conn: sqlite3.Connection, query: str, limit: int = 100
) -> list[tuple[int, float]]:
    """FTS5 search on topics_fts. Returns [(topic_id, bm25_score)]."""
    safe_q = _fts5_safe_query(query)
    try:
        rows = conn.execute(
            """
            SELECT rowid, bm25(topics_fts) as score
            FROM topics_fts
            WHERE topics_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_q, limit),
        ).fetchall()
    except Exception:
        return []
    return [(row[0], -row[1]) for row in rows]


def _fts_search_chunks(
    conn: sqlite3.Connection, query: str, limit: int = 200
) -> list[tuple[int, float]]:
    """
    FTS5 search on chunks_fts. Returns [(book_id, best_bm25_score)].

    Maps chunk matches back to books, keeping the best score per book.
    """
    safe_q = _fts5_safe_query(query)
    try:
        rows = conn.execute(
            """
            SELECT c.book_id, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_q, limit),
        ).fetchall()
    except Exception:
        return []

    # Keep best score per book
    best: dict[int, float] = {}
    for row in rows:
        book_id, score = row[0], -row[1]
        if book_id not in best or score > best[book_id]:
            best[book_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _semantic_search_topics(
    conn: sqlite3.Connection, query: str, limit: int = 200, query_bytes: bytes | None = None
) -> list[tuple[int, float]]:
    """sqlite-vec cosine search on topic_vectors. Returns [(topic_id, similarity)]."""
    if query_bytes is None:
        query_bytes = embedding_to_bytes(embed_text(query))
    try:
        rows = conn.execute(
            """
            SELECT topic_id, distance
            FROM topic_vectors
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (query_bytes, limit),
        ).fetchall()
    except Exception:
        return []
    # Convert cosine distance to similarity
    return [(row[0], 1.0 - row[1]) for row in rows]


def _semantic_search_book_themes(
    conn: sqlite3.Connection, query: str, limit: int = 50, query_bytes: bytes | None = None
) -> list[tuple[int, float]]:
    """sqlite-vec cosine search on book_theme_vectors. Returns [(book_id, best_similarity)]."""
    if query_bytes is None:
        query_bytes = embedding_to_bytes(embed_text(query))
    try:
        rows = conn.execute(
            """
            SELECT btv.theme_id, btv.distance, bte.book_id
            FROM book_theme_vectors btv
            JOIN book_theme_entries bte ON bte.id = btv.theme_id
            WHERE btv.embedding MATCH ? AND k = ?
            ORDER BY btv.distance
            """,
            (query_bytes, limit),
        ).fetchall()
    except Exception:
        return []

    # Aggregate best similarity per book
    best: dict[int, float] = {}
    for row in rows:
        book_id = row[2]
        similarity = 1.0 - row[1]
        if book_id not in best or similarity > best[book_id]:
            best[book_id] = similarity
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _semantic_search_chunks(
    conn: sqlite3.Connection, query: str, limit: int = 100, query_bytes: bytes | None = None
) -> list[tuple[int, float]]:
    """sqlite-vec cosine search on chunk_vectors. Returns [(book_id, best_similarity)]."""
    if query_bytes is None:
        query_bytes = embedding_to_bytes(embed_text(query))
    try:
        rows = conn.execute(
            """
            SELECT cv.chunk_id, cv.distance, c.book_id
            FROM chunk_vectors cv
            JOIN chunks c ON c.id = cv.chunk_id
            WHERE cv.embedding MATCH ? AND k = ?
            ORDER BY cv.distance
            """,
            (query_bytes, limit),
        ).fetchall()
    except Exception:
        return []

    # Aggregate best similarity per book
    best: dict[int, float] = {}
    for row in rows:
        book_id = row[2]
        similarity = 1.0 - row[1]
        if book_id not in best or similarity > best[book_id]:
            best[book_id] = similarity
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _semantic_search_books_direct(
    conn: sqlite3.Connection, query: str, limit: int = 50, query_bytes: bytes | None = None
) -> list[tuple[int, float]]:
    """sqlite-vec cosine search on book_vectors. Returns [(book_id, similarity)]."""
    if query_bytes is None:
        query_bytes = embedding_to_bytes(embed_text(query))
    try:
        rows = conn.execute(
            """
            SELECT book_id, distance
            FROM book_vectors
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (query_bytes, limit),
        ).fetchall()
    except Exception:
        return []
    return [(row[0], 1.0 - row[1]) for row in rows]


def _topics_to_books(
    conn: sqlite3.Connection, topic_scores: list[tuple[int, float]]
) -> list[tuple[int, float]]:
    """Map topic scores to books. Returns [(book_id, best_score)]."""
    if not topic_scores:
        return []
    score_map = dict(topic_scores)
    topic_ids = list(score_map.keys())
    placeholders = ",".join("?" * len(topic_ids))

    rows = conn.execute(
        f"""
        SELECT DISTINCT c.book_id, ctl.topic_id
        FROM chunk_topic_links ctl
        JOIN chunks c ON c.id = ctl.chunk_id
        WHERE ctl.topic_id IN ({placeholders})
        """,
        topic_ids,
    ).fetchall()

    best: dict[int, float] = {}
    for row in rows:
        book_id, topic_id = row[0], row[1]
        score = score_map.get(topic_id, 0.0)
        if book_id not in best or score > best[book_id]:
            best[book_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _topics_to_clusters(
    conn: sqlite3.Connection, topic_scores: list[tuple[int, float]]
) -> list[tuple[int, float]]:
    """Map topic scores to clusters. Returns [(cluster_id, best_score)]."""
    if not topic_scores:
        return []
    score_map = dict(topic_scores)
    topic_ids = list(score_map.keys())
    placeholders = ",".join("?" * len(topic_ids))

    rows = conn.execute(
        f"""
        SELECT id, cluster_id FROM topics
        WHERE id IN ({placeholders}) AND cluster_id IS NOT NULL
        """,
        topic_ids,
    ).fetchall()

    best: dict[int, float] = {}
    for row in rows:
        topic_id, cluster_id = row[0], row[1]
        score = score_map.get(topic_id, 0.0)
        if cluster_id not in best or score > best[cluster_id]:
            best[cluster_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


# ── Scope-specific hybrid search functions ──


def hybrid_search_books(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """
    Hybrid book search: fuses FTS5 + semantic signals across books, topics, themes, and chunks.
    """

    # Embed query once for all semantic signals
    query_bytes = embedding_to_bytes(embed_text(query))

    # Gather ranked lists
    fts_books = _fts_search_books(conn, query)
    fts_topics = _fts_search_topics(conn, query)
    fts_topic_books = _topics_to_books(conn, fts_topics)
    fts_chunk_books = _fts_search_chunks(conn, query)
    sem_topics = _semantic_search_topics(conn, query, query_bytes=query_bytes)
    sem_topic_books = _topics_to_books(conn, sem_topics)
    sem_theme_books = _semantic_search_book_themes(conn, query, query_bytes=query_bytes)
    sem_book_direct = _semantic_search_books_direct(conn, query, query_bytes=query_bytes)
    sem_chunk_books = _semantic_search_chunks(conn, query, query_bytes=query_bytes)

    # RRF fusion over all 7 signals
    fused = rrf_fuse(
        [
            fts_books,
            fts_topic_books,
            fts_chunk_books,
            sem_topic_books,
            sem_theme_books,
            sem_book_direct,
            sem_chunk_books,
        ]
    )[:limit]
    if not fused:
        return []

    book_ids = [bid for bid, _ in fused]
    placeholders = ",".join("?" * len(book_ids))

    rows = conn.execute(
        f"""
        SELECT id, title, author, calibre_id
        FROM books WHERE id IN ({placeholders})
        """,
        book_ids,
    ).fetchall()

    book_map = {row[0]: dict(row) for row in rows}

    # Determine match_type per book
    fts_book_ids = {bid for bid, _ in fts_books}
    sem_book_ids = {bid for bid, _ in sem_book_direct}
    sem_theme_ids = {bid for bid, _ in sem_theme_books}
    fts_chunk_ids = {bid for bid, _ in fts_chunk_books}
    sem_chunk_ids = {bid for bid, _ in sem_chunk_books}
    sem_topic_ids = {bid for bid, _ in sem_topic_books}

    results = []
    for book_id, score in fused:
        if book_id not in book_map:
            continue
        b = book_map[book_id]
        if book_id in fts_book_ids:
            match_type = "keyword"
        elif book_id in sem_book_ids:
            match_type = "book"
        elif book_id in sem_theme_ids:
            match_type = "theme"
        elif book_id in fts_chunk_ids:
            match_type = "content"
        elif book_id in sem_chunk_ids:
            match_type = "chunk_semantic"
        elif book_id in sem_topic_ids:
            match_type = "semantic"
        else:
            match_type = "topic"
        results.append(
            {
                "book_id": book_id,
                "title": b["title"],
                "author": b["author"],
                "calibre_id": b["calibre_id"],
                "score": round(score, 4),
                "match_type": match_type,
            }
        )

    return results


def hybrid_search_clusters(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """Hybrid cluster search: fuses FTS5 topic→cluster + semantic topic→cluster."""

    fts_topics = _fts_search_topics(conn, query)
    fts_clusters = _topics_to_clusters(conn, fts_topics)
    sem_topics = _semantic_search_topics(conn, query)
    sem_clusters = _topics_to_clusters(conn, sem_topics)

    fused = rrf_fuse([fts_clusters, sem_clusters])[:limit]
    if not fused:
        return []

    cluster_ids = [cid for cid, _ in fused]
    placeholders = ",".join("?" * len(cluster_ids))

    # Get cluster info from cluster_stats (materialized)
    rows = conn.execute(
        f"""
        SELECT cs.cluster_id, cs.size, cs.book_count,
               COALESCE(cl.label, cs.top_label, 'Cluster ' || cs.cluster_id) as label,
               cs.sample_books_json
        FROM cluster_stats cs
        LEFT JOIN cluster_labels cl ON cl.cluster_id = cs.cluster_id
        WHERE cs.cluster_id IN ({placeholders})
        """,
        cluster_ids,
    ).fetchall()

    cluster_map = {row[0]: dict(row) for row in rows}

    results = []
    for cluster_id, score in fused:
        if cluster_id not in cluster_map:
            continue
        c = cluster_map[cluster_id]
        results.append(
            {
                "cluster_id": cluster_id,
                "label": c["label"],
                "size": c["size"],
                "book_count": c["book_count"],
                "score": round(score, 4),
                "sample_books_json": c["sample_books_json"],
            }
        )

    return results


def hybrid_search_domains(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """Hybrid domain search: aggregates cluster results by domain."""
    # Get more clusters than needed, then aggregate
    cluster_results = hybrid_search_clusters(conn, query, limit=100)
    if not cluster_results:
        return []

    cluster_ids = [c["cluster_id"] for c in cluster_results]
    placeholders = ",".join("?" * len(cluster_ids))

    rows = conn.execute(
        f"""
        SELECT cd.cluster_id, cd.domain_id, d.label
        FROM cluster_domains cd
        JOIN domains d ON d.id = cd.domain_id
        WHERE cd.cluster_id IN ({placeholders})
        """,
        cluster_ids,
    ).fetchall()

    cluster_to_domain = {row[0]: (row[1], row[2]) for row in rows}
    cluster_score_map = {c["cluster_id"]: c["score"] for c in cluster_results}

    # Aggregate: best score per domain + count of matching clusters
    domain_scores: dict[int, dict] = {}
    for cluster_id, score in cluster_score_map.items():
        if cluster_id not in cluster_to_domain:
            continue
        domain_id, domain_label = cluster_to_domain[cluster_id]
        if domain_id not in domain_scores:
            domain_scores[domain_id] = {
                "domain_id": domain_id,
                "label": domain_label,
                "score": score,
                "matching_clusters": 1,
            }
        else:
            d = domain_scores[domain_id]
            d["score"] = max(d["score"], score)
            d["matching_clusters"] += 1

    results = sorted(domain_scores.values(), key=lambda x: x["score"], reverse=True)
    for r in results:
        r["score"] = round(r["score"], 4)
    return results[:limit]


def hybrid_search_universe(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Universe search: returns minimal cluster_id + score pairs for 3D highlighting."""
    cluster_results = hybrid_search_clusters(conn, query, limit=limit)
    return [{"cluster_id": c["cluster_id"], "score": c["score"]} for c in cluster_results]
