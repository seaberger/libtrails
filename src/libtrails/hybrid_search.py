"""Hybrid search: FTS5 + sqlite-vec + Reciprocal Rank Fusion."""

import json
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
            WITH knn AS (
                SELECT theme_id, distance
                FROM book_theme_vectors
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT knn.theme_id, knn.distance, bte.book_id
            FROM knn
            JOIN book_theme_entries bte ON bte.id = knn.theme_id
            ORDER BY knn.distance
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
            WITH knn AS (
                SELECT chunk_id, distance
                FROM chunk_vectors
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT knn.chunk_id, knn.distance, c.book_id
            FROM knn
            JOIN chunks c ON c.id = knn.chunk_id
            ORDER BY knn.distance
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


def _search_cluster_labels(
    conn: sqlite3.Connection, query: str, limit: int = 50
) -> list[tuple[int, float]]:
    """Search cluster labels directly via LIKE. Returns [(cluster_id, score)]."""
    tokens = re.sub(r"[^\w\s]", " ", query).lower().split()
    if not tokens:
        return []

    conditions = " OR ".join("LOWER(COALESCE(cl.label, cs.top_label, '')) LIKE ?" for _ in tokens)
    params = [f"%{t}%" for t in tokens]

    try:
        rows = conn.execute(
            f"""
            SELECT cs.cluster_id,
                   COALESCE(cl.label, cs.top_label, '') as label
            FROM cluster_stats cs
            LEFT JOIN cluster_labels cl ON cl.cluster_id = cs.cluster_id
            WHERE {conditions}
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        label = row[1].lower()
        matched = sum(1 for t in tokens if t in label)
        results.append((row[0], matched / len(tokens)))
    return sorted(results, key=lambda x: x[1], reverse=True)


def _fts_chunks_to_clusters(
    conn: sqlite3.Connection, query: str, limit: int = 200
) -> list[tuple[int, float]]:
    """FTS5 chunk search mapped to clusters. Returns [(cluster_id, best_score)]."""
    safe_q = _fts5_safe_query(query)
    try:
        rows = conn.execute(
            """
            SELECT chunks_fts.rowid, bm25(chunks_fts) as score
            FROM chunks_fts
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (safe_q, limit),
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    chunk_scores = {row[0]: -row[1] for row in rows}
    chunk_ids = list(chunk_scores.keys())
    placeholders = ",".join("?" * len(chunk_ids))

    try:
        topic_rows = conn.execute(
            f"""
            SELECT ctl.chunk_id, t.cluster_id
            FROM chunk_topic_links ctl
            JOIN topics t ON ctl.topic_id = t.id
            WHERE ctl.chunk_id IN ({placeholders}) AND t.cluster_id IS NOT NULL
            """,
            chunk_ids,
        ).fetchall()
    except Exception:
        return []

    best: dict[int, float] = {}
    for row in topic_rows:
        chunk_id, cluster_id = row[0], row[1]
        score = chunk_scores.get(chunk_id, 0.0)
        if cluster_id not in best or score > best[cluster_id]:
            best[cluster_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _semantic_chunks_to_clusters(
    conn: sqlite3.Connection, query: str, limit: int = 100, query_bytes: bytes | None = None
) -> list[tuple[int, float]]:
    """Semantic chunk search mapped to clusters. Returns [(cluster_id, best_similarity)]."""
    if query_bytes is None:
        query_bytes = embedding_to_bytes(embed_text(query))
    try:
        rows = conn.execute(
            """
            WITH knn AS (
                SELECT chunk_id, distance
                FROM chunk_vectors
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT knn.chunk_id, knn.distance
            FROM knn
            ORDER BY knn.distance
            """,
            (query_bytes, limit),
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    chunk_scores = {row[0]: 1.0 - row[1] for row in rows}
    chunk_ids = list(chunk_scores.keys())
    placeholders = ",".join("?" * len(chunk_ids))

    try:
        topic_rows = conn.execute(
            f"""
            SELECT ctl.chunk_id, t.cluster_id
            FROM chunk_topic_links ctl
            JOIN topics t ON ctl.topic_id = t.id
            WHERE ctl.chunk_id IN ({placeholders}) AND t.cluster_id IS NOT NULL
            """,
            chunk_ids,
        ).fetchall()
    except Exception:
        return []

    best: dict[int, float] = {}
    for row in topic_rows:
        chunk_id, cluster_id = row[0], row[1]
        score = chunk_scores.get(chunk_id, 0.0)
        if cluster_id not in best or score > best[cluster_id]:
            best[cluster_id] = score
    return sorted(best.items(), key=lambda x: x[1], reverse=True)


def _search_domain_labels(
    conn: sqlite3.Connection, query: str, limit: int = 26
) -> list[tuple[int, float]]:
    """Search domain labels directly via LIKE. Returns [(domain_id, score)]."""
    tokens = re.sub(r"[^\w\s]", " ", query).lower().split()
    if not tokens:
        return []

    conditions = " OR ".join("LOWER(label) LIKE ?" for _ in tokens)
    params = [f"%{t}%" for t in tokens]

    try:
        rows = conn.execute(
            f"""
            SELECT id, label FROM domains
            WHERE {conditions}
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        label = row[1].lower()
        matched = sum(1 for t in tokens if t in label)
        results.append((row[0], matched / len(tokens)))
    return sorted(results, key=lambda x: x[1], reverse=True)


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

    # Determine match_type per book: whichever signal ranked the book highest
    signal_labels = [
        (fts_books, "keyword"),
        (fts_topic_books, "topic"),
        (fts_chunk_books, "content"),
        (sem_topic_books, "semantic"),
        (sem_theme_books, "theme"),
        (sem_book_direct, "book"),
        (sem_chunk_books, "chunk"),
    ]
    # For each book, find the signal where it appeared at the lowest rank (= strongest)
    best_signal: dict[int, tuple[int, str]] = {}  # book_id -> (best_rank, label)
    for ranked_list, label in signal_labels:
        for rank, (bid, _score) in enumerate(ranked_list):
            if bid not in best_signal or rank < best_signal[bid][0]:
                best_signal[bid] = (rank, label)

    results = []
    for book_id, score in fused:
        if book_id not in book_map:
            continue
        b = book_map[book_id]
        match_type = best_signal[book_id][1] if book_id in best_signal else "topic"
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
    """Hybrid cluster search: fuses 5 signals — topic FTS/semantic, cluster labels, chunk FTS/semantic."""

    # Embed query once for all semantic signals
    query_bytes = embedding_to_bytes(embed_text(query))

    # 5 signals
    fts_topics = _fts_search_topics(conn, query)
    fts_topic_clusters = _topics_to_clusters(conn, fts_topics)
    sem_topics = _semantic_search_topics(conn, query, query_bytes=query_bytes)
    sem_topic_clusters = _topics_to_clusters(conn, sem_topics)
    fts_label_clusters = _search_cluster_labels(conn, query)
    fts_chunk_clusters = _fts_chunks_to_clusters(conn, query)
    sem_chunk_clusters = _semantic_chunks_to_clusters(conn, query, query_bytes=query_bytes)

    fused = rrf_fuse(
        [
            fts_topic_clusters,
            sem_topic_clusters,
            fts_label_clusters,
            fts_chunk_clusters,
            sem_chunk_clusters,
        ]
    )[:limit]
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


def hybrid_search_communities(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """Hybrid community search: aggregates cluster results by community + direct label match."""
    # Get more clusters than needed, then aggregate
    cluster_results = hybrid_search_clusters(conn, query, limit=100)

    # Direct community label search
    community_label_matches = _search_community_labels(conn, query)
    community_label_scores = dict(community_label_matches)

    # Build community→label mapping for label matches not found via clusters
    community_labels: dict[int, str] = {}
    if community_label_matches:
        community_ids = [cid for cid, _ in community_label_matches]
        placeholders_c = ",".join("?" * len(community_ids))
        label_rows = conn.execute(
            f"SELECT community_id, top_label FROM community_stats WHERE community_id IN ({placeholders_c})",
            community_ids,
        ).fetchall()
        community_labels = {row[0]: row[1] for row in label_rows}

    # Aggregate cluster results by community
    community_scores: dict[int, dict] = {}

    if cluster_results:
        cluster_ids = [c["cluster_id"] for c in cluster_results]
        placeholders = ",".join("?" * len(cluster_ids))

        rows = conn.execute(
            f"""
            SELECT cc.cluster_id, cc.community_id,
                   cs.top_label as label, cs.topic_count, cs.book_count,
                   cs.sample_books_json
            FROM cluster_communities cc
            JOIN community_stats cs ON cs.community_id = cc.community_id
            WHERE cc.cluster_id IN ({placeholders})
            """,
            cluster_ids,
        ).fetchall()

        cluster_to_community = {row[0]: row for row in rows}
        cluster_score_map = {c["cluster_id"]: c["score"] for c in cluster_results}

        for cluster_id, score in cluster_score_map.items():
            if cluster_id not in cluster_to_community:
                continue
            row = cluster_to_community[cluster_id]
            community_id = row["community_id"]
            if community_id not in community_scores:
                community_scores[community_id] = {
                    "community_id": community_id,
                    "label": row["label"],
                    "topic_count": row["topic_count"],
                    "book_count": row["book_count"],
                    "score": score,
                    "sample_books_json": row["sample_books_json"],
                    "matching_clusters": 1,
                }
            else:
                d = community_scores[community_id]
                d["score"] = max(d["score"], score)
                d["matching_clusters"] += 1

    # Merge direct community label matches
    label_boost = 0.02
    for community_id, label_score in community_label_scores.items():
        if community_id in community_scores:
            community_scores[community_id]["score"] += label_boost * label_score
        elif community_id in community_labels:
            # Need to fetch stats for this community
            row = conn.execute(
                "SELECT topic_count, book_count, sample_books_json FROM community_stats WHERE community_id = ?",
                (community_id,),
            ).fetchone()
            if row:
                community_scores[community_id] = {
                    "community_id": community_id,
                    "label": community_labels[community_id],
                    "topic_count": row["topic_count"],
                    "book_count": row["book_count"],
                    "score": label_boost * label_score,
                    "sample_books_json": row["sample_books_json"],
                    "matching_clusters": 0,
                }

    results = sorted(community_scores.values(), key=lambda x: x["score"], reverse=True)
    for r in results:
        r["score"] = round(r["score"], 4)
    return results[:limit]


def _search_community_labels(
    conn: sqlite3.Connection, query: str, limit: int = 50
) -> list[tuple[int, float]]:
    """Search community labels directly via LIKE. Returns [(community_id, score)]."""
    tokens = re.sub(r"[^\w\s]", " ", query).lower().split()
    if not tokens:
        return []

    conditions = " OR ".join("LOWER(COALESCE(top_label, '')) LIKE ?" for _ in tokens)
    params = [f"%{t}%" for t in tokens]

    try:
        rows = conn.execute(
            f"""
            SELECT community_id, top_label FROM community_stats
            WHERE {conditions}
            LIMIT ?
            """,
            [*params, limit],
        ).fetchall()
    except Exception:
        return []

    results = []
    for row in rows:
        label = (row[1] or "").lower()
        matched = sum(1 for t in tokens if t in label)
        results.append((row[0], matched / len(tokens)))
    return sorted(results, key=lambda x: x[1], reverse=True)


def hybrid_search_domains(conn: sqlite3.Connection, query: str, limit: int = 20) -> list[dict]:
    """Hybrid domain search: aggregates cluster results by domain + direct domain label match."""
    # Get more clusters than needed, then aggregate
    cluster_results = hybrid_search_clusters(conn, query, limit=100)

    # Direct domain label search as additional signal
    domain_label_matches = _search_domain_labels(conn, query)
    domain_label_scores = dict(domain_label_matches)

    # Build domain→label mapping from DB for label matches not found via clusters
    domain_labels: dict[int, str] = {}
    if domain_label_matches:
        domain_ids = [did for did, _ in domain_label_matches]
        placeholders_d = ",".join("?" * len(domain_ids))
        label_rows = conn.execute(
            f"SELECT id, label FROM domains WHERE id IN ({placeholders_d})",
            domain_ids,
        ).fetchall()
        domain_labels = {row[0]: row[1] for row in label_rows}

    # Aggregate cluster results by domain
    domain_scores: dict[int, dict] = {}

    if cluster_results:
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

    # Merge direct domain label matches (additive boost scaled to RRF range)
    # RRF scores are ~0.01-0.1; label scores are 0-1. Scale label to RRF range.
    label_boost = 0.02  # ~1/(k+1) = 0.0164 for k=60
    for domain_id, label_score in domain_label_scores.items():
        if domain_id in domain_scores:
            domain_scores[domain_id]["score"] += label_boost * label_score
        elif domain_id in domain_labels:
            domain_scores[domain_id] = {
                "domain_id": domain_id,
                "label": domain_labels[domain_id],
                "score": label_boost * label_score,
                "matching_clusters": 0,
            }

    results = sorted(domain_scores.values(), key=lambda x: x["score"], reverse=True)
    for r in results:
        r["score"] = round(r["score"], 4)
    return results[:limit]


def hybrid_search_universe(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Universe search: returns minimal community_id + score pairs for 3D highlighting.

    Aggregates cluster-level search results by community, keeping the best
    score per community. Same pattern as hybrid_search_communities() but
    returns only the IDs and scores needed for sphere highlighting.
    """
    cluster_results = hybrid_search_clusters(conn, query, limit=200)
    if not cluster_results:
        return []

    cluster_ids = [c["cluster_id"] for c in cluster_results]
    placeholders = ",".join("?" * len(cluster_ids))

    rows = conn.execute(
        f"""
        SELECT cluster_id, community_id
        FROM cluster_communities
        WHERE cluster_id IN ({placeholders})
        """,
        cluster_ids,
    ).fetchall()

    cluster_to_community = {row[0]: row[1] for row in rows}

    # Aggregate best score per community
    best: dict[int, float] = {}
    for c in cluster_results:
        community_id = cluster_to_community.get(c["cluster_id"])
        if community_id is None:
            continue
        score = c["score"]
        if community_id not in best or score > best[community_id]:
            best[community_id] = score

    results = [
        {"community_id": cid, "score": round(score, 4)}
        for cid, score in sorted(best.items(), key=lambda x: x[1], reverse=True)
    ]
    return results[:limit]


def find_related_books(conn: sqlite3.Connection, book_id: int, limit: int = 12) -> list[dict]:
    """Find related books using hybrid search on the source book's metadata.

    Builds a query from the book's title, author, and themes, runs it through
    hybrid_search_books(), filters out the source book, and normalizes scores
    to 0-1 range.
    """
    row = conn.execute(
        "SELECT title, author, book_themes FROM books WHERE id = ?",
        (book_id,),
    ).fetchone()
    if not row:
        return []

    title = row["title"] or ""
    author = row["author"] or ""
    themes_json = row["book_themes"]
    themes = json.loads(themes_json) if themes_json else []

    # Build query string matching the book_vectors embedding format
    query_parts = [f"{title} by {author}"]
    if themes:
        query_parts.append("Themes: " + ", ".join(themes[:10]))
    query = ". ".join(query_parts)

    # Fetch extra results to account for self-filtering
    raw_results = hybrid_search_books(conn, query, limit + 5)

    # Filter out the source book
    filtered = [r for r in raw_results if r["book_id"] != book_id][:limit]
    if not filtered:
        return filtered

    # Normalize scores to 0-1 (top match ≈ 1.0)
    max_score = filtered[0]["score"]
    if max_score > 0:
        for r in filtered:
            r["similarity"] = round(r["score"] / max_score, 3)
    else:
        for r in filtered:
            r["similarity"] = 0.0

    return filtered
