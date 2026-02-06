"""Topic graph construction using igraph."""

import math
from collections import defaultdict

import igraph as ig
import numpy as np

from .config import COOCCURRENCE_MIN_COUNT, EMBEDDING_EDGE_THRESHOLD, PMI_MIN_THRESHOLD
from .database import get_all_topics, get_db, get_topic_embeddings, save_cooccurrence
from .embeddings import bytes_to_embedding


def compute_cooccurrences(progress_file: str | None = None) -> dict:
    """
    Compute topic co-occurrences from chunks.

    Two topics co-occur if they appear in the same chunk.

    Args:
        progress_file: Optional file path to write progress updates (for background runs)

    Returns:
        Dict with statistics about co-occurrences
    """
    import time
    from datetime import datetime

    def log_progress(msg: str):
        """Write progress message to stdout and optionally to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)
        if progress_file:
            with open(progress_file, "w") as f:
                f.write(f"{full_msg}\n")

    with get_db() as conn:
        cursor = conn.cursor()

        # Get all chunk -> topics mappings
        cursor.execute("""
            SELECT chunk_id, topic_id
            FROM chunk_topic_links
            ORDER BY chunk_id
        """)

        # Group topics by chunk
        chunk_topics = defaultdict(list)
        for row in cursor.fetchall():
            chunk_topics[row[0]].append(row[1])

        total_chunks = len(chunk_topics)
        log_progress(f"Computing co-occurrences for {total_chunks:,} chunks...")

        # Count co-occurrences
        cooccur_counts = defaultdict(int)
        start_time = time.time()
        log_interval = 10000  # Log every 10k chunks

        for i, (chunk_id, topics) in enumerate(chunk_topics.items()):
            for j, t1 in enumerate(topics):
                for t2 in topics[j + 1 :]:
                    # Ensure consistent ordering
                    pair = (min(t1, t2), max(t1, t2))
                    cooccur_counts[pair] += 1

            # Log progress periodically
            if (i + 1) % log_interval == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                remaining = (total_chunks - i - 1) / rate if rate > 0 else 0
                pct = 100 * (i + 1) / total_chunks
                log_progress(
                    f"Co-occurrence counting: {i + 1:,}/{total_chunks:,} ({pct:.1f}%) | "
                    f"Pairs: {len(cooccur_counts):,} | Rate: {rate:.1f}/sec | ETA: {remaining / 60:.1f} min"
                )

        log_progress(f"Found {len(cooccur_counts):,} co-occurrence pairs")

        # Compute PMI (Pointwise Mutual Information)
        # Get topic occurrence counts
        cursor.execute("SELECT id, occurrence_count FROM topics")
        topic_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Save to database with PMI
        log_progress(f"Computing PMI and saving {len(cooccur_counts):,} pairs...")
        saved = 0
        save_interval = 50000  # Log every 50k saves
        start_time = time.time()

        for (t1, t2), count in cooccur_counts.items():
            # PMI = log(P(t1,t2) / (P(t1) * P(t2)))
            p_t1 = topic_counts.get(t1, 1) / total_chunks
            p_t2 = topic_counts.get(t2, 1) / total_chunks
            p_joint = count / total_chunks

            if p_t1 > 0 and p_t2 > 0 and p_joint > 0:
                pmi = math.log(p_joint / (p_t1 * p_t2))
            else:
                pmi = 0

            save_cooccurrence(t1, t2, count, pmi)
            saved += 1

            # Log progress periodically
            if saved % save_interval == 0:
                elapsed = time.time() - start_time
                rate = saved / elapsed if elapsed > 0 else 0
                remaining = (len(cooccur_counts) - saved) / rate if rate > 0 else 0
                pct = 100 * saved / len(cooccur_counts)
                log_progress(
                    f"Saving PMI: {saved:,}/{len(cooccur_counts):,} ({pct:.1f}%) | "
                    f"Rate: {rate:.1f}/sec | ETA: {remaining / 60:.1f} min"
                )

        log_progress(f"Co-occurrence computation complete: {saved:,} pairs saved")

        return {
            "total_chunks": total_chunks,
            "cooccurrence_pairs": saved,
            "unique_topics": len(topic_counts),
        }


def build_topic_graph(
    embedding_threshold: float = 0.5,
    cooccurrence_min: int = 2,
    pmi_min: float = 0.0,
    progress_file: str | None = None,
) -> ig.Graph:
    """
    Build a topic graph with edges based on embedding similarity and co-occurrence.

    Args:
        embedding_threshold: Minimum cosine similarity for embedding edges
        cooccurrence_min: Minimum co-occurrence count for edges
        pmi_min: Minimum PMI for co-occurrence edges
        progress_file: Optional file path to write progress updates (for background runs)

    Returns:
        igraph.Graph with topic nodes and weighted edges
    """
    import time
    from datetime import datetime

    def log_progress(msg: str):
        """Write progress message to stdout and optionally to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)
        if progress_file:
            with open(progress_file, "w") as f:
                f.write(f"{full_msg}\n")

    # Get all topics
    topics = get_all_topics()
    if not topics:
        return ig.Graph()

    topic_ids = [t["id"] for t in topics]
    topic_labels = [t["label"] for t in topics]
    id_to_idx = {tid: idx for idx, tid in enumerate(topic_ids)}

    # Create graph with topic nodes
    g = ig.Graph()
    g.add_vertices(len(topics))
    g.vs["topic_id"] = topic_ids
    g.vs["label"] = topic_labels
    g.vs["occurrence_count"] = [t["occurrence_count"] for t in topics]
    g.vs["cluster_id"] = [t.get("cluster_id") for t in topics]

    edges = []
    weights = []
    edge_types = []

    # Add edges from embedding similarity
    topic_data = get_topic_embeddings()
    if topic_data:
        topic_id_to_embedding = {t[0]: bytes_to_embedding(t[1]) for t in topic_data}
        embedding_ids = [t[0] for t in topic_data]
        embeddings = np.array([topic_id_to_embedding[tid] for tid in embedding_ids])

        n = len(embedding_ids)
        total_pairs = n * (n - 1) // 2
        log_progress(f"Computing embedding similarity: {n:,} topics, {total_pairs:,} pairs")

        # Compute similarity matrix
        log_progress("Building similarity matrix...")
        sim_matrix = np.dot(embeddings, embeddings.T)
        log_progress("Similarity matrix complete, scanning for edges...")

        start_time = time.time()
        checked = 0
        log_interval = 1_000_000  # Log every 1M pairs
        edges_found = 0

        for i in range(len(embedding_ids)):
            for j in range(i + 1, len(embedding_ids)):
                checked += 1
                sim = sim_matrix[i, j]
                if sim >= embedding_threshold:
                    t1, t2 = embedding_ids[i], embedding_ids[j]
                    if t1 in id_to_idx and t2 in id_to_idx:
                        edges.append((id_to_idx[t1], id_to_idx[t2]))
                        weights.append(float(sim))
                        edge_types.append("embedding")
                        edges_found += 1

                # Log progress periodically
                if checked % log_interval == 0:
                    elapsed = time.time() - start_time
                    rate = checked / elapsed if elapsed > 0 else 0
                    remaining = (total_pairs - checked) / rate if rate > 0 else 0
                    pct = 100 * checked / total_pairs
                    log_progress(
                        f"Embedding similarity: {checked:,}/{total_pairs:,} ({pct:.1f}%) | "
                        f"Edges: {edges_found:,} | Rate: {rate:,.0f}/sec | ETA: {remaining / 60:.1f} min"
                    )

        # Final log
        elapsed = time.time() - start_time
        log_progress(f"Embedding similarity complete: {edges_found:,} edges in {elapsed:.1f}s")

    # Add edges from co-occurrence
    log_progress("Adding co-occurrence edges...")
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT topic1_id, topic2_id, count, pmi
            FROM topic_cooccurrences
            WHERE count >= ? AND (pmi >= ? OR pmi IS NULL)
        """,
            (cooccurrence_min, pmi_min),
        )

        cooccur_count = 0
        for row in cursor.fetchall():
            t1, t2 = row[0], row[1]
            if t1 in id_to_idx and t2 in id_to_idx:
                idx1, idx2 = id_to_idx[t1], id_to_idx[t2]
                # Avoid duplicate edges
                if (idx1, idx2) not in [(e[0], e[1]) for e in edges]:
                    edges.append((idx1, idx2))
                    # Weight by normalized PMI (or count if PMI is null)
                    weight = row[3] if row[3] else math.log(row[2] + 1)
                    weights.append(float(max(0.1, weight)))
                    edge_types.append("cooccurrence")
                    cooccur_count += 1

    log_progress(f"Co-occurrence edges added: {cooccur_count:,}")

    if edges:
        g.add_edges(edges)
        g.es["weight"] = weights
        g.es["type"] = edge_types

    log_progress(f"Graph complete: {g.vcount()} nodes, {g.ecount()} edges")
    return g


def build_topic_graph_cooccurrence_only(
    cooccurrence_min: int = 5,
    pmi_min: float = 0.0,
) -> ig.Graph:
    """
    Build a topic graph using ONLY co-occurrence edges.

    This is much faster than the full graph builder because it skips
    the O(n²) embedding similarity computation.

    Args:
        cooccurrence_min: Minimum co-occurrence count for edges
        pmi_min: Minimum PMI for co-occurrence edges

    Returns:
        igraph.Graph with topic nodes and weighted edges
    """
    topics = get_all_topics()
    if not topics:
        return ig.Graph()

    topic_ids = [t["id"] for t in topics]
    topic_labels = [t["label"] for t in topics]
    id_to_idx = {tid: idx for idx, tid in enumerate(topic_ids)}

    # Create graph with topic nodes
    g = ig.Graph()
    g.add_vertices(len(topics))
    g.vs["topic_id"] = topic_ids
    g.vs["label"] = topic_labels
    g.vs["occurrence_count"] = [t["occurrence_count"] for t in topics]
    g.vs["cluster_id"] = [t.get("cluster_id") for t in topics]

    edges = []
    weights = []
    edge_types = []

    # Add edges from co-occurrence only
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT topic1_id, topic2_id, count, pmi
            FROM topic_cooccurrences
            WHERE count >= ? AND (pmi >= ? OR pmi IS NULL)
        """,
            (cooccurrence_min, pmi_min),
        )

        for row in cursor.fetchall():
            t1, t2 = row[0], row[1]
            if t1 in id_to_idx and t2 in id_to_idx:
                idx1, idx2 = id_to_idx[t1], id_to_idx[t2]
                edges.append((idx1, idx2))
                # Weight by normalized PMI (or count if PMI is null)
                weight = row[3] if row[3] else math.log(row[2] + 1)
                weights.append(float(max(0.1, weight)))
                edge_types.append("cooccurrence")

    if edges:
        g.add_edges(edges)
        g.es["weight"] = weights
        g.es["type"] = edge_types

    return g


def build_topic_graph_knn(
    cooccurrence_min: int = 5,
    pmi_min: float = 0.0,
    k: int = 10,
) -> ig.Graph:
    """
    Build a topic graph with co-occurrence edges plus k-nearest neighbor embedding edges.

    This adds a controlled number of embedding-based edges (k per topic) instead of
    the O(n²) all-pairs comparison in the full graph.

    Args:
        cooccurrence_min: Minimum co-occurrence count for edges
        pmi_min: Minimum PMI for co-occurrence edges
        k: Number of nearest neighbors per topic

    Returns:
        igraph.Graph with topic nodes and weighted edges
    """
    from sklearn.neighbors import NearestNeighbors

    topics = get_all_topics()
    if not topics:
        return ig.Graph()

    topic_ids = [t["id"] for t in topics]
    topic_labels = [t["label"] for t in topics]
    id_to_idx = {tid: idx for idx, tid in enumerate(topic_ids)}

    # Create graph with topic nodes
    g = ig.Graph()
    g.add_vertices(len(topics))
    g.vs["topic_id"] = topic_ids
    g.vs["label"] = topic_labels
    g.vs["occurrence_count"] = [t["occurrence_count"] for t in topics]
    g.vs["cluster_id"] = [t.get("cluster_id") for t in topics]

    edges = []
    weights = []
    edge_types = []
    edge_set = set()  # Track added edges to avoid duplicates

    # Add edges from co-occurrence
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT topic1_id, topic2_id, count, pmi
            FROM topic_cooccurrences
            WHERE count >= ? AND (pmi >= ? OR pmi IS NULL)
        """,
            (cooccurrence_min, pmi_min),
        )

        for row in cursor.fetchall():
            t1, t2 = row[0], row[1]
            if t1 in id_to_idx and t2 in id_to_idx:
                idx1, idx2 = id_to_idx[t1], id_to_idx[t2]
                edge_key = (min(idx1, idx2), max(idx1, idx2))
                if edge_key not in edge_set:
                    edges.append((idx1, idx2))
                    weight = row[3] if row[3] else math.log(row[2] + 1)
                    weights.append(float(max(0.1, weight)))
                    edge_types.append("cooccurrence")
                    edge_set.add(edge_key)

    # Add k-NN embedding edges
    topic_data = get_topic_embeddings()
    if topic_data:
        topic_id_to_embedding = {t[0]: bytes_to_embedding(t[1]) for t in topic_data}
        embedding_ids = [t[0] for t in topic_data]
        embeddings = np.array([topic_id_to_embedding[tid] for tid in embedding_ids])

        # Build k-NN index with cosine metric
        # Note: sklearn's NearestNeighbors uses "cosine" which is (1 - cosine_similarity)
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings)), metric="cosine")
        knn.fit(embeddings)

        # Query k nearest neighbors for each topic
        distances, indices = knn.kneighbors(embeddings)

        for i, topic_id in enumerate(embedding_ids):
            if topic_id not in id_to_idx:
                continue
            idx1 = id_to_idx[topic_id]

            # Skip self (first neighbor) and add edges to k nearest
            for j in range(1, len(indices[i])):
                neighbor_topic_id = embedding_ids[indices[i][j]]
                if neighbor_topic_id not in id_to_idx:
                    continue
                idx2 = id_to_idx[neighbor_topic_id]

                edge_key = (min(idx1, idx2), max(idx1, idx2))
                if edge_key not in edge_set:
                    # Convert distance to similarity: sim = 1 - distance
                    similarity = 1.0 - distances[i][j]
                    edges.append((idx1, idx2))
                    weights.append(float(max(0.1, similarity)))
                    edge_types.append("embedding_knn")
                    edge_set.add(edge_key)

    if edges:
        g.add_edges(edges)
        g.es["weight"] = weights
        g.es["type"] = edge_types

    return g


def get_graph_stats(g: ig.Graph) -> dict:
    """Get statistics about the topic graph."""
    if g.vcount() == 0:
        return {"nodes": 0, "edges": 0}

    return {
        "nodes": g.vcount(),
        "edges": g.ecount(),
        "density": g.density(),
        "components": len(g.components()),
        "avg_degree": sum(g.degree()) / g.vcount() if g.vcount() > 0 else 0,
        "embedding_edges": sum(1 for t in g.es["type"] if t == "embedding")
        if g.ecount() > 0
        else 0,
        "cooccurrence_edges": sum(1 for t in g.es["type"] if t == "cooccurrence")
        if g.ecount() > 0
        else 0,
    }


def get_related_topics(topic_label: str, limit: int = 10) -> list[dict]:
    """
    Find topics related to the given topic via graph connections.

    Args:
        topic_label: The topic to find relations for
        limit: Maximum number of related topics to return

    Returns:
        List of related topics with connection info
    """
    g = build_topic_graph(
        embedding_threshold=EMBEDDING_EDGE_THRESHOLD,
        cooccurrence_min=COOCCURRENCE_MIN_COUNT,
        pmi_min=PMI_MIN_THRESHOLD,
    )
    if g.vcount() == 0:
        return []

    # Find the topic node
    try:
        node_idx = g.vs["label"].index(topic_label.lower())
    except ValueError:
        # Try partial match
        matches = [i for i, label in enumerate(g.vs["label"]) if topic_label.lower() in label]
        if not matches:
            return []
        node_idx = matches[0]

    # Get neighbors
    neighbors = g.neighbors(node_idx)
    if not neighbors:
        return []

    # Get edge weights to neighbors
    results = []
    for neighbor_idx in neighbors:
        edge_id = g.get_eid(node_idx, neighbor_idx)
        results.append(
            {
                "topic_id": g.vs[neighbor_idx]["topic_id"],
                "label": g.vs[neighbor_idx]["label"],
                "occurrence_count": g.vs[neighbor_idx]["occurrence_count"],
                "connection_weight": g.es[edge_id]["weight"],
                "connection_type": g.es[edge_id]["type"],
            }
        )

    # Sort by weight
    results.sort(key=lambda x: x["connection_weight"], reverse=True)
    return results[:limit]


def export_graph_gml(output_path: str) -> str:
    """Export the topic graph to GML format for visualization."""
    g = build_topic_graph(
        embedding_threshold=EMBEDDING_EDGE_THRESHOLD,
        cooccurrence_min=COOCCURRENCE_MIN_COUNT,
        pmi_min=PMI_MIN_THRESHOLD,
    )
    g.write_gml(output_path)
    return output_path
