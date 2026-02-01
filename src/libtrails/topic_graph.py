"""Topic graph construction using igraph."""

import math
from collections import defaultdict

import igraph as ig
import numpy as np

from .database import get_all_topics, get_db, get_topic_embeddings, save_cooccurrence
from .embeddings import bytes_to_embedding


def compute_cooccurrences() -> dict:
    """
    Compute topic co-occurrences from chunks.

    Two topics co-occur if they appear in the same chunk.

    Returns:
        Dict with statistics about co-occurrences
    """
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

        # Count co-occurrences
        cooccur_counts = defaultdict(int)
        for chunk_id, topics in chunk_topics.items():
            for i, t1 in enumerate(topics):
                for t2 in topics[i + 1:]:
                    # Ensure consistent ordering
                    pair = (min(t1, t2), max(t1, t2))
                    cooccur_counts[pair] += 1

        # Compute PMI (Pointwise Mutual Information)
        # Get topic occurrence counts
        cursor.execute("SELECT id, occurrence_count FROM topics")
        topic_counts = {row[0]: row[1] for row in cursor.fetchall()}
        total_chunks = len(chunk_topics)

        # Save to database with PMI
        saved = 0
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

        return {
            "total_chunks": total_chunks,
            "cooccurrence_pairs": saved,
            "unique_topics": len(topic_counts),
        }


def build_topic_graph(
    embedding_threshold: float = 0.5,
    cooccurrence_min: int = 2,
    pmi_min: float = 0.0,
) -> ig.Graph:
    """
    Build a topic graph with edges based on embedding similarity and co-occurrence.

    Args:
        embedding_threshold: Minimum cosine similarity for embedding edges
        cooccurrence_min: Minimum co-occurrence count for edges
        pmi_min: Minimum PMI for co-occurrence edges

    Returns:
        igraph.Graph with topic nodes and weighted edges
    """
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

        # Compute similarity matrix
        sim_matrix = np.dot(embeddings, embeddings.T)

        for i in range(len(embedding_ids)):
            for j in range(i + 1, len(embedding_ids)):
                sim = sim_matrix[i, j]
                if sim >= embedding_threshold:
                    t1, t2 = embedding_ids[i], embedding_ids[j]
                    if t1 in id_to_idx and t2 in id_to_idx:
                        edges.append((id_to_idx[t1], id_to_idx[t2]))
                        weights.append(float(sim))
                        edge_types.append("embedding")

    # Add edges from co-occurrence
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT topic1_id, topic2_id, count, pmi
            FROM topic_cooccurrences
            WHERE count >= ? AND (pmi >= ? OR pmi IS NULL)
        """, (cooccurrence_min, pmi_min))

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
        "embedding_edges": sum(1 for t in g.es["type"] if t == "embedding") if g.ecount() > 0 else 0,
        "cooccurrence_edges": sum(1 for t in g.es["type"] if t == "cooccurrence") if g.ecount() > 0 else 0,
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
    g = build_topic_graph()
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
        results.append({
            "topic_id": g.vs[neighbor_idx]["topic_id"],
            "label": g.vs[neighbor_idx]["label"],
            "occurrence_count": g.vs[neighbor_idx]["occurrence_count"],
            "connection_weight": g.es[edge_id]["weight"],
            "connection_type": g.es[edge_id]["type"],
        })

    # Sort by weight
    results.sort(key=lambda x: x["connection_weight"], reverse=True)
    return results[:limit]


def export_graph_gml(output_path: str) -> str:
    """Export the topic graph to GML format for visualization."""
    g = build_topic_graph()
    g.write_gml(output_path)
    return output_path
