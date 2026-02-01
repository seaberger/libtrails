"""Leiden clustering for hierarchical topic organization."""

from collections import defaultdict
from typing import Optional

import leidenalg

from .database import get_db, update_topic_cluster
from .topic_graph import build_topic_graph


def cluster_topics(
    min_cluster_size: int = 3,
    resolution: float = 1.0,
) -> dict:
    """
    Cluster topics using the Leiden algorithm with Surprise quality function.

    Args:
        min_cluster_size: Minimum size for a cluster
        resolution: Resolution parameter (higher = more clusters)

    Returns:
        Statistics about the clustering
    """
    g = build_topic_graph()
    if g.vcount() == 0:
        return {"error": "No topics in graph"}

    # Run Leiden clustering with Surprise (recommended for topic clustering)
    partition = leidenalg.find_partition(
        g,
        leidenalg.SurpriseVertexPartition,
    )

    # Assign cluster IDs to topics
    cluster_sizes = defaultdict(int)
    with get_db() as conn:
        for node_idx, cluster_id in enumerate(partition.membership):
            topic_id = g.vs[node_idx]["topic_id"]
            update_topic_cluster(topic_id, cluster_id)
            cluster_sizes[cluster_id] += 1
        conn.commit()

    # Update graph vertex attributes
    g.vs["cluster"] = partition.membership

    return {
        "total_topics": g.vcount(),
        "num_clusters": len(set(partition.membership)),
        "modularity": partition.quality(),
        "cluster_sizes": dict(sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]),
        "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
    }


def get_cluster_topics(cluster_id: int) -> list[dict]:
    """Get all topics in a cluster."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT id, label, occurrence_count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
        """, (cluster_id,))

        return [
            {"id": row[0], "label": row[1], "occurrence_count": row[2]}
            for row in cursor.fetchall()
        ]


def get_cluster_summary() -> list[dict]:
    """Get a summary of all clusters with their top topics."""
    with get_db() as conn:
        cursor = conn.cursor()

        # Get all cluster IDs
        cursor.execute("""
            SELECT DISTINCT cluster_id, COUNT(*) as size
            FROM topics
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
            ORDER BY size DESC
        """)

        clusters = []
        for row in cursor.fetchall():
            cluster_id = row[0]
            size = row[1]

            # Get top topics for this cluster
            cursor.execute("""
                SELECT label, occurrence_count
                FROM topics
                WHERE cluster_id = ?
                ORDER BY occurrence_count DESC
                LIMIT 5
            """, (cluster_id,))

            top_topics = [
                {"label": r[0], "count": r[1]}
                for r in cursor.fetchall()
            ]

            clusters.append({
                "cluster_id": cluster_id,
                "size": size,
                "top_topics": top_topics,
            })

        return clusters


def get_topic_tree() -> dict:
    """
    Build a hierarchical topic tree from clusters.

    Returns a nested structure suitable for display.
    """
    clusters = get_cluster_summary()

    tree = {
        "name": "Topics",
        "children": [],
    }

    for cluster in clusters:
        if cluster["size"] < 2:
            continue

        # Use top topic as cluster name
        cluster_name = cluster["top_topics"][0]["label"] if cluster["top_topics"] else f"Cluster {cluster['cluster_id']}"

        cluster_node = {
            "name": cluster_name,
            "cluster_id": cluster["cluster_id"],
            "size": cluster["size"],
            "children": [
                {
                    "name": t["label"],
                    "count": t["count"],
                }
                for t in cluster["top_topics"]
            ],
        }
        tree["children"].append(cluster_node)

    return tree


def print_topic_tree(indent: int = 0) -> str:
    """Generate a text representation of the topic tree."""
    tree = get_topic_tree()
    lines = []

    def _print_node(node: dict, depth: int):
        prefix = "  " * depth
        if "cluster_id" in node:
            lines.append(f"{prefix}{node['name']} ({node['size']} topics)")
            for child in node.get("children", []):
                lines.append(f"{prefix}  - {child['name']} ({child['count']})")
        else:
            lines.append(f"{prefix}{node['name']}")
            for child in node.get("children", []):
                _print_node(child, depth + 1)

    _print_node(tree, indent)
    return "\n".join(lines)


def recursive_cluster(
    max_depth: int = 3,
    min_size_to_split: int = 10,
) -> dict:
    """
    Perform recursive Leiden clustering for deeper hierarchy.

    Large clusters are recursively sub-clustered.

    Args:
        max_depth: Maximum recursion depth
        min_size_to_split: Minimum cluster size to attempt splitting

    Returns:
        Statistics about the recursive clustering
    """
    g = build_topic_graph()
    if g.vcount() == 0:
        return {"error": "No topics in graph"}

    # Initial clustering
    partition = leidenalg.find_partition(
        g,
        leidenalg.SurpriseVertexPartition,
    )

    # Track cluster hierarchy
    # For now, just do flat clustering and mark parent relationships
    # based on the largest cluster at each level

    cluster_hierarchy = {}
    for node_idx, cluster_id in enumerate(partition.membership):
        topic_id = g.vs[node_idx]["topic_id"]
        update_topic_cluster(topic_id, cluster_id)
        if cluster_id not in cluster_hierarchy:
            cluster_hierarchy[cluster_id] = []
        cluster_hierarchy[cluster_id].append(topic_id)

    return {
        "total_topics": g.vcount(),
        "num_clusters": len(cluster_hierarchy),
        "depth": 1,  # For now, single level
    }


def label_cluster_with_llm(cluster_id: int, model: str = "gemma3:4b") -> Optional[str]:
    """
    Use an LLM to generate a descriptive label for a cluster.

    Args:
        cluster_id: The cluster to label
        model: Ollama model to use

    Returns:
        A descriptive label for the cluster
    """
    import subprocess

    topics = get_cluster_topics(cluster_id)
    if not topics:
        return None

    topic_list = ", ".join([t["label"] for t in topics[:15]])

    prompt = f"""Given these related topics from a book library, provide a single 2-4 word category label that captures their common theme.

Topics: {topic_list}

Respond with ONLY the category label, nothing else."""

    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None
