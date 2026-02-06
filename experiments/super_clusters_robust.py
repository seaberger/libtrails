"""
Super-cluster experiment with robust centroids.

Robust centroid approach:
- Filter topics with labels < 4 chars (junk like "a", "the", etc.)
- Use top N topics by occurrence count (not all topics)
- Weight by log1p(occurrence_count) for stable centroids
"""

import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

# Config
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "ipad_library.db"
OUTPUT_PATH = Path(__file__).parent / "super_clusters_robust.json"

N_SUPER_CLUSTERS = 25
TOP_N_TOPICS = 15
MIN_LABEL_LENGTH = 4


def get_cluster_topics(cursor, cluster_id: int) -> list[dict]:
    """Get topics for a cluster with their embeddings and occurrence counts."""
    cursor.execute("""
        SELECT id, label, embedding, occurrence_count
        FROM topics
        WHERE cluster_id = ? AND embedding IS NOT NULL
        ORDER BY occurrence_count DESC
    """, (cluster_id,))

    topics = []
    for row in cursor.fetchall():
        if row["embedding"]:
            topics.append({
                "id": row["id"],
                "label": row["label"],
                "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
                "occurrence_count": row["occurrence_count"] or 1
            })
    return topics


def compute_robust_centroid(topics: list[dict], top_n: int = TOP_N_TOPICS, min_label_length: int = MIN_LABEL_LENGTH) -> np.ndarray | None:
    """
    Robust centroid: top N topics, minimum label length, weighted.

    - Filter out short labels (< min_label_length chars)
    - Take top N by occurrence count
    - Weight by log1p(occurrence_count)
    """
    # Filter short labels
    topics = [t for t in topics if len(t['label']) >= min_label_length]

    # Take top N by occurrence
    topics = sorted(topics, key=lambda t: t['occurrence_count'], reverse=True)[:top_n]

    # Need minimum topics for a meaningful centroid
    if len(topics) < 3:
        return None

    embeddings = np.array([t['embedding'] for t in topics])
    weights = np.array([np.log1p(t['occurrence_count']) for t in topics])
    weights = weights / weights.sum()  # Normalize

    return np.average(embeddings, axis=0, weights=weights)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all Leiden clusters
    cursor.execute("""
        SELECT DISTINCT cluster_id, COUNT(*) as size
        FROM topics
        WHERE cluster_id IS NOT NULL AND cluster_id >= 0
        GROUP BY cluster_id
        HAVING size >= 3
        ORDER BY size DESC
    """)

    clusters = cursor.fetchall()
    print(f"Found {len(clusters)} Leiden clusters with 3+ topics")

    # Compute robust centroids
    cluster_ids = []
    centroids = []
    skipped = 0

    for row in clusters:
        cluster_id = row["cluster_id"]
        topics = get_cluster_topics(cursor, cluster_id)
        centroid = compute_robust_centroid(topics)

        if centroid is not None:
            cluster_ids.append(cluster_id)
            centroids.append(centroid)
        else:
            skipped += 1

    print(f"Computed {len(centroids)} robust centroids (skipped {skipped} clusters with insufficient quality topics)")

    # K-means clustering
    X = np.array(centroids)
    kmeans = KMeans(n_clusters=N_SUPER_CLUSTERS, random_state=42, n_init=10)
    super_labels = kmeans.fit_predict(X)

    # Build super-cluster mapping
    super_clusters = {}
    for i, (cluster_id, super_id) in enumerate(zip(cluster_ids, super_labels)):
        super_id = int(super_id)
        if super_id not in super_clusters:
            super_clusters[super_id] = {
                "super_cluster_id": super_id,
                "leiden_clusters": [],
                "top_topics": []
            }

        # Get top topics for this Leiden cluster
        cursor.execute("""
            SELECT label, occurrence_count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
            LIMIT 3
        """, (cluster_id,))
        top = [{"label": r["label"], "count": r["occurrence_count"]} for r in cursor.fetchall()]

        super_clusters[super_id]["leiden_clusters"].append({
            "cluster_id": cluster_id,
            "top_topics": top
        })

    # Generate labels for each super-cluster
    for super_id, data in super_clusters.items():
        # Aggregate top topics across all Leiden clusters
        all_topics = {}
        for lc in data["leiden_clusters"]:
            for t in lc["top_topics"]:
                label = t["label"]
                if label not in all_topics:
                    all_topics[label] = 0
                all_topics[label] += t["count"]

        # Filter short labels and get top 10
        filtered = [(k, v) for k, v in all_topics.items() if len(k) >= MIN_LABEL_LENGTH]
        sorted_topics = sorted(filtered, key=lambda x: x[1], reverse=True)[:10]
        data["top_topics"] = [{"label": t[0], "total_count": t[1]} for t in sorted_topics]

        # Auto-generate a label from top 3 topics
        if sorted_topics:
            data["auto_label"] = " / ".join([t[0] for t in sorted_topics[:3]])
        else:
            data["auto_label"] = f"Domain {super_id}"

    # Sort by number of Leiden clusters
    result = sorted(super_clusters.values(), key=lambda x: len(x["leiden_clusters"]), reverse=True)

    # Summary
    print(f"\n=== {N_SUPER_CLUSTERS} Super-Clusters (Robust Centroids) ===\n")
    for sc in result:
        n_leiden = len(sc["leiden_clusters"])
        label = sc["auto_label"]
        print(f"[{sc['super_cluster_id']:2d}] {n_leiden:3d} clusters: {label}")

    # Save results
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    conn.close()


if __name__ == "__main__":
    main()
