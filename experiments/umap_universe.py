"""
Generate UMAP projection of Leiden cluster centroids for Galaxy/Constellation visualization.

Creates a 2D map where semantically similar themes appear close together.
"""

import json
import sqlite3
from pathlib import Path

import numpy as np

try:
    from umap import UMAP
except ImportError:
    print("UMAP not installed. Run: pip install umap-learn")
    raise

# Config
DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = DATA_DIR / "ipad_library.db"
SUPER_CLUSTERS_PATH = Path(__file__).parent / "super_clusters_robust.json"
OUTPUT_PATH = Path(__file__).parent / "universe_coords.json"

# UMAP params
N_NEIGHBORS = 15
MIN_DIST = 0.3
RANDOM_STATE = 42

# Robust centroid params (match super_clusters_robust.py)
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


def compute_robust_centroid(topics: list[dict]) -> np.ndarray | None:
    """Robust centroid: top N topics, minimum label length, weighted."""
    topics = [t for t in topics if len(t['label']) >= MIN_LABEL_LENGTH]
    topics = sorted(topics, key=lambda t: t['occurrence_count'], reverse=True)[:TOP_N_TOPICS]

    if len(topics) < 3:
        return None

    embeddings = np.array([t['embedding'] for t in topics])
    weights = np.array([np.log1p(t['occurrence_count']) for t in topics])
    weights = weights / weights.sum()

    return np.average(embeddings, axis=0, weights=weights)


def get_cluster_label(cursor, cluster_id: int) -> str:
    """Get label for cluster from top topic."""
    cursor.execute("""
        SELECT label FROM topics
        WHERE cluster_id = ? AND LENGTH(label) >= 4
        ORDER BY occurrence_count DESC
        LIMIT 1
    """, (cluster_id,))
    row = cursor.fetchone()
    return row["label"] if row else f"cluster_{cluster_id}"


def get_cluster_book_count(cursor, cluster_id: int) -> int:
    """Get book count for cluster."""
    cursor.execute("""
        SELECT COUNT(DISTINCT b.id) as count
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.cluster_id = ?
    """, (cluster_id,))
    return cursor.fetchone()["count"]


def load_super_cluster_assignments() -> dict[int, dict]:
    """Load super-cluster assignments from robust experiment."""
    if not SUPER_CLUSTERS_PATH.exists():
        print(f"Warning: {SUPER_CLUSTERS_PATH} not found. Run super_clusters_robust.py first.")
        return {}

    with open(SUPER_CLUSTERS_PATH) as f:
        super_clusters = json.load(f)

    # Build reverse mapping: cluster_id -> {super_id, super_label}
    assignments = {}
    for sc in super_clusters:
        super_id = sc["super_cluster_id"]
        super_label = sc["auto_label"]
        for lc in sc["leiden_clusters"]:
            assignments[lc["cluster_id"]] = {
                "domain_id": super_id,
                "domain_label": super_label
            }

    return assignments


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
    print(f"Found {len(clusters)} Leiden clusters")

    # Load super-cluster assignments
    super_assignments = load_super_cluster_assignments()
    print(f"Loaded {len(super_assignments)} super-cluster assignments")

    # Compute robust centroids
    cluster_data = []
    centroids = []

    for row in clusters:
        cluster_id = row["cluster_id"]
        topics = get_cluster_topics(cursor, cluster_id)
        centroid = compute_robust_centroid(topics)

        if centroid is not None:
            label = get_cluster_label(cursor, cluster_id)
            book_count = get_cluster_book_count(cursor, cluster_id)
            domain_info = super_assignments.get(cluster_id, {"domain_id": -1, "domain_label": "Unknown"})

            cluster_data.append({
                "cluster_id": cluster_id,
                "label": label,
                "size": row["size"],
                "book_count": book_count,
                "domain_id": domain_info["domain_id"],
                "domain_label": domain_info["domain_label"]
            })
            centroids.append(centroid)

    print(f"Computed {len(centroids)} robust centroids")

    # UMAP projection
    print(f"\nRunning UMAP (n_neighbors={N_NEIGHBORS}, min_dist={MIN_DIST})...")
    X = np.array(centroids)

    umap = UMAP(
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric='cosine',
        random_state=RANDOM_STATE
    )
    coords_2d = umap.fit_transform(X)

    # Normalize coordinates to [0, 1] range
    coords_2d[:, 0] = (coords_2d[:, 0] - coords_2d[:, 0].min()) / (coords_2d[:, 0].max() - coords_2d[:, 0].min())
    coords_2d[:, 1] = (coords_2d[:, 1] - coords_2d[:, 1].min()) / (coords_2d[:, 1].max() - coords_2d[:, 1].min())

    # Add coordinates to cluster data
    for i, cd in enumerate(cluster_data):
        cd["x"] = float(coords_2d[i, 0])
        cd["y"] = float(coords_2d[i, 1])

    # Build domains list with colors
    domain_ids = sorted(set(cd["domain_id"] for cd in cluster_data))
    colors = generate_domain_colors(len(domain_ids))
    domains = []

    for i, domain_id in enumerate(domain_ids):
        # Find label from first cluster in this domain
        domain_label = next(
            (cd["domain_label"] for cd in cluster_data if cd["domain_id"] == domain_id),
            f"Domain {domain_id}"
        )
        domains.append({
            "domain_id": domain_id,
            "label": domain_label,
            "color": colors[i]
        })

    result = {
        "clusters": cluster_data,
        "domains": domains
    }

    # Save
    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nSaved {len(cluster_data)} clusters to {OUTPUT_PATH}")

    # Summary
    print("\n=== Universe Summary ===")
    print(f"Clusters: {len(cluster_data)}")
    print(f"Domains: {len(domains)}")
    print(f"\nSample clusters:")
    for cd in cluster_data[:5]:
        print(f"  [{cd['cluster_id']:3d}] ({cd['x']:.3f}, {cd['y']:.3f}) {cd['label'][:30]} - {cd['book_count']} books")

    conn.close()


def generate_domain_colors(n: int) -> list[str]:
    """Generate n visually distinct colors using HSL color space."""
    colors = []
    for i in range(n):
        hue = (i * 360 / n) % 360
        # Use medium saturation and lightness for visibility
        colors.append(f"hsl({hue}, 65%, 55%)")
    return colors


if __name__ == "__main__":
    main()
