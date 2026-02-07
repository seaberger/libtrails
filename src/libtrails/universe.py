"""
Generate UMAP projection of Leiden cluster centroids for Galaxy visualization.

Creates a 3D map where semantically similar themes appear close together,
with each cluster colored by its domain assignment.
"""

import colorsys
import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from .config import IPAD_DB_PATH, UNIVERSE_JSON_PATH
from .domains import compute_robust_centroid, get_cluster_topics

# UMAP defaults
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.3
RANDOM_STATE = 42


def get_cluster_label(cursor: sqlite3.Cursor, cluster_id: int) -> str:
    """Get label for cluster from its top topic."""
    cursor.execute(
        """
        SELECT label FROM topics
        WHERE cluster_id = ? AND LENGTH(label) >= 4
        ORDER BY occurrence_count DESC
        LIMIT 1
    """,
        (cluster_id,),
    )
    row = cursor.fetchone()
    return row["label"] if row else f"cluster_{cluster_id}"


def get_cluster_book_count(cursor: sqlite3.Cursor, cluster_id: int) -> int:
    """Get number of distinct books touching a cluster."""
    cursor.execute(
        """
        SELECT COUNT(DISTINCT b.id) as count
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.cluster_id = ?
    """,
        (cluster_id,),
    )
    return cursor.fetchone()["count"]


def get_cluster_top_topics(cursor: sqlite3.Cursor, cluster_id: int, limit: int = 5) -> list[str]:
    """Get top topic labels for a cluster."""
    cursor.execute(
        """
        SELECT label FROM topics
        WHERE cluster_id = ? AND LENGTH(label) >= 4
        ORDER BY occurrence_count DESC
        LIMIT ?
    """,
        (cluster_id, limit),
    )
    return [row["label"] for row in cursor.fetchall()]


def load_domain_assignments_from_db(cursor: sqlite3.Cursor) -> dict[int, dict]:
    """Load domain assignments from the domains + cluster_domains tables."""
    cursor.execute(
        """
        SELECT cd.cluster_id, d.id as domain_id, d.label as domain_label
        FROM cluster_domains cd
        JOIN domains d ON d.id = cd.domain_id
    """
    )
    assignments = {}
    for row in cursor.fetchall():
        assignments[row["cluster_id"]] = {
            "domain_id": row["domain_id"],
            "domain_label": row["domain_label"],
        }
    return assignments


def generate_semantic_colors(
    domain_ids: list[int], domain_embeddings: dict[int, np.ndarray]
) -> dict[int, str]:
    """Map domains to hues based on semantic similarity via PCA.

    Projects domain centroid embeddings onto a single axis with PCA,
    then maps each domain's position to a hue in [0, 330] degrees
    (avoiding wrap-back to red).

    Returns a dict mapping domain_id → hex color string.
    """
    ordered_ids = [did for did in domain_ids if did in domain_embeddings]
    if not ordered_ids:
        return {}

    matrix = np.array([domain_embeddings[did] for did in ordered_ids])

    if len(ordered_ids) == 1:
        positions = np.array([0.5])
    else:
        pca = PCA(n_components=1, random_state=RANDOM_STATE)
        projected = pca.fit_transform(matrix).ravel()
        lo, hi = projected.min(), projected.max()
        if hi > lo:
            positions = (projected - lo) / (hi - lo)
        else:
            positions = np.full(len(ordered_ids), 0.5)

    colors: dict[int, str] = {}
    for did, pos in zip(ordered_ids, positions):
        hue = pos * 330  # 0-330° avoids red↔red wrap
        saturation = 0.70
        lightness = 0.60
        r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
        colors[did] = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

    return colors


def generate_universe_data(
    output_path: Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    db_path: Path | None = None,
) -> dict:
    """
    Generate 3D UMAP projection of all Leiden cluster centroids.

    Args:
        output_path: Where to write JSON (defaults to UNIVERSE_JSON_PATH)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        db_path: Path to database (defaults to IPAD_DB_PATH)

    Returns:
        Dict with 'clusters' and 'domains' lists
    """
    # Lazy import — umap-learn is heavy
    from umap import UMAP

    output_path = output_path or UNIVERSE_JSON_PATH
    db_path = db_path or IPAD_DB_PATH

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all Leiden clusters with 3+ topics
    cursor.execute("""
        SELECT DISTINCT cluster_id, COUNT(*) as size
        FROM topics
        WHERE cluster_id IS NOT NULL AND cluster_id >= 0
        GROUP BY cluster_id
        HAVING size >= 3
        ORDER BY size DESC
    """)
    clusters = cursor.fetchall()

    # Load domain assignments from DB
    domain_assignments = load_domain_assignments_from_db(cursor)

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
            top_topics = get_cluster_top_topics(cursor, cluster_id)
            domain_info = domain_assignments.get(
                cluster_id, {"domain_id": -1, "domain_label": "Unknown"}
            )

            cluster_data.append(
                {
                    "cluster_id": cluster_id,
                    "label": label,
                    "size": row["size"],
                    "book_count": book_count,
                    "domain_id": domain_info["domain_id"],
                    "domain_label": domain_info["domain_label"],
                    "top_topics": top_topics,
                }
            )
            centroids.append(centroid)

    conn.close()

    # 3D UMAP projection
    X = np.array(centroids)
    umap = UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords_3d = umap.fit_transform(X)

    # Normalize coordinates to [-1, 1] centered at origin (better for 3D viewing)
    for axis in range(3):
        col = coords_3d[:, axis]
        col_min, col_max = col.min(), col.max()
        if col_max > col_min:
            coords_3d[:, axis] = 2 * (col - col_min) / (col_max - col_min) - 1

    # Attach coordinates
    for i, cd in enumerate(cluster_data):
        cd["x"] = float(coords_3d[i, 0])
        cd["y"] = float(coords_3d[i, 1])
        cd["z"] = float(coords_3d[i, 2])

    # Compute domain centroid embeddings (average of cluster centroids per domain)
    domain_centroids: dict[int, list[np.ndarray]] = {}
    for cd, centroid in zip(cluster_data, centroids):
        did = cd["domain_id"]
        domain_centroids.setdefault(did, []).append(centroid)
    domain_embeddings = {
        did: np.mean(vecs, axis=0) for did, vecs in domain_centroids.items()
    }

    # Build domains list with semantic colors, sorted alphabetically by label
    domain_ids = sorted(set(cd["domain_id"] for cd in cluster_data))
    colors = generate_semantic_colors(domain_ids, domain_embeddings)
    domains = []

    for domain_id in domain_ids:
        domain_label = next(
            (cd["domain_label"] for cd in cluster_data if cd["domain_id"] == domain_id),
            f"Domain {domain_id}",
        )
        domains.append(
            {
                "domain_id": domain_id,
                "label": domain_label,
                "color": colors.get(domain_id, "#888888"),
            }
        )

    domains.sort(key=lambda d: d["label"])

    result = {"clusters": cluster_data, "domains": domains}

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
