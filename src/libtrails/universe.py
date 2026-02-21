"""
Generate UMAP projection of community centroids for Galaxy visualization.

Creates a 3D map where semantically similar communities appear close together,
with each community colored by its domain assignment.
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


def compute_community_centroid(cursor: sqlite3.Cursor, community_id: int) -> np.ndarray | None:
    """Compute community centroid by averaging its child cluster centroids.

    For each child cluster, computes a robust centroid (weighted by topic
    occurrence count), then averages all child centroids and L2-normalizes.
    """
    cursor.execute(
        "SELECT cluster_id FROM cluster_communities WHERE community_id = ?",
        (community_id,),
    )
    cluster_ids = [row["cluster_id"] for row in cursor.fetchall()]
    if not cluster_ids:
        return None

    child_centroids = []
    for cluster_id in cluster_ids:
        topics = get_cluster_topics(cursor, cluster_id)
        centroid = compute_robust_centroid(topics)
        if centroid is not None:
            child_centroids.append(centroid)

    if not child_centroids:
        return None

    avg = np.mean(child_centroids, axis=0)
    norm = np.linalg.norm(avg)
    if norm > 0:
        avg = avg / norm
    return avg


def get_community_book_ids(cursor: sqlite3.Cursor, community_id: int) -> list[int]:
    """Get distinct book IDs in a community from book_communities."""
    cursor.execute(
        """
        SELECT book_id FROM book_communities
        WHERE community_id = ?
        ORDER BY book_id
    """,
        (community_id,),
    )
    return [row["book_id"] for row in cursor.fetchall()]


def generate_semantic_colors(
    domain_ids: list[int], domain_embeddings: dict[int, np.ndarray]
) -> dict[int, str]:
    """Map domains to hues based on semantic similarity via PCA.

    Projects domain centroid embeddings onto a single axis with PCA,
    then maps each domain's position to a hue in [0, 330] degrees
    (avoiding wrap-back to red).

    Returns a dict mapping domain_id -> hex color string.
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
        colors[did] = f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    return colors


def generate_universe_data(
    output_path: Path | None = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    db_path: Path | None = None,
) -> dict:
    """
    Generate 3D UMAP projection of community centroids.

    Queries ~200 communities (not ~2,400 raw clusters), computes a centroid
    for each by averaging child cluster centroids, then runs UMAP for 3D layout.

    Args:
        output_path: Where to write JSON (defaults to UNIVERSE_JSON_PATH)
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        db_path: Path to database (defaults to IPAD_DB_PATH)

    Returns:
        Dict with 'communities' and 'domains' lists
    """
    # Lazy import — umap-learn is heavy
    from umap import UMAP

    output_path = output_path or UNIVERSE_JSON_PATH
    db_path = db_path or IPAD_DB_PATH

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get all communities with 3+ topics from materialized stats
    cursor.execute("""
        SELECT cs.community_id, cs.topic_count, cs.book_count,
               cs.top_label, cs.top_topics_json,
               cs.domain_id, cs.domain_label
        FROM community_stats cs
        WHERE cs.topic_count >= 3
        ORDER BY cs.topic_count DESC
    """)
    communities = cursor.fetchall()

    # Pre-fetch cluster counts and top cluster labels per community
    cluster_info: dict[int, dict] = {}
    cursor.execute("""
        SELECT cc.community_id,
               COUNT(*) as cluster_count,
               GROUP_CONCAT(cs.top_label, '||') as cluster_labels
        FROM cluster_communities cc
        JOIN cluster_stats cs ON cs.cluster_id = cc.cluster_id
        GROUP BY cc.community_id
    """)
    for row in cursor.fetchall():
        labels_raw = row["cluster_labels"] or ""
        labels = [lbl for lbl in labels_raw.split("||") if lbl][:8]
        cluster_info[row["community_id"]] = {
            "cluster_count": row["cluster_count"],
            "top_clusters": labels,
        }

    # Compute community centroids (average of child cluster centroids)
    community_data = []
    centroids = []

    for row in communities:
        community_id = row["community_id"]
        centroid = compute_community_centroid(cursor, community_id)

        if centroid is not None:
            label = row["top_label"] or f"community_{community_id}"
            book_ids = get_community_book_ids(cursor, community_id)
            top_topics_raw = json.loads(row["top_topics_json"] or "[]")
            top_topics = [t["label"] if isinstance(t, dict) else t for t in top_topics_raw]
            domain_id = row["domain_id"] if row["domain_id"] is not None else -1
            domain_label = row["domain_label"] or "Unknown"

            ci = cluster_info.get(community_id, {})
            community_data.append(
                {
                    "community_id": community_id,
                    "label": label,
                    "size": row["topic_count"],
                    "book_count": row["book_count"],
                    "book_ids": book_ids,
                    "domain_id": domain_id,
                    "domain_label": domain_label,
                    "cluster_count": ci.get("cluster_count", 0),
                    "top_clusters": ci.get("top_clusters", []),
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
    for i, cd in enumerate(community_data):
        cd["x"] = float(coords_3d[i, 0])
        cd["y"] = float(coords_3d[i, 1])
        cd["z"] = float(coords_3d[i, 2])

    # Compute domain centroid embeddings (average of community centroids per domain)
    domain_centroids: dict[int, list[np.ndarray]] = {}
    for cd, centroid in zip(community_data, centroids):
        did = cd["domain_id"]
        domain_centroids.setdefault(did, []).append(centroid)
    domain_embeddings = {did: np.mean(vecs, axis=0) for did, vecs in domain_centroids.items()}

    # Build domains list with semantic colors, sorted alphabetically by label
    domain_ids = sorted(set(cd["domain_id"] for cd in community_data))
    colors = generate_semantic_colors(domain_ids, domain_embeddings)
    domains = []

    for domain_id in domain_ids:
        domain_label = next(
            (cd["domain_label"] for cd in community_data if cd["domain_id"] == domain_id),
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

    result = {"communities": community_data, "domains": domains}

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    return result
