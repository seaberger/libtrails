"""
Domain (super-cluster) generation and management.

Domains are high-level thematic groupings of Leiden clusters.
The process:
1. Compute robust centroids for each Leiden cluster
2. Use K-means to group clusters into super-clusters
3. Apply human-refined labels to create final domains
"""

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .config import IPAD_DB_PATH

# Default configuration
DEFAULT_N_DOMAINS = 25
DEFAULT_TOP_N_TOPICS = 15
DEFAULT_MIN_LABEL_LENGTH = 4


def disparity_filter(g: "ig.Graph", alpha: float = 0.05) -> "ig.Graph":
    """Remove statistically insignificant edges using the disparity filter.

    Keeps edges where the weight is significant (p < alpha) for at
    least one endpoint, given the null hypothesis of uniform weight
    distribution across each node's edges.

    Reference: Serrano et al. (2009) "Extracting the multiscale backbone
    of complex weighted networks."

    Args:
        g: Weighted igraph Graph.
        alpha: Significance level (lower = more aggressive pruning).

    Returns:
        A new Graph with only significant edges retained.
    """
    import igraph as ig

    if g.ecount() == 0:
        return g.copy()

    weights = np.array(g.es["weight"], dtype=np.float64)

    # Compute node strengths (sum of edge weights per node)
    strengths = np.zeros(g.vcount())
    for eid, edge in enumerate(g.es):
        strengths[edge.source] += weights[eid]
        strengths[edge.target] += weights[eid]

    # Test each edge for significance from both endpoints
    keep = []
    for eid, edge in enumerate(g.es):
        w = weights[eid]
        significant = False

        for node in (edge.source, edge.target):
            k = g.degree(node)
            if k <= 1:
                # Single-edge nodes always keep their edge
                significant = True
                break
            s = strengths[node]
            if s == 0:
                continue
            p_ij = w / s  # fraction of node's weight on this edge
            # Probability under null (uniform weight distribution)
            alpha_ij = (1.0 - p_ij) ** (k - 1)
            if alpha_ij < alpha:
                significant = True
                break

        if significant:
            keep.append(eid)

    # Build filtered graph
    g_filtered = ig.Graph(n=g.vcount())
    # Copy vertex attributes
    for attr in g.vs.attributes():
        g_filtered.vs[attr] = g.vs[attr]

    if keep:
        kept_edges = [(g.es[eid].source, g.es[eid].target) for eid in keep]
        kept_weights = [weights[eid] for eid in keep]
        g_filtered.add_edges(kept_edges)
        g_filtered.es["weight"] = kept_weights

    return g_filtered


def compute_participation_coefficients(
    g: "ig.Graph", membership: list[int]
) -> list[dict]:
    """Compute participation coefficient for each node.

    The participation coefficient measures how evenly a node's edges are
    distributed across communities. High P = bridge/outlier node with
    edges spread across many domains.

    Reference: Guimera & Amaral (2005) "Functional cartography of complex
    metabolic networks."

    Args:
        g: igraph Graph.
        membership: Community assignment for each node.

    Returns:
        List of dicts with node_idx, cluster_id, participation, internal_frac,
        sorted by participation descending.
    """
    results = []
    for i in range(g.vcount()):
        neighbors = g.neighbors(i)
        k_i = len(neighbors)
        if k_i == 0:
            results.append({
                "node_idx": i,
                "cluster_id": g.vs[i]["cluster_id"] if "cluster_id" in g.vs.attributes() else i,
                "participation": 0.0,
                "internal_frac": 1.0,
            })
            continue

        # Count edges to each community
        community_counts = defaultdict(int)
        for j in neighbors:
            community_counts[membership[j]] += 1

        # Participation coefficient: P_i = 1 - sum((k_ic / k_i)^2)
        p_i = 1.0 - sum((count / k_i) ** 2 for count in community_counts.values())

        # Internal fraction: edges to own community / total edges
        own_community = membership[i]
        internal = community_counts.get(own_community, 0)
        internal_frac = internal / k_i

        results.append({
            "node_idx": i,
            "cluster_id": g.vs[i]["cluster_id"] if "cluster_id" in g.vs.attributes() else i,
            "participation": p_i,
            "internal_frac": internal_frac,
        })

    results.sort(key=lambda x: x["participation"], reverse=True)
    return results


def get_cluster_topics(cursor: sqlite3.Cursor, cluster_id: int) -> list[dict]:
    """Get topics for a cluster with their embeddings and occurrence counts."""
    cursor.execute(
        """
        SELECT id, label, embedding, occurrence_count
        FROM topics
        WHERE cluster_id = ? AND embedding IS NOT NULL
        ORDER BY occurrence_count DESC
    """,
        (cluster_id,),
    )

    topics = []
    for row in cursor.fetchall():
        if row["embedding"]:
            topics.append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "embedding": np.frombuffer(row["embedding"], dtype=np.float32),
                    "occurrence_count": row["occurrence_count"] or 1,
                }
            )
    return topics


def compute_robust_centroid(
    topics: list[dict],
    top_n: int = DEFAULT_TOP_N_TOPICS,
    min_label_length: int = DEFAULT_MIN_LABEL_LENGTH,
) -> np.ndarray | None:
    """
    Compute a robust centroid for a cluster.

    Robust centroid approach:
    - Filter topics with labels < min_label_length chars (junk like "a", "the")
    - Take top N topics by occurrence count
    - Weight by log1p(occurrence_count) for stable centroids
    """
    # Filter short labels
    topics = [t for t in topics if len(t["label"]) >= min_label_length]

    # Take top N by occurrence
    topics = sorted(topics, key=lambda t: t["occurrence_count"], reverse=True)[:top_n]

    # Need minimum topics for a meaningful centroid
    if len(topics) < 3:
        return None

    embeddings = np.array([t["embedding"] for t in topics])
    weights = np.array([np.log1p(t["occurrence_count"]) for t in topics])
    weights = weights / weights.sum()  # Normalize

    centroid = np.average(embeddings, axis=0, weights=weights)
    # L2 normalize so centroid is unit-norm (consistent with cosine similarity)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def generate_super_clusters(
    n_domains: int = DEFAULT_N_DOMAINS,
    top_n_topics: int = DEFAULT_TOP_N_TOPICS,
    min_label_length: int = DEFAULT_MIN_LABEL_LENGTH,
    db_path: Path | None = None,
) -> list[dict]:
    """
    Generate super-clusters by grouping Leiden clusters using K-means.

    Args:
        n_domains: Number of super-clusters (domains) to create
        top_n_topics: Number of top topics to use for centroid computation
        min_label_length: Minimum label length to include in centroid
        db_path: Path to database (defaults to IPAD_DB_PATH)

    Returns:
        List of super-cluster dicts with leiden_clusters and auto_label
    """
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

    # Compute robust centroids
    cluster_ids = []
    centroids = []

    for row in clusters:
        cluster_id = row["cluster_id"]
        topics = get_cluster_topics(cursor, cluster_id)
        centroid = compute_robust_centroid(topics, top_n_topics, min_label_length)

        if centroid is not None:
            cluster_ids.append(cluster_id)
            centroids.append(centroid)

    if len(centroids) < n_domains:
        n_domains = len(centroids)

    # K-means clustering
    X = np.array(centroids)
    kmeans = KMeans(n_clusters=n_domains, random_state=42, n_init=10)
    super_labels = kmeans.fit_predict(X)

    # Build super-cluster mapping
    super_clusters = {}
    for cluster_id, super_id in zip(cluster_ids, super_labels):
        super_id = int(super_id)
        if super_id not in super_clusters:
            super_clusters[super_id] = {
                "super_cluster_id": super_id,
                "leiden_clusters": [],
                "top_topics": [],
            }

        # Get top topics for this Leiden cluster
        cursor.execute(
            """
            SELECT label, occurrence_count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
            LIMIT 3
        """,
            (cluster_id,),
        )
        top = [{"label": r["label"], "count": r["occurrence_count"]} for r in cursor.fetchall()]

        super_clusters[super_id]["leiden_clusters"].append(
            {"cluster_id": cluster_id, "top_topics": top}
        )

    # Generate auto-labels for each super-cluster
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
        filtered = [(k, v) for k, v in all_topics.items() if len(k) >= min_label_length]
        sorted_topics = sorted(filtered, key=lambda x: x[1], reverse=True)[:10]
        data["top_topics"] = [{"label": t[0], "total_count": t[1]} for t in sorted_topics]

        # Auto-generate a label from top 3 topics
        if sorted_topics:
            data["auto_label"] = " / ".join([t[0] for t in sorted_topics[:3]])
        else:
            data["auto_label"] = f"Domain {super_id}"

    conn.close()

    # Sort by number of Leiden clusters
    return sorted(super_clusters.values(), key=lambda x: len(x["leiden_clusters"]), reverse=True)


def apply_domain_labels(super_clusters: list[dict], label_mapping: dict[int, str]) -> list[dict]:
    """
    Apply human-refined labels to super-clusters, creating final domains.

    Args:
        super_clusters: Output from generate_super_clusters()
        label_mapping: Dict mapping super_cluster_id -> domain label

    Returns:
        List of domain dicts ready for database loading
    """
    # Build final domains, merging where labels match
    domains = {}

    for sc in super_clusters:
        old_id = sc["super_cluster_id"]
        if old_id not in label_mapping:
            label = sc["auto_label"]  # Fallback to auto-label
        else:
            label = label_mapping[old_id]

        if label not in domains:
            domains[label] = {
                "label": label,
                "original_ids": [],
                "leiden_clusters": [],
                "top_topics": {},
            }

        domains[label]["original_ids"].append(old_id)
        domains[label]["leiden_clusters"].extend(sc["leiden_clusters"])

        # Aggregate top topics
        for t in sc["top_topics"]:
            topic_label = t["label"]
            count = t["total_count"]
            if topic_label not in domains[label]["top_topics"]:
                domains[label]["top_topics"][topic_label] = 0
            domains[label]["top_topics"][topic_label] += count

    # Convert to list and assign new sequential IDs
    result = []
    for i, (label, data) in enumerate(
        sorted(domains.items(), key=lambda x: len(x[1]["leiden_clusters"]), reverse=True)
    ):
        # Sort topics by count and take top 10
        sorted_topics = sorted(data["top_topics"].items(), key=lambda x: x[1], reverse=True)[:10]

        result.append(
            {
                "domain_id": i,
                "label": label,
                "cluster_count": len(data["leiden_clusters"]),
                "original_super_ids": data["original_ids"],
                "leiden_cluster_ids": [lc["cluster_id"] for lc in data["leiden_clusters"]],
                "top_topics": [{"label": t[0], "count": t[1]} for t in sorted_topics],
            }
        )

    return result


def save_domains_json(domains: list[dict], output_path: Path) -> None:
    """Save domains to JSON file."""
    with open(output_path, "w") as f:
        json.dump(domains, f, indent=2)


def regenerate_domains(
    n_domains: int = DEFAULT_N_DOMAINS,
    label_mapping: dict[int, str] | None = None,
    output_path: Path | None = None,
) -> list[dict]:
    """
    Full pipeline to regenerate domains from current Leiden clusters.

    Args:
        n_domains: Number of super-clusters to generate
        label_mapping: Optional mapping of super_cluster_id -> domain label
        output_path: Optional path to save JSON output

    Returns:
        List of domain dicts
    """
    # Step 1: Generate super-clusters
    super_clusters = generate_super_clusters(n_domains=n_domains)

    # Step 2: Apply labels (or use auto-labels if no mapping)
    if label_mapping:
        domains = apply_domain_labels(super_clusters, label_mapping)
    else:
        # Use auto-labels as domains
        domains = []
        for i, sc in enumerate(super_clusters):
            domains.append(
                {
                    "domain_id": i,
                    "label": sc["auto_label"],
                    "cluster_count": len(sc["leiden_clusters"]),
                    "original_super_ids": [sc["super_cluster_id"]],
                    "leiden_cluster_ids": [lc["cluster_id"] for lc in sc["leiden_clusters"]],
                    "top_topics": sc["top_topics"],
                }
            )

    # Step 3: Save if output path provided
    if output_path:
        save_domains_json(domains, output_path)

    return domains


def split_catchall_superclusters(
    super_clusters: list[dict],
    catchall_splits: dict[int, int],
    db_path: Path | None = None,
) -> list[dict]:
    """
    Sub-cluster specific super-clusters that are catch-alls.

    Args:
        super_clusters: Output from generate_super_clusters()
        catchall_splits: Dict mapping super_cluster_id -> number of sub-clusters
                        e.g., {24: 4, 29: 3} to split cluster 24 into 4 and 29 into 3
        db_path: Path to database (defaults to IPAD_DB_PATH)

    Returns:
        Updated list of super-clusters with catch-alls replaced by sub-clusters
    """
    db_path = db_path or IPAD_DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Find the next available super-cluster ID
    max_id = max(sc["super_cluster_id"] for sc in super_clusters)
    next_id = max_id + 1

    new_super_clusters = []

    for sc in super_clusters:
        sc_id = sc["super_cluster_id"]

        if sc_id not in catchall_splits:
            # Keep this super-cluster as-is
            new_super_clusters.append(sc)
            continue

        # This is a catch-all - split it
        n_splits = catchall_splits[sc_id]
        leiden_clusters = sc["leiden_clusters"]

        if len(leiden_clusters) < n_splits:
            # Not enough clusters to split
            new_super_clusters.append(sc)
            continue

        print(
            f"Splitting super-cluster {sc_id} ({len(leiden_clusters)} clusters) into {n_splits} sub-groups..."
        )

        # Get centroids for each Leiden cluster
        centroids = []
        valid_clusters = []

        for lc in leiden_clusters:
            cluster_id = lc["cluster_id"]
            topics = get_cluster_topics(cursor, cluster_id)
            centroid = compute_robust_centroid(topics)

            if centroid is not None:
                centroids.append(centroid)
                valid_clusters.append(lc)

        if len(centroids) < n_splits:
            print(f"  Warning: Only {len(centroids)} valid centroids, keeping original")
            new_super_clusters.append(sc)
            continue

        # K-means sub-clustering
        X = np.array(centroids)
        kmeans = KMeans(n_clusters=n_splits, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(X)

        # Group clusters by sub-label
        sub_groups = {}
        for lc, sub_label in zip(valid_clusters, sub_labels):
            sub_label = int(sub_label)
            if sub_label not in sub_groups:
                sub_groups[sub_label] = []
            sub_groups[sub_label].append(lc)

        # Create new super-clusters for each sub-group
        for sub_label, group_clusters in sub_groups.items():
            new_sc_id = next_id
            next_id += 1

            # Aggregate top topics
            all_topics = {}
            for lc in group_clusters:
                for t in lc["top_topics"]:
                    label = t["label"]
                    count = t["count"]
                    if label not in all_topics:
                        all_topics[label] = 0
                    all_topics[label] += count

            # Filter and sort topics
            filtered = [(k, v) for k, v in all_topics.items() if len(k) >= 4]
            sorted_topics = sorted(filtered, key=lambda x: x[1], reverse=True)[:10]

            # Auto-label from top 3
            if sorted_topics:
                auto_label = " / ".join([t[0] for t in sorted_topics[:3]])
            else:
                auto_label = f"Sub-domain {new_sc_id}"

            new_super_clusters.append(
                {
                    "super_cluster_id": new_sc_id,
                    "leiden_clusters": group_clusters,
                    "top_topics": [{"label": t[0], "total_count": t[1]} for t in sorted_topics],
                    "auto_label": auto_label,
                    "split_from": sc_id,
                }
            )

            print(
                f"  → Sub-cluster {new_sc_id}: {len(group_clusters)} clusters - {auto_label[:50]}"
            )

    conn.close()

    # Sort by number of clusters (descending)
    new_super_clusters.sort(key=lambda x: len(x["leiden_clusters"]), reverse=True)

    return new_super_clusters


def build_cluster_graph(
    k: int = 8,
    min_similarity: float = 0.3,
    backbone_alpha: float | None = None,
    db_path: Path | None = None,
) -> "ig.Graph":
    """Build a k-NN cosine similarity graph between Leiden cluster centroids.

    Each node is a Leiden cluster. Edges connect clusters whose centroids
    are among each other's k nearest neighbors with similarity >= min_similarity.

    If backbone_alpha is set, applies the disparity filter to prune
    statistically insignificant edges after k-NN construction.

    Args:
        k: Number of nearest neighbors per cluster.
        min_similarity: Minimum cosine similarity to include an edge.
        backbone_alpha: Disparity filter significance level (None = no filter).
        db_path: Path to database (defaults to IPAD_DB_PATH).

    Returns:
        igraph Graph with cluster_id vertex attribute and similarity edge weights.
    """
    import igraph as ig
    from sklearn.neighbors import NearestNeighbors

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

    # Compute robust centroids
    cluster_ids = []
    centroids = []
    for row in clusters:
        cluster_id = row["cluster_id"]
        topics = get_cluster_topics(cursor, cluster_id)
        centroid = compute_robust_centroid(topics)
        if centroid is not None:
            cluster_ids.append(cluster_id)
            centroids.append(centroid)

    conn.close()

    if len(centroids) < 2:
        g = ig.Graph(n=len(centroids))
        if cluster_ids:
            g.vs["cluster_id"] = cluster_ids
        return g

    X = np.array(centroids)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    # k-NN in cosine space
    effective_k = min(k + 1, len(centroids))  # +1 because self is included
    nn = NearestNeighbors(n_neighbors=effective_k, metric="cosine")
    nn.fit(X_norm)
    distances, indices = nn.kneighbors(X_norm)

    # Build graph
    g = ig.Graph(n=len(cluster_ids))
    g.vs["cluster_id"] = cluster_ids

    edges = set()
    edge_weights = {}
    for i in range(len(cluster_ids)):
        for j_pos in range(1, effective_k):  # skip self at position 0
            j = indices[i][j_pos]
            similarity = 1.0 - distances[i][j_pos]
            if similarity >= min_similarity:
                edge = (min(i, j), max(i, j))
                if edge not in edges:
                    edges.add(edge)
                    edge_weights[edge] = similarity

    if edges:
        edge_list = list(edges)
        g.add_edges(edge_list)
        g.es["weight"] = [edge_weights[e] for e in edge_list]

    # Apply disparity filter if requested
    if backbone_alpha is not None and g.ecount() > 0:
        before = g.ecount()
        g = disparity_filter(g, alpha=backbone_alpha)
        print(f"Disparity filter (alpha={backbone_alpha}): {before} → {g.ecount()} edges")

    return g


def generate_super_clusters_leiden(
    resolution: float | None = None,
    sweep: bool = False,
    k: int = 8,
    min_similarity: float = 0.3,
    backbone_alpha: float | None = None,
    remove_outliers: bool = False,
    outlier_threshold: float = 0.7,
    auto_select: bool = False,
    db_path: Path | None = None,
) -> dict:
    """Generate super-clusters using Leiden CPM on a cluster-level graph.

    Alternative to K-means that uses graph community detection, with
    optional multi-resolution sweep to find optimal resolution.

    Args:
        resolution: CPM resolution (required if sweep=False).
        sweep: If True, run multi-resolution sweep.
        k: Number of nearest neighbors for cluster graph.
        min_similarity: Minimum cosine similarity for cluster graph edges.
        backbone_alpha: Disparity filter significance level (None = no filter).
        remove_outliers: If True, reassign high-participation outlier clusters.
        outlier_threshold: Participation coefficient above which a node is an outlier.
        auto_select: If True during sweep, apply recommended resolution.
        db_path: Path to database.

    Returns:
        Dict with super_clusters list and optional sweep_summary.
    """
    import leidenalg

    from .config import DOMAIN_SWEEP_RESOLUTION_RANGE, SWEEP_SEED

    db_path = db_path or IPAD_DB_PATH

    # Build cluster-level graph
    print(f"Building cluster graph (k={k}, min_similarity={min_similarity})...")
    g = build_cluster_graph(
        k=k, min_similarity=min_similarity, backbone_alpha=backbone_alpha, db_path=db_path
    )
    print(f"Cluster graph: {g.vcount()} nodes, {g.ecount()} edges")

    if g.vcount() == 0:
        return {"error": "No clusters found", "super_clusters": []}

    result = {}

    if sweep:
        from .sweep import format_sweep_table, run_sweep, log_spaced_resolutions

        resolutions = log_spaced_resolutions(
            low=DOMAIN_SWEEP_RESOLUTION_RANGE[0],
            high=DOMAIN_SWEEP_RESOLUTION_RANGE[1],
        )
        summary = run_sweep(g, resolutions=resolutions, seed=SWEEP_SEED)
        result["sweep_summary"] = summary

        if auto_select and summary.recommended_resolution is not None:
            resolution = summary.recommended_resolution
            print(f"Auto-selected resolution: {resolution:.6f}")
        elif summary.recommended_resolution is not None:
            print(f"Recommended resolution: {summary.recommended_resolution:.6f}")
            print("Use --auto-select to apply, or --resolution to set manually.")
            result["super_clusters"] = []
            return result
        else:
            print("No stable resolution found in sweep.")
            result["super_clusters"] = []
            return result

    if resolution is None:
        resolution = 0.01  # sensible default for cluster-level graph

    # Run Leiden on cluster graph
    weights = g.es["weight"] if g.ecount() > 0 and "weight" in g.es.attributes() else None
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=SWEEP_SEED,
    )

    membership = list(partition.membership)
    n_super = len(set(membership))
    print(f"Leiden produced {n_super} super-clusters at resolution {resolution:.6f}")

    # Outlier detection and reassignment
    outliers_reassigned = 0
    if remove_outliers and g.vcount() > 0:
        coeffs = compute_participation_coefficients(g, membership)
        for info in coeffs:
            if (
                info["participation"] > outlier_threshold
                and info["internal_frac"] < 0.3
            ):
                node_idx = info["node_idx"]
                # Reassign to strongest-connected domain (most edge weight)
                neighbor_weights = defaultdict(float)
                for j in g.neighbors(node_idx):
                    eid = g.get_eid(node_idx, j)
                    w = g.es[eid]["weight"] if "weight" in g.es.attributes() else 1.0
                    neighbor_weights[membership[j]] += w

                if neighbor_weights:
                    best_community = max(neighbor_weights, key=neighbor_weights.get)
                    membership[node_idx] = best_community
                    outliers_reassigned += 1

        if outliers_reassigned > 0:
            n_super = len(set(membership))
            print(
                f"Outlier reassignment: {outliers_reassigned} clusters reassigned "
                f"(threshold={outlier_threshold}), now {n_super} super-clusters"
            )

    result["outliers_reassigned"] = outliers_reassigned

    # Build super-cluster structures (same format as K-means output)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    super_clusters = {}
    for node_idx, super_id in enumerate(membership):
        super_id = int(super_id)
        cluster_id = g.vs[node_idx]["cluster_id"]

        if super_id not in super_clusters:
            super_clusters[super_id] = {
                "super_cluster_id": super_id,
                "leiden_clusters": [],
                "top_topics": [],
            }

        # Get top topics for this Leiden cluster
        cursor.execute(
            """
            SELECT label, occurrence_count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
            LIMIT 3
        """,
            (cluster_id,),
        )
        top = [{"label": r["label"], "count": r["occurrence_count"]} for r in cursor.fetchall()]

        super_clusters[super_id]["leiden_clusters"].append(
            {"cluster_id": cluster_id, "top_topics": top}
        )

    # Generate auto-labels
    for super_id, data in super_clusters.items():
        all_topics = {}
        for lc in data["leiden_clusters"]:
            for t in lc["top_topics"]:
                label = t["label"]
                if label not in all_topics:
                    all_topics[label] = 0
                all_topics[label] += t["count"]

        filtered = [(k, v) for k, v in all_topics.items() if len(k) >= DEFAULT_MIN_LABEL_LENGTH]
        sorted_topics = sorted(filtered, key=lambda x: x[1], reverse=True)[:10]
        data["top_topics"] = [{"label": t[0], "total_count": t[1]} for t in sorted_topics]

        if sorted_topics:
            data["auto_label"] = " / ".join([t[0] for t in sorted_topics[:3]])
        else:
            data["auto_label"] = f"Domain {super_id}"

    conn.close()

    result["super_clusters"] = sorted(
        super_clusters.values(),
        key=lambda x: len(x["leiden_clusters"]),
        reverse=True,
    )
    result["resolution"] = resolution
    result["n_super_clusters"] = n_super

    return result
