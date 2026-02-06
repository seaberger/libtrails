"""Leiden clustering for hierarchical topic organization."""

import sys
from collections import defaultdict
from typing import Optional

import igraph as ig
import leidenalg

from .config import (
    CLUSTER_KNN_K,
    CLUSTER_MODE,
    CLUSTER_PARTITION_TYPE,
    CLUSTER_RESOLUTION,
    COOCCURRENCE_MIN_COUNT,
    EMBEDDING_EDGE_THRESHOLD,
    PMI_MIN_THRESHOLD,
)
from .database import get_db, update_topic_cluster


def _log_memory(label: str) -> float:
    """Log current memory usage for debugging OOM issues.

    Returns:
        Memory usage in MB, or 0 if not available.
    """
    try:
        # Get memory in MB (on macOS this is in bytes, on Linux it's in KB)
        import platform
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if platform.system() == "Darwin":
            mem_mb = usage / (1024 * 1024)
        else:
            mem_mb = usage / 1024
        print(f"  [{label}] Memory: {mem_mb:.0f} MB", file=sys.stderr)
        return mem_mb
    except Exception:
        return 0


def diagnose_hubs(g: ig.Graph, top_n: int = 50) -> dict:
    """
    Analyze hub topics to understand what's creating transitivity problems.

    Args:
        g: The topic graph
        top_n: Number of top hubs to show

    Returns:
        Dictionary with degree distribution stats and top hubs
    """
    import numpy as np

    degrees = [(v.index, v["label"], v.degree()) for v in g.vs]
    degrees.sort(key=lambda x: x[2], reverse=True)

    all_degrees = [d[2] for d in degrees]

    stats = {
        "min": int(min(all_degrees)),
        "median": int(np.median(all_degrees)),
        "mean": float(np.mean(all_degrees)),
        "p95": int(np.percentile(all_degrees, 95)),
        "p99": int(np.percentile(all_degrees, 99)),
        "max": int(max(all_degrees)),
    }

    top_hubs = [{"label": label, "degree": deg} for idx, label, deg in degrees[:top_n]]

    return {
        "degree_stats": stats,
        "top_hubs": top_hubs,
        "total_nodes": g.vcount(),
        "total_edges": g.ecount(),
    }


def identify_hub_topics(
    g: ig.Graph,
    method: str = "degree",
    percentile: float = 95,
    generic_patterns: set = None,
) -> set[int]:
    """
    Identify hub topics that should be excluded from clustering.

    Args:
        g: The topic graph
        method: "degree" (by connection count), "generic" (by term patterns), or "both"
        percentile: Percentile threshold for degree-based hub detection
        generic_patterns: Set of generic term patterns to match (for "generic" or "both")

    Returns:
        Set of node indices identified as hubs
    """
    import numpy as np

    hub_indices = set()

    if method in ("degree", "both"):
        degrees = g.degree()
        threshold = np.percentile(degrees, percentile)

        for i, d in enumerate(degrees):
            if d > threshold:
                hub_indices.add(i)
        print(
            f"  Degree hubs (>{threshold:.0f} edges, top {100 - percentile}%): {len(hub_indices)}"
        )

    if method in ("generic", "both"):
        if generic_patterns is None:
            # Default generic terms that create false connections
            generic_patterns = {
                "relationships",
                "technology",
                "society",
                "culture",
                "history",
                "nature",
                "life",
                "death",
                "love",
                "war",
                "family",
                "work",
                "home",
                "money",
                "power",
                "time",
                "people",
                "world",
                "change",
                "future",
                "past",
                "furniture",
                "food",
                "travel",
                "art",
                "music",
                "communication",
                "education",
                "health",
                "science",
                "problem",
                "solution",
                "idea",
                "concept",
                "experience",
            }

        generic_count = 0
        for v in g.vs:
            label = v["label"].lower() if "label" in v.attributes() else ""
            if label in generic_patterns:
                hub_indices.add(v.index)
                generic_count += 1
        print(f"  Generic term hubs: {generic_count}")

    return hub_indices


def cluster_without_hubs(
    g: ig.Graph,
    hub_indices: set[int],
    partition_type: str = "cpm",
    resolution: float = 0.005,
) -> tuple[dict, leidenalg.VertexPartition]:
    """
    Cluster with hub topics removed, then assign hubs to nearest cluster.

    Args:
        g: The full topic graph
        hub_indices: Set of node indices to exclude from clustering
        partition_type: Leiden partition type
        resolution: Resolution parameter

    Returns:
        Tuple of (assignments dict mapping topic_id -> cluster_id, partition object)
    """
    from collections import Counter

    # Create subgraph without hubs
    non_hub_indices = [i for i in range(g.vcount()) if i not in hub_indices]
    subgraph = g.subgraph(non_hub_indices)

    # Map between subgraph and original indices
    orig_to_sub = {orig: sub for sub, orig in enumerate(non_hub_indices)}
    sub_to_orig = {sub: orig for orig, sub in orig_to_sub.items()}

    print(f"  Clustering {subgraph.vcount()} non-hub topics (excluded {len(hub_indices)} hubs)...")
    print(f"  Subgraph edges: {subgraph.ecount()} (original: {g.ecount()})")

    # Cluster the non-hub subgraph
    partition = _get_partition(subgraph, partition_type, resolution)

    # Assign clusters to non-hub topics
    assignments = {}
    for sub_idx, cluster_id in enumerate(partition.membership):
        orig_idx = sub_to_orig[sub_idx]
        topic_id = g.vs[orig_idx]["topic_id"]
        assignments[topic_id] = cluster_id

    # Assign each hub to its most common neighbor cluster (or -1 for "unclustered")
    hubs_assigned = 0
    hubs_orphaned = 0

    for hub_idx in hub_indices:
        neighbor_clusters = []
        for neighbor_idx in g.neighbors(hub_idx):
            if neighbor_idx in orig_to_sub:
                sub_idx = orig_to_sub[neighbor_idx]
                neighbor_clusters.append(partition.membership[sub_idx])

        hub_topic_id = g.vs[hub_idx]["topic_id"]

        if neighbor_clusters:
            # Assign to most common neighboring cluster
            most_common = Counter(neighbor_clusters).most_common(1)[0][0]
            assignments[hub_topic_id] = most_common
            hubs_assigned += 1
        else:
            # Hub has no non-hub neighbors; mark as unclustered
            assignments[hub_topic_id] = -1
            hubs_orphaned += 1

    print(f"  Assigned {hubs_assigned} hubs to neighboring clusters, {hubs_orphaned} orphaned")

    return assignments, partition


def assign_hubs_multi_cluster(
    g: ig.Graph,
    hub_indices: set[int],
    primary_assignments: dict,
    min_connections: int = 3,
    max_clusters: int = 5,
) -> dict:
    """
    Assign hub topics to multiple relevant clusters with strength scores.

    Args:
        g: The full topic graph
        hub_indices: Set of node indices that are hubs
        primary_assignments: dict mapping topic_id -> primary cluster_id
        min_connections: Minimum connection count to include a cluster
        max_clusters: Maximum clusters per hub

    Returns:
        dict mapping topic_id -> list of (cluster_id, strength) tuples
    """
    from collections import Counter

    hub_multi_assignments = {}

    for hub_idx in hub_indices:
        topic_id = g.vs[hub_idx]["topic_id"]

        # Count weighted connections to each cluster
        cluster_connections = Counter()
        for neighbor_idx in g.neighbors(hub_idx):
            neighbor_topic = g.vs[neighbor_idx]["topic_id"]
            if neighbor_topic in primary_assignments:
                cluster_id = primary_assignments[neighbor_topic]
                if cluster_id >= 0:  # Skip unclustered
                    # Weight by edge weight if available
                    try:
                        edge_id = g.get_eid(hub_idx, neighbor_idx)
                        weight = g.es[edge_id]["weight"] if "weight" in g.es.attributes() else 1.0
                    except Exception:
                        weight = 1.0
                    cluster_connections[cluster_id] += weight

        if not cluster_connections:
            continue

        # Normalize to proportions
        total_weight = sum(cluster_connections.values())
        significant_clusters = [
            (cid, count / total_weight)
            for cid, count in cluster_connections.most_common(max_clusters)
            if count >= min_connections
        ]

        if significant_clusters:
            hub_multi_assignments[topic_id] = significant_clusters

    return hub_multi_assignments


def save_multi_cluster_assignments(
    assignments: dict,
    primary_assignments: dict,
) -> int:
    """
    Save multi-cluster hub assignments to database.

    Args:
        assignments: dict mapping topic_id -> list of (cluster_id, strength)
        primary_assignments: dict mapping topic_id -> primary cluster_id

    Returns:
        Number of memberships saved
    """
    from .database import get_db

    count = 0
    with get_db() as conn:
        cursor = conn.cursor()

        # Clear existing multi-cluster assignments
        cursor.execute("DELETE FROM topic_cluster_memberships")

        for topic_id, cluster_list in assignments.items():
            primary_cluster = primary_assignments.get(topic_id, -1)

            for cluster_id, strength in cluster_list:
                is_primary = cluster_id == primary_cluster
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO topic_cluster_memberships
                    (topic_id, cluster_id, strength, is_primary)
                    VALUES (?, ?, ?, ?)
                """,
                    (topic_id, cluster_id, strength, is_primary),
                )
                count += 1

        conn.commit()

    return count


def _get_partition(
    g: ig.Graph,
    partition_type: str,
    resolution: float,
) -> leidenalg.VertexPartition:
    """Get Leiden partition with appropriate settings.

    Args:
        g: The igraph Graph to cluster
        partition_type: One of "modularity", "surprise", or "cpm"
        resolution: Resolution parameter (used for modularity and cpm)

    Returns:
        A leidenalg VertexPartition
    """
    weights = g.es["weight"] if "weight" in g.es.attributes() else None

    if weights:
        print(f"  Using edge weights (min={min(weights):.3f}, max={max(weights):.3f})")

    if partition_type == "modularity":
        # Use RB version for resolution tuning (γ=1.0 gives standard modularity)
        return leidenalg.find_partition(
            g,
            leidenalg.RBConfigurationVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
        )

    elif partition_type == "surprise":
        # Surprise doesn't use resolution but does use weights
        return leidenalg.find_partition(
            g,
            leidenalg.SurpriseVertexPartition,
            weights=weights,
        )

    elif partition_type == "cpm":
        return leidenalg.find_partition(
            g,
            leidenalg.CPMVertexPartition,
            weights=weights,
            resolution_parameter=resolution,
        )

    else:
        raise ValueError(f"Unknown partition type: {partition_type}")


def cluster_topics(
    min_cluster_size: int = 3,
    resolution: float = None,
    mode: str = None,
    partition_type: str = None,
    cooccurrence_min: int = None,
    knn_k: int = None,
    dry_run: bool = False,
    sample_size: int = None,
    remove_hubs: bool = False,
    hub_percentile: float = 95,
    hub_method: str = "degree",
    progress_file: str | None = None,
) -> dict:
    """
    Cluster topics using the Leiden algorithm.

    Args:
        min_cluster_size: Minimum size for a cluster
        resolution: Resolution parameter for CPM/modularity (higher = more clusters)
        mode: Graph construction mode - "cooccurrence", "knn", or "full"
        partition_type: Leiden partition type - "modularity", "surprise", or "cpm"
        cooccurrence_min: Minimum co-occurrence count (overrides config if set)
        knn_k: Number of nearest neighbors for k-NN mode
        dry_run: If True, don't save results to database
        sample_size: If set, only cluster this many topics (for testing)
        remove_hubs: If True, exclude hub topics from clustering then assign post-hoc
        hub_percentile: Percentile threshold for hub detection (default 95 = top 5%)
        hub_method: Hub detection method - "degree", "generic", or "both"

    Returns:
        Statistics about the clustering
    """
    import time
    from datetime import datetime

    from .topic_graph import build_topic_graph, build_topic_graph_cooccurrence_only

    def log_progress(msg: str):
        """Write progress message to stdout and optionally to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)
        if progress_file:
            with open(progress_file, "w") as f:
                f.write(f"{full_msg}\n")

    # Apply config defaults for None parameters
    if mode is None:
        mode = CLUSTER_MODE
    if partition_type is None:
        partition_type = CLUSTER_PARTITION_TYPE
    if resolution is None:
        resolution = CLUSTER_RESOLUTION
    if knn_k is None:
        knn_k = CLUSTER_KNN_K

    # Determine co-occurrence threshold
    min_cooccur = cooccurrence_min if cooccurrence_min is not None else COOCCURRENCE_MIN_COUNT

    mem_before = _log_memory("Before build_topic_graph")
    log_progress(f"Building topic graph (mode={mode})...")
    graph_start = time.time()

    # Build graph based on mode
    if mode == "cooccurrence":
        g = build_topic_graph_cooccurrence_only(
            cooccurrence_min=min_cooccur,
            pmi_min=PMI_MIN_THRESHOLD,
        )
    elif mode == "full":
        g = build_topic_graph(
            embedding_threshold=EMBEDDING_EDGE_THRESHOLD,
            cooccurrence_min=min_cooccur,
            pmi_min=PMI_MIN_THRESHOLD,
        )
    elif mode == "knn":
        # k-NN mode: co-occurrence + k-nearest neighbor embedding edges
        from .topic_graph import build_topic_graph_knn

        g = build_topic_graph_knn(
            cooccurrence_min=min_cooccur,
            pmi_min=PMI_MIN_THRESHOLD,
            k=knn_k,
        )
    else:
        return {"error": f"Unknown mode: {mode}. Use 'cooccurrence', 'knn', or 'full'"}

    graph_elapsed = time.time() - graph_start
    log_progress(f"Graph built in {graph_elapsed:.1f}s: {g.vcount()} nodes, {g.ecount()} edges")
    _log_memory(f"After build_topic_graph: {g.vcount()} nodes, {g.ecount()} edges")

    if g.vcount() == 0:
        return {"error": "No topics in graph"}

    # Sample subset if requested (for dry-run testing)
    full_node_count = g.vcount()
    full_edge_count = g.ecount()

    if sample_size and sample_size < g.vcount():
        print(f"  [DRY RUN] Sampling {sample_size} of {g.vcount()} topics...")
        import random

        random.seed(42)  # Reproducible sampling
        sample_nodes = random.sample(range(g.vcount()), sample_size)
        g = g.subgraph(sample_nodes)
        print(f"  [DRY RUN] Sampled graph: {g.vcount()} nodes, {g.ecount()} edges")

    # Log graph statistics
    edge_type_counts = {}
    if g.ecount() > 0:
        for t in g.es["type"]:
            edge_type_counts[t] = edge_type_counts.get(t, 0) + 1

    log_progress(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")
    for edge_type, count in edge_type_counts.items():
        print(f"    - {edge_type}: {count}", flush=True)

    if partition_type not in ["modularity", "surprise", "cpm"]:
        return {
            "error": f"Unknown partition type: {partition_type}. Use 'modularity', 'surprise', or 'cpm'"
        }

    log_progress(f"Running Leiden ({partition_type}, resolution={resolution}) on graph...")
    _log_memory("Before Leiden")

    # Run Leiden clustering
    import time

    start_time = time.time()
    hub_indices = None  # Initialize for scope

    if remove_hubs:
        # Hub removal approach: cluster without hubs, then assign hubs post-hoc
        print(
            f"\n  [HUB REMOVAL] Identifying hubs (method={hub_method}, percentile={hub_percentile})..."
        )
        hub_indices = identify_hub_topics(g, method=hub_method, percentile=hub_percentile)

        assignments, partition = cluster_without_hubs(
            g, hub_indices, partition_type=partition_type, resolution=resolution
        )

        # Build membership list for result calculation
        membership = [assignments.get(g.vs[i]["topic_id"], -1) for i in range(g.vcount())]
    else:
        # Standard clustering
        partition = _get_partition(g, partition_type, resolution)
        membership = partition.membership
        assignments = None

    elapsed = time.time() - start_time
    mem_after_leiden = _log_memory("After Leiden")
    log_progress(f"Leiden completed in {elapsed:.1f}s")

    # Calculate cluster sizes
    cluster_sizes = defaultdict(int)
    for cluster_id in membership:
        if cluster_id >= 0:  # Skip unclustered (-1)
            cluster_sizes[cluster_id] += 1

    # Build result
    result = {
        "total_topics": g.vcount(),
        "total_edges": g.ecount(),
        "edge_types": edge_type_counts,
        "num_clusters": len(set(c for c in membership if c >= 0)),
        "modularity": partition.quality(),
        "partition_type": partition_type,
        "resolution": resolution,
        "mode": mode,
        "leiden_time_seconds": elapsed,
        "cluster_sizes": dict(sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:10]),
        "min_cluster_size": min(cluster_sizes.values()) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes.values()) if cluster_sizes else 0,
        "dry_run": dry_run or (sample_size is not None),
    }

    if remove_hubs:
        result["hubs_removed"] = len(hub_indices) if hub_indices else 0
        result["hub_method"] = hub_method
        result["hub_percentile"] = hub_percentile

    # Add memory and scaling estimates for dry runs
    if sample_size and sample_size < full_node_count:
        scale_factor = full_node_count / sample_size
        result["sample_size"] = sample_size
        result["full_node_count"] = full_node_count
        result["full_edge_count"] = full_edge_count
        result["scale_factor"] = scale_factor
        # Rough estimates (memory scales ~linearly with edges, time scales ~O(n log n))
        result["estimated_full_time_seconds"] = elapsed * scale_factor * 1.5  # Conservative
        result["memory_used_mb"] = (
            mem_after_leiden - mem_before if mem_after_leiden and mem_before else 0
        )

        print("\n  [DRY RUN ESTIMATES]")
        print(
            f"    Sample: {sample_size} topics -> {result['num_clusters']} clusters in {elapsed:.1f}s"
        )
        print(f"    Full run: {full_node_count} topics ({scale_factor:.1f}x)")
        print(f"    Estimated time: {result['estimated_full_time_seconds'] / 60:.1f} minutes")

    # Save results to database (unless dry run)
    if not dry_run and not sample_size:
        log_progress(f"Saving {g.vcount()} cluster assignments to database...")
        save_start = time.time()
        with get_db() as conn:
            if remove_hubs and assignments:
                # Use assignments dict from hub removal
                for topic_id, cluster_id in assignments.items():
                    update_topic_cluster(topic_id, cluster_id)
            else:
                # Standard membership list
                for node_idx, cluster_id in enumerate(membership):
                    topic_id = g.vs[node_idx]["topic_id"]
                    update_topic_cluster(topic_id, cluster_id)
            conn.commit()
        save_elapsed = time.time() - save_start
        log_progress(f"Saved cluster assignments in {save_elapsed:.1f}s")

        # Compute and save multi-cluster assignments for hubs
        if remove_hubs and hub_indices and assignments:
            print("  Computing multi-cluster hub assignments...")
            multi_assignments = assign_hubs_multi_cluster(
                g,
                hub_indices,
                assignments,
                min_connections=3,
                max_clusters=5,
            )
            count = save_multi_cluster_assignments(multi_assignments, assignments)
            result["multi_cluster_hubs"] = len(multi_assignments)
            result["multi_cluster_memberships"] = count
            print(f"  Saved {count} multi-cluster memberships for {len(multi_assignments)} hubs")

        # Update graph vertex attributes
        g.vs["cluster"] = membership
    else:
        print("\n  [DRY RUN] Skipping database update")

    return result


def get_cluster_topics(cluster_id: int) -> list[dict]:
    """Get all topics in a cluster."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, label, occurrence_count
            FROM topics
            WHERE cluster_id = ?
            ORDER BY occurrence_count DESC
        """,
            (cluster_id,),
        )

        return [
            {"id": row[0], "label": row[1], "occurrence_count": row[2]} for row in cursor.fetchall()
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
            cursor.execute(
                """
                SELECT label, occurrence_count
                FROM topics
                WHERE cluster_id = ?
                ORDER BY occurrence_count DESC
                LIMIT 5
            """,
                (cluster_id,),
            )

            top_topics = [{"label": r[0], "count": r[1]} for r in cursor.fetchall()]

            clusters.append(
                {
                    "cluster_id": cluster_id,
                    "size": size,
                    "top_topics": top_topics,
                }
            )

        return clusters


def recluster_mega_clusters(
    size_threshold: int = 1000,
    sub_resolution: float = 0.05,
    dry_run: bool = False,
    progress_file: str | None = None,
) -> dict:
    """
    Re-cluster only oversized clusters, preserving good smaller clusters.

    This is the fastest path to fixing mega-clusters without re-running
    the entire clustering pipeline.

    Args:
        size_threshold: Clusters with more topics than this get re-clustered
        sub_resolution: Resolution for sub-clustering (higher = more splits)
        dry_run: If True, don't save changes to database

    Returns:
        Statistics about the re-clustering
    """
    import time
    from datetime import datetime

    from .topic_graph import build_topic_graph_knn

    def log_progress(msg: str):
        """Write progress message to stdout and optionally to file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {msg}"
        print(full_msg, flush=True)
        if progress_file:
            with open(progress_file, "w") as f:
                f.write(f"{full_msg}\n")

    _log_memory("Before recluster_mega_clusters")

    # 1. Identify mega-clusters from database
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT cluster_id, COUNT(*) as size
            FROM topics
            WHERE cluster_id IS NOT NULL
            GROUP BY cluster_id
            HAVING COUNT(*) >= ?
            ORDER BY size DESC
        """,
            (size_threshold,),
        )

        mega_clusters = [(row[0], row[1]) for row in cursor.fetchall()]

    if not mega_clusters:
        return {
            "mega_clusters_found": 0,
            "message": f"No clusters with {size_threshold}+ topics found",
        }

    log_progress(f"Found {len(mega_clusters)} mega-clusters to re-cluster")
    for cid, size in mega_clusters:
        print(f"    Cluster {cid}: {size} topics", flush=True)

    # 2. Build full graph (needed to extract subgraphs with edges)
    log_progress("Building topic graph...")
    g = build_topic_graph_knn(
        cooccurrence_min=COOCCURRENCE_MIN_COUNT,
        pmi_min=PMI_MIN_THRESHOLD,
        k=CLUSTER_KNN_K,
    )

    _log_memory("After build_topic_graph")

    # Create topic_id -> node_idx mapping
    topic_id_to_idx = {v["topic_id"]: v.index for v in g.vs}

    # 3. Get max existing cluster ID (for new cluster numbering)
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(cluster_id) FROM topics")
        max_cluster_id = cursor.fetchone()[0] or 0

    new_cluster_id_start = max_cluster_id + 1
    new_assignments = {}  # topic_id -> new_cluster_id
    stats_per_mega = []

    # 4. Re-cluster each mega-cluster
    total_mega = len(mega_clusters)
    start_time = time.time()

    for idx, (mega_id, mega_size) in enumerate(mega_clusters):
        elapsed = time.time() - start_time
        if idx > 0:
            rate = idx / elapsed
            remaining = (total_mega - idx) / rate
            log_progress(
                f"Re-clustering mega {idx + 1}/{total_mega}: cluster {mega_id} ({mega_size} topics) | ETA: {remaining / 60:.1f} min"
            )
        else:
            log_progress(
                f"Re-clustering mega {idx + 1}/{total_mega}: cluster {mega_id} ({mega_size} topics)"
            )

        # Get topic IDs in this cluster
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id FROM topics WHERE cluster_id = ?
            """,
                (mega_id,),
            )
            mega_topic_ids = [row[0] for row in cursor.fetchall()]

        # Map to graph node indices (some topics might not be in graph)
        mega_node_indices = [
            topic_id_to_idx[tid] for tid in mega_topic_ids if tid in topic_id_to_idx
        ]

        if len(mega_node_indices) < 10:
            print(f"    Skipping: only {len(mega_node_indices)} topics found in graph")
            continue

        # Extract subgraph
        subgraph = g.subgraph(mega_node_indices)
        print(f"    Subgraph: {subgraph.vcount()} nodes, {subgraph.ecount()} edges")

        if subgraph.ecount() == 0:
            print("    Skipping: no edges in subgraph")
            continue

        # Run Leiden with higher resolution
        weights = subgraph.es["weight"] if "weight" in subgraph.es.attributes() else None
        sub_partition = leidenalg.find_partition(
            subgraph,
            leidenalg.CPMVertexPartition,
            weights=weights,
            resolution_parameter=sub_resolution,
        )

        num_sub_clusters = len(set(sub_partition.membership))
        print(f"    Split into {num_sub_clusters} sub-clusters", flush=True)

        # Map sub-cluster assignments back to topic IDs
        sub_cluster_sizes = defaultdict(int)
        for sub_idx, sub_cluster in enumerate(sub_partition.membership):
            original_node_idx = mega_node_indices[sub_idx]
            topic_id = g.vs[original_node_idx]["topic_id"]
            new_cluster = new_cluster_id_start + sub_cluster
            new_assignments[topic_id] = new_cluster
            sub_cluster_sizes[new_cluster] += 1

        # Update for next mega-cluster
        new_cluster_id_start += num_sub_clusters

        stats_per_mega.append(
            {
                "original_cluster_id": mega_id,
                "original_size": mega_size,
                "sub_clusters": num_sub_clusters,
                "largest_sub": max(sub_cluster_sizes.values()) if sub_cluster_sizes else 0,
            }
        )

    total_elapsed = time.time() - start_time
    log_progress(f"Re-clustered {total_mega} mega-clusters in {total_elapsed:.1f}s")

    # 5. Update database (unless dry run)
    if not dry_run and new_assignments:
        log_progress(f"Updating {len(new_assignments)} topic cluster assignments...")
        with get_db() as conn:
            cursor = conn.cursor()
            for topic_id, new_cluster in new_assignments.items():
                cursor.execute(
                    "UPDATE topics SET cluster_id = ? WHERE id = ?", (new_cluster, topic_id)
                )
            conn.commit()
        print("  Database updated.")
    elif dry_run:
        print(f"\n  [DRY RUN] Would update {len(new_assignments)} topics")

    _log_memory("After recluster_mega_clusters")

    return {
        "mega_clusters_processed": len(mega_clusters),
        "topics_reassigned": len(new_assignments),
        "new_clusters_created": new_cluster_id_start - max_cluster_id - 1,
        "stats_per_mega": stats_per_mega,
        "dry_run": dry_run,
    }


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
        cluster_name = (
            cluster["top_topics"][0]["label"]
            if cluster["top_topics"]
            else f"Cluster {cluster['cluster_id']}"
        )

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
    from .topic_graph import build_topic_graph

    g = build_topic_graph(
        embedding_threshold=EMBEDDING_EDGE_THRESHOLD,
        cooccurrence_min=COOCCURRENCE_MIN_COUNT,
        pmi_min=PMI_MIN_THRESHOLD,
    )
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
            ["ollama", "run", model], input=prompt, capture_output=True, text=True, timeout=30
        )

        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass

    return None


def save_cluster_label(cluster_id: int, label: str):
    """Save a cluster label to the database."""
    from .database import get_db

    with get_db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO cluster_labels (cluster_id, label)
            VALUES (?, ?)
        """,
            (cluster_id, label),
        )
        conn.commit()


def get_cluster_label(cluster_id: int) -> Optional[str]:
    """Get the saved label for a cluster."""
    from .database import get_db

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT label FROM cluster_labels WHERE cluster_id = ?", (cluster_id,))
        row = cursor.fetchone()
        return row[0] if row else None


def label_clusters_batch(
    limit: int = None,
    min_size: int = 10,
    model: str = "gemma3:4b",
    skip_existing: bool = True,
    progress_file: str | None = None,
) -> dict:
    """
    Generate LLM labels for multiple clusters.

    Args:
        limit: Maximum clusters to label (None = all)
        min_size: Minimum cluster size to label
        model: Ollama model to use
        skip_existing: Skip clusters that already have labels

    Returns:
        Stats about labeling
    """
    from .database import get_db

    # Get clusters to label
    with get_db() as conn:
        cursor = conn.cursor()

        if skip_existing:
            cursor.execute(
                """
                SELECT t.cluster_id, COUNT(*) as size
                FROM topics t
                LEFT JOIN cluster_labels cl ON t.cluster_id = cl.cluster_id
                WHERE t.cluster_id IS NOT NULL AND cl.cluster_id IS NULL
                GROUP BY t.cluster_id
                HAVING COUNT(*) >= ?
                ORDER BY size DESC
            """,
                (min_size,),
            )
        else:
            cursor.execute(
                """
                SELECT cluster_id, COUNT(*) as size
                FROM topics
                WHERE cluster_id IS NOT NULL
                GROUP BY cluster_id
                HAVING COUNT(*) >= ?
                ORDER BY size DESC
            """,
                (min_size,),
            )

        clusters = cursor.fetchall()

    if limit:
        clusters = clusters[:limit]

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

    labeled = 0
    failed = 0
    total = len(clusters)
    start_time = time.time()

    log_progress(f"Starting LLM labeling for {total} clusters (model={model})...")

    for i, (cluster_id, size) in enumerate(clusters):
        label = label_cluster_with_llm(cluster_id, model=model)
        if label:
            save_cluster_label(cluster_id, label)
            labeled += 1
            print(f"  Cluster {cluster_id} ({size} topics): {label}", flush=True)
        else:
            failed += 1
            print(f"  Cluster {cluster_id} ({size} topics): FAILED", flush=True)

        # Log progress every 10 clusters or at key milestones
        done = i + 1
        if done % 10 == 0 or done == total:
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            pct = 100 * done / total
            log_progress(
                f"Labeling progress: {done}/{total} ({pct:.1f}%) | "
                f"Labeled: {labeled} | Failed: {failed} | "
                f"Rate: {rate:.2f}/sec | ETA: {remaining / 60:.1f} min"
            )

    return {
        "clusters_processed": len(clusters),
        "labeled": labeled,
        "failed": failed,
    }
