"""Topic deduplication using embedding similarity."""

from collections import defaultdict

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .database import get_db, get_topic_embeddings
from .embeddings import bytes_to_embedding, cosine_similarity_matrix
from .vector_search import get_vec_db


class UnionFind:
    """Union-Find data structure for grouping similar topics."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

    def groups(self) -> dict[int, list[int]]:
        """Get all groups as {root: [members]}."""
        result = defaultdict(list)
        for i in range(len(self.parent)):
            result[self.find(i)].append(i)
        return dict(result)

    def to_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "parent": self.parent,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UnionFind":
        """Restore from checkpoint."""
        n = len(data["parent"])
        uf = cls(n)
        uf.parent = data["parent"]
        uf.rank = data["rank"]
        return uf


def find_duplicate_groups_batch(
    threshold: float = 0.85,
    batch_size: int = 1000,
    k_neighbors: int = 50,
    show_progress: bool = True,
    progress_file: str | None = None,
    checkpoint_file: str | None = "/tmp/libtrails_dedupe_checkpoint.json",
    checkpoint_interval: int = 10000,
) -> list[list[dict]]:
    """
    Find groups of topics that should be merged using batch vector search.

    This is more memory-efficient than the full matrix approach as it uses
    sqlite-vec for finding similar topics.

    Args:
        threshold: Cosine similarity threshold for merging (0.85 = very similar)
        batch_size: Number of topics to process per batch
        k_neighbors: Number of nearest neighbors to check per topic
        show_progress: Whether to show progress bars
        progress_file: Optional file path to write progress updates (for background runs)
        checkpoint_file: Optional file path for saving/resuming checkpoints
        checkpoint_interval: Save checkpoint every N topics processed

    Returns:
        List of groups, where each group is a list of topic dicts
        that should be merged together. First topic in each group is canonical.
    """
    import json
    import os
    import time
    from datetime import datetime

    # Distance threshold: similarity = 1 - distance, so distance = 1 - similarity
    distance_threshold = 1.0 - threshold

    conn = get_vec_db()
    cursor = conn.cursor()

    # Get all topic IDs with embeddings
    cursor.execute("""
        SELECT t.id, t.label, t.occurrence_count, t.embedding
        FROM topics t
        WHERE t.embedding IS NOT NULL
        ORDER BY t.id
    """)
    all_topics = cursor.fetchall()

    if not all_topics:
        conn.close()
        return []

    n = len(all_topics)
    topic_ids = [t["id"] for t in all_topics]
    id_to_idx = {tid: idx for idx, tid in enumerate(topic_ids)}

    # Check for checkpoint to resume from
    start_idx = 0
    pairs_found = 0
    uf = UnionFind(n)

    if checkpoint_file and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            # Validate checkpoint is for same dataset
            if checkpoint.get("n") == n and checkpoint.get("threshold") == threshold:
                uf = UnionFind.from_dict(checkpoint["union_find"])
                start_idx = checkpoint["iteration"]
                pairs_found = checkpoint["pairs_found"]
                print(
                    f"Resuming from checkpoint: iteration {start_idx:,}/{n:,}, pairs found: {pairs_found:,}",
                    flush=True,
                )
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"Warning: Could not load checkpoint, starting fresh: {e}", flush=True)

    start_time = time.time()

    # Progress logging interval
    log_interval = 5000  # Log every 5000 topics

    def write_progress(i: int, pairs: int):
        """Write progress to file and stdout."""
        elapsed = time.time() - start_time
        # Adjust rate calculation for resumed runs
        processed = i - start_idx
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (n - i) / rate if rate > 0 else 0
        pct = 100 * i / n

        msg = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Progress: {i:,}/{n:,} ({pct:.1f}%) | "
            f"Pairs found: {pairs:,} | "
            f"Rate: {rate:.1f}/sec | "
            f"ETA: {remaining / 60:.1f} min"
        )
        print(msg, flush=True)

        if progress_file:
            with open(progress_file, "w") as f:
                f.write(f"{msg}\n")
                f.write(f"elapsed_sec={elapsed:.1f}\n")
                f.write(f"processed={i}\n")
                f.write(f"total={n}\n")
                f.write(f"pairs_found={pairs}\n")

    def save_checkpoint(i: int, pairs: int):
        """Save checkpoint for crash recovery."""
        if not checkpoint_file:
            return
        checkpoint = {
            "n": n,
            "threshold": threshold,
            "iteration": i,
            "pairs_found": pairs,
            "union_find": uf.to_dict(),
            "timestamp": datetime.now().isoformat(),
        }
        # Write to temp file first, then rename (atomic on POSIX)
        tmp_file = checkpoint_file + ".tmp"
        with open(tmp_file, "w") as f:
            json.dump(checkpoint, f)
        os.rename(tmp_file, checkpoint_file)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("Finding similar topics via vector search...", total=n)

        # Initial progress message
        if start_idx == 0:
            print(f"Starting vector search for {n:,} topics (threshold={threshold})...", flush=True)
        else:
            print(f"Continuing vector search from {start_idx:,}/{n:,}...", flush=True)

        # Advance progress bar to start position if resuming
        if start_idx > 0:
            progress.update(task, completed=start_idx)

        for i, topic in enumerate(all_topics):
            # Skip already processed topics when resuming
            if i < start_idx:
                continue

            topic_id = topic["id"]
            embedding = topic["embedding"]

            # Query sqlite-vec for similar topics
            cursor.execute(
                """
                SELECT tv.topic_id, tv.distance
                FROM topic_vectors tv
                WHERE tv.embedding MATCH ? AND k = ?
                ORDER BY tv.distance
            """,
                (embedding, k_neighbors),
            )

            for row in cursor.fetchall():
                neighbor_id = row["topic_id"]
                distance = row["distance"]

                # Skip self
                if neighbor_id == topic_id:
                    continue

                # Check if similar enough
                if distance <= distance_threshold:
                    neighbor_idx = id_to_idx.get(neighbor_id)
                    if neighbor_idx is not None:
                        uf.union(i, neighbor_idx)
                        pairs_found += 1

            progress.advance(task)

            # Log progress periodically
            if (i + 1) % log_interval == 0:
                write_progress(i + 1, pairs_found)

            # Save checkpoint periodically
            if checkpoint_file and (i + 1) % checkpoint_interval == 0:
                save_checkpoint(i + 1, pairs_found)

        # Final progress
        write_progress(n, pairs_found)
        progress.console.print(f"Found {pairs_found} similar pairs")

    # IMPORTANT: Close the vec connection completely before reopening
    conn.close()
    del conn
    del cursor

    # Delete checkpoint file on successful completion
    if checkpoint_file and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("Checkpoint file removed (completed successfully)", flush=True)

    # Get groups and build result
    groups = uf.groups()

    # Use regular database connection (not vec) to fetch topic details
    # This avoids any lingering locks from sqlite-vec
    import sqlite3

    from .config import IPAD_DB_PATH

    db_conn = sqlite3.connect(IPAD_DB_PATH, timeout=30.0)
    db_conn.row_factory = sqlite3.Row
    cursor = db_conn.cursor()

    result = []
    for root, members in groups.items():
        if len(members) > 1:  # Only groups with duplicates
            group_topics = []
            for idx in members:
                topic_id = topic_ids[idx]
                cursor.execute(
                    "SELECT id, label, occurrence_count FROM topics WHERE id = ?", (topic_id,)
                )
                row = cursor.fetchone()
                if row:
                    group_topics.append(
                        {
                            "id": row[0],
                            "label": row[1],
                            "occurrence_count": row[2],
                        }
                    )

            # Sort by occurrence count (keep most frequent as canonical)
            group_topics.sort(key=lambda x: x["occurrence_count"], reverse=True)

            if len(group_topics) > 1:
                result.append(group_topics)

    db_conn.close()
    return result


def find_duplicate_groups(threshold: float = 0.85, show_progress: bool = True) -> list[list[dict]]:
    """
    Find groups of topics that should be merged based on embedding similarity.

    Uses Union-Find for initial grouping, then validates that all members
    are directly similar to the canonical topic (highest occurrence count).
    This prevents transitive chains where A~B and B~C would group A,B,C
    even when A and C aren't similar.

    Args:
        threshold: Cosine similarity threshold for merging (0.85 = very similar)
        show_progress: Whether to show progress bars

    Returns:
        List of groups, where each group is a list of topic dicts
        that should be merged together. First topic in each group is canonical.
    """
    # Get all topic embeddings
    topic_data = get_topic_embeddings()
    if not topic_data:
        return []

    topic_ids = [t[0] for t in topic_data]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        disable=not show_progress,
    ) as progress:
        # Convert embeddings with progress
        embed_task = progress.add_task("Converting embeddings...", total=len(topic_data))
        embeddings_list = []
        for t in topic_data:
            embeddings_list.append(bytes_to_embedding(t[1]))
            progress.advance(embed_task)
        embeddings = np.array(embeddings_list)

        # Build index mapping topic_id -> array index
        {tid: idx for idx, tid in enumerate(topic_ids)}

        # Compute similarity matrix
        progress.add_task("Computing similarity matrix...", total=None)
        sim_matrix = cosine_similarity_matrix(embeddings)

        # Use Union-Find to group similar topics (initial pass)
        n = len(topic_ids)
        uf = UnionFind(n)

        # Calculate total pairs for progress
        total_pairs = n * (n - 1) // 2
        pair_task = progress.add_task("Finding similar pairs...", total=total_pairs)

        pairs_checked = 0
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    uf.union(i, j)
                pairs_checked += 1
                if pairs_checked % 100000 == 0:  # Update every 100k pairs
                    progress.update(pair_task, completed=pairs_checked)
        progress.update(pair_task, completed=total_pairs)

    # Get groups and fetch topic details
    groups = uf.groups()

    # Get topic details and validate groups
    with get_db() as conn:
        cursor = conn.cursor()

        result = []
        for root, members in groups.items():
            if len(members) > 1:  # Only process groups with potential duplicates
                group_topics = []
                for idx in members:
                    topic_id = topic_ids[idx]
                    cursor.execute(
                        "SELECT id, label, occurrence_count FROM topics WHERE id = ?", (topic_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        group_topics.append(
                            {
                                "id": row[0],
                                "label": row[1],
                                "occurrence_count": row[2],
                                "_idx": idx,  # Keep track of embedding index
                            }
                        )

                # Sort by occurrence count (keep most frequent as canonical)
                group_topics.sort(key=lambda x: x["occurrence_count"], reverse=True)

                # Validate: only keep members that are similar to canonical
                canonical = group_topics[0]
                canonical_idx = canonical["_idx"]

                validated_group = [canonical]
                for topic in group_topics[1:]:
                    topic_idx = topic["_idx"]
                    if sim_matrix[canonical_idx, topic_idx] >= threshold:
                        validated_group.append(topic)

                # Clean up internal index before returning
                for topic in validated_group:
                    del topic["_idx"]

                # Only include groups with actual duplicates after validation
                if len(validated_group) > 1:
                    result.append(validated_group)

        return result


def merge_topic_group(group: list[dict], dry_run: bool = False) -> dict:
    """
    Merge a group of duplicate topics into the canonical (most frequent) one.

    Args:
        group: List of topic dicts to merge (first one is canonical)
        dry_run: If True, don't actually modify the database

    Returns:
        Dict with merge statistics
    """
    if len(group) < 2:
        return {"merged": 0, "canonical": None}

    canonical = group[0]
    duplicates = group[1:]

    if dry_run:
        return {
            "canonical": canonical["label"],
            "merged": [d["label"] for d in duplicates],
            "dry_run": True,
        }

    with get_db() as conn:
        cursor = conn.cursor()

        for dup in duplicates:
            # Update chunk_topic_links to point to canonical
            cursor.execute(
                """
                UPDATE chunk_topic_links
                SET topic_id = ?
                WHERE topic_id = ?
                AND chunk_id NOT IN (
                    SELECT chunk_id FROM chunk_topic_links WHERE topic_id = ?
                )
            """,
                (canonical["id"], dup["id"], canonical["id"]),
            )

            # Delete duplicate links (where canonical already exists)
            cursor.execute("DELETE FROM chunk_topic_links WHERE topic_id = ?", (dup["id"],))

            # Handle cooccurrences - merge counts for conflicts, then delete duplicates
            # First, update cooccurrences where canonical doesn't already have a relationship
            # For topic1_id updates
            cursor.execute(
                """
                UPDATE topic_cooccurrences
                SET topic1_id = ?
                WHERE topic1_id = ?
                AND NOT EXISTS (
                    SELECT 1 FROM topic_cooccurrences tc2
                    WHERE tc2.topic1_id = ? AND tc2.topic2_id = topic_cooccurrences.topic2_id
                )
            """,
                (canonical["id"], dup["id"], canonical["id"]),
            )

            # For topic2_id updates
            cursor.execute(
                """
                UPDATE topic_cooccurrences
                SET topic2_id = ?
                WHERE topic2_id = ?
                AND NOT EXISTS (
                    SELECT 1 FROM topic_cooccurrences tc2
                    WHERE tc2.topic1_id = topic_cooccurrences.topic1_id AND tc2.topic2_id = ?
                )
            """,
                (canonical["id"], dup["id"], canonical["id"]),
            )

            # Add counts from duplicate's cooccurrences to canonical's existing ones
            cursor.execute(
                """
                UPDATE topic_cooccurrences
                SET count = count + COALESCE((
                    SELECT count FROM topic_cooccurrences dup_co
                    WHERE dup_co.topic1_id = ? AND dup_co.topic2_id = topic_cooccurrences.topic2_id
                ), 0)
                WHERE topic1_id = ?
            """,
                (dup["id"], canonical["id"]),
            )

            cursor.execute(
                """
                UPDATE topic_cooccurrences
                SET count = count + COALESCE((
                    SELECT count FROM topic_cooccurrences dup_co
                    WHERE dup_co.topic1_id = topic_cooccurrences.topic1_id AND dup_co.topic2_id = ?
                ), 0)
                WHERE topic2_id = ?
            """,
                (dup["id"], canonical["id"]),
            )

            # Delete all remaining cooccurrences for the duplicate topic
            cursor.execute(
                "DELETE FROM topic_cooccurrences WHERE topic1_id = ? OR topic2_id = ?",
                (dup["id"], dup["id"]),
            )

            # Add occurrence count to canonical
            cursor.execute(
                """
                UPDATE topics
                SET occurrence_count = occurrence_count + ?
                WHERE id = ?
            """,
                (dup["occurrence_count"], canonical["id"]),
            )

            # Delete duplicate topic
            cursor.execute("DELETE FROM topics WHERE id = ?", (dup["id"],))

        conn.commit()

    return {
        "canonical": canonical["label"],
        "merged": [d["label"] for d in duplicates],
        "dry_run": False,
    }


def merge_groups_batch(groups: list[list[dict]], commit_every: int = 100) -> dict:
    """
    Merge all duplicate groups in a single transaction with periodic commits.

    This is much faster than merging one group at a time.

    Args:
        groups: List of duplicate groups to merge
        commit_every: Commit after this many groups

    Returns:
        Statistics about the merge operation
    """
    import sqlite3

    from .config import IPAD_DB_PATH

    if not groups:
        return {"groups_merged": 0, "topics_merged": 0}

    conn = sqlite3.connect(IPAD_DB_PATH, timeout=60.0)
    cursor = conn.cursor()

    total_topics_merged = 0
    groups_processed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        task = progress.add_task("Merging duplicate groups...", total=len(groups))

        for group in groups:
            if len(group) < 2:
                progress.advance(task)
                continue

            canonical = group[0]
            duplicates = group[1:]

            for dup in duplicates:
                # Update chunk_topic_links to point to canonical
                cursor.execute(
                    """
                    UPDATE chunk_topic_links
                    SET topic_id = ?
                    WHERE topic_id = ?
                    AND chunk_id NOT IN (
                        SELECT chunk_id FROM chunk_topic_links WHERE topic_id = ?
                    )
                """,
                    (canonical["id"], dup["id"], canonical["id"]),
                )

                # Delete duplicate links
                cursor.execute("DELETE FROM chunk_topic_links WHERE topic_id = ?", (dup["id"],))

                # Delete cooccurrences for duplicate (skip complex merge for speed)
                cursor.execute(
                    "DELETE FROM topic_cooccurrences WHERE topic1_id = ? OR topic2_id = ?",
                    (dup["id"], dup["id"]),
                )

                # Add occurrence count to canonical
                cursor.execute(
                    """
                    UPDATE topics
                    SET occurrence_count = occurrence_count + ?
                    WHERE id = ?
                """,
                    (dup["occurrence_count"], canonical["id"]),
                )

                # Delete duplicate topic
                cursor.execute("DELETE FROM topics WHERE id = ?", (dup["id"],))

                # Delete from vector index
                cursor.execute("DELETE FROM topic_vectors WHERE topic_id = ?", (dup["id"],))

                total_topics_merged += 1

            groups_processed += 1
            progress.advance(task)

            # Periodic commit for safety
            if groups_processed % commit_every == 0:
                conn.commit()

        # Final commit
        conn.commit()

    conn.close()

    return {
        "groups_merged": groups_processed,
        "topics_merged": total_topics_merged,
    }


def deduplicate_topics(
    threshold: float = 0.85,
    dry_run: bool = False,
    use_batch: bool = True,
    batch_size: int = 1000,
    k_neighbors: int = 50,
    progress_file: str | None = None,
) -> dict:
    """
    Deduplicate all topics based on embedding similarity.

    Args:
        threshold: Cosine similarity threshold for merging
        dry_run: If True, only report what would be merged
        use_batch: If True, use memory-efficient batch approach with vector search
        batch_size: Number of topics to process per batch (only for batch mode)
        k_neighbors: Number of nearest neighbors to check (only for batch mode)
        progress_file: Optional file path to write progress updates (for background runs)

    Returns:
        Statistics about the deduplication
    """
    if use_batch:
        groups = find_duplicate_groups_batch(
            threshold=threshold,
            batch_size=batch_size,
            k_neighbors=k_neighbors,
            progress_file=progress_file,
        )
    else:
        groups = find_duplicate_groups(threshold)

    if not groups:
        return {
            "duplicate_groups": 0,
            "topics_merged": 0,
            "dry_run": dry_run,
        }

    if dry_run:
        # Just count without merging
        total_merged = sum(len(g) - 1 for g in groups)
        return {
            "duplicate_groups": len(groups),
            "topics_merged": total_merged,
            "dry_run": True,
            "sample_merges": [
                {"canonical": g[0]["label"], "duplicates": [d["label"] for d in g[1:]]}
                for g in groups[:10]
            ],
        }

    # Use batch merge for speed
    result = merge_groups_batch(groups)

    return {
        "duplicate_groups": result["groups_merged"],
        "topics_merged": result["topics_merged"],
        "dry_run": False,
    }


def deduplicate_topics_old(
    threshold: float = 0.85,
    dry_run: bool = False,
    use_batch: bool = True,
    batch_size: int = 1000,
    k_neighbors: int = 50,
) -> dict:
    """Old version - kept for reference. Use deduplicate_topics instead."""
    if use_batch:
        groups = find_duplicate_groups_batch(
            threshold=threshold,
            batch_size=batch_size,
            k_neighbors=k_neighbors,
        )
    else:
        groups = find_duplicate_groups(threshold)

    if not groups:
        return {
            "duplicate_groups": 0,
            "topics_merged": 0,
            "dry_run": dry_run,
        }

    total_merged = 0
    merge_results = []

    for group in groups:
        result = merge_topic_group(group, dry_run=dry_run)
        merge_results.append(result)
        total_merged += len(result.get("merged", []))

    return {
        "duplicate_groups": len(groups),
        "topics_merged": total_merged,
        "merges": merge_results[:10],  # Show first 10 merges
        "dry_run": dry_run,
    }


def get_deduplication_preview(threshold: float = 0.85, limit: int = 20) -> list[dict]:
    """
    Preview what topics would be deduplicated without making changes.

    Args:
        threshold: Cosine similarity threshold
        limit: Maximum number of groups to return

    Returns:
        List of duplicate groups with their topics
    """
    groups = find_duplicate_groups(threshold)

    preview = []
    for group in groups[:limit]:
        canonical = group[0]
        duplicates = group[1:]
        preview.append(
            {
                "canonical": canonical["label"],
                "canonical_count": canonical["occurrence_count"],
                "duplicates": [
                    {"label": d["label"], "count": d["occurrence_count"]} for d in duplicates
                ],
            }
        )

    return preview
