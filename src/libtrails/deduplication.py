"""Topic deduplication using embedding similarity."""

import sqlite3
import time
from collections import defaultdict
from datetime import datetime

import numpy as np
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn

from .config import IPAD_DB_PATH
from .database import get_db
from .embeddings import bytes_to_embedding


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


def find_duplicate_groups_numpy(
    threshold: float = 0.85,
    show_progress: bool = True,
    progress_file: str | None = None,
    sample_size: int | None = None,
    batch_size: int = 5000,
) -> list[list[dict]]:
    """
    Find duplicate groups using numpy batch operations (FAST).

    Instead of querying sqlite-vec one topic at a time, this loads all
    embeddings into memory and uses vectorized numpy operations to find
    similar topics in batches.

    Args:
        threshold: Cosine similarity threshold (0.85 = very similar)
        show_progress: Whether to show progress bars
        progress_file: Optional file path to write progress updates
        sample_size: If set, only process this many topics (for testing)
        batch_size: Process this many topics per batch (tune for memory)

    Returns:
        List of duplicate groups, first topic in each is canonical.
    """
    start_time = time.time()

    # Load all embeddings into memory
    conn = sqlite3.connect(IPAD_DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT t.id, t.label, t.occurrence_count, t.embedding
        FROM topics t
        WHERE t.embedding IS NOT NULL
        ORDER BY t.id
    """)
    all_topics = cursor.fetchall()
    conn.close()

    if not all_topics:
        return []

    total_topics = len(all_topics)

    # Sample if requested
    if sample_size and sample_size < total_topics:
        import random
        random.seed(42)
        all_topics = random.sample(list(all_topics), sample_size)
        print(f"Sampled {sample_size:,} topics from {total_topics:,}", flush=True)

    n = len(all_topics)
    topic_ids = [t["id"] for t in all_topics]

    # Convert all embeddings to numpy array
    print(f"Loading {n:,} embeddings into memory...", flush=True)
    embeddings = np.array([bytes_to_embedding(t["embedding"]) for t in all_topics])

    # Normalize embeddings for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms

    print(f"Embeddings shape: {embeddings.shape}", flush=True)

    def write_progress(processed: int, pairs: int):
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        remaining = (n - processed) / rate if rate > 0 else 0
        pct = 100 * processed / n

        msg = (
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"Progress: {processed:,}/{n:,} ({pct:.1f}%) | "
            f"Pairs found: {pairs:,} | "
            f"Rate: {rate:.0f}/sec | "
            f"ETA: {remaining / 60:.1f} min"
        )
        print(msg, flush=True)

        if progress_file:
            with open(progress_file, "w") as f:
                f.write(f"{msg}\n")

    # Union-Find for grouping
    uf = UnionFind(n)
    pairs_found = 0

    # Process in batches
    num_batches = (n + batch_size - 1) // batch_size

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        disable=not show_progress,
    ) as progress:
        task = progress.add_task("Finding duplicates (numpy batch)...", total=n)

        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, n)
            batch_embeddings = embeddings[batch_start:batch_end]

            # Compute similarity: batch x all_embeddings
            # For normalized vectors, cosine similarity = dot product
            similarities = np.dot(batch_embeddings, embeddings.T)

            # For each topic in batch, find neighbors above threshold
            for i, row in enumerate(similarities):
                topic_idx = batch_start + i

                # Set self-similarity to 0 to exclude it
                row[topic_idx] = 0

                # Find neighbors above threshold
                above_threshold = np.where(row >= threshold)[0]

                for neighbor_idx in above_threshold:
                    if neighbor_idx != topic_idx:
                        uf.union(topic_idx, neighbor_idx)
                        pairs_found += 1

            progress.update(task, completed=batch_end)

            # Progress logging
            if (batch_idx + 1) % 5 == 0 or batch_idx == num_batches - 1:
                write_progress(batch_end, pairs_found)

    elapsed = time.time() - start_time
    print(f"Complete: {n:,} topics in {elapsed:.1f}s ({n/elapsed:.0f}/sec)", flush=True)
    print(f"Found {pairs_found:,} similar pairs", flush=True)

    # Build duplicate groups
    groups = uf.groups()
    duplicate_groups = {k: v for k, v in groups.items() if len(v) > 1}

    print(f"Building {len(duplicate_groups):,} duplicate groups...", flush=True)

    # Reconnect to get topic details
    conn = sqlite3.connect(IPAD_DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    result = []
    for root, members in duplicate_groups.items():
        group_topics = []
        for idx in members:
            topic_id = topic_ids[idx]
            cursor.execute(
                "SELECT id, label, occurrence_count FROM topics WHERE id = ?", (topic_id,)
            )
            row = cursor.fetchone()
            if row:
                group_topics.append({
                    "id": row[0],
                    "label": row[1],
                    "occurrence_count": row[2],
                })

        # Sort by occurrence count (most frequent = canonical)
        group_topics.sort(key=lambda x: x["occurrence_count"], reverse=True)

        if len(group_topics) > 1:
            result.append(group_topics)

    conn.close()
    print(f"Ready: {len(result):,} duplicate groups", flush=True)

    if progress_file:
        with open(progress_file, "w") as f:
            f.write(f"Complete: {len(result):,} duplicate groups in {elapsed:.1f}s\n")

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
    threshold: float = 0.94,
    dry_run: bool = False,
    sample_size: int | None = None,
    batch_size: int = 5000,
    progress_file: str | None = None,
) -> dict:
    """
    Deduplicate all topics using fast numpy batch processing.

    Args:
        threshold: Cosine similarity threshold for merging (0.95 recommended)
        dry_run: If True, only report what would be merged
        sample_size: If set, only process this many topics (for testing)
        batch_size: Number of topics to process per numpy batch
        progress_file: Optional file path to write progress updates

    Returns:
        Statistics about the deduplication
    """
    groups = find_duplicate_groups_numpy(
        threshold=threshold,
        sample_size=sample_size,
        batch_size=batch_size,
        progress_file=progress_file,
    )

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


def get_deduplication_preview(
    threshold: float = 0.94,
    limit: int = 20,
    sample_size: int | None = None,
    batch_size: int = 5000,
    progress_file: str | None = None,
) -> list[dict]:
    """
    Preview what topics would be deduplicated without making changes.

    Args:
        threshold: Cosine similarity threshold
        limit: Maximum number of groups to return
        sample_size: If set, only process this many topics (for testing)
        batch_size: Batch size for numpy processing
        progress_file: Optional file path to write progress updates

    Returns:
        List of duplicate groups with their topics
    """
    groups = find_duplicate_groups_numpy(
        threshold=threshold,
        sample_size=sample_size,
        batch_size=batch_size,
        progress_file=progress_file,
    )

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
