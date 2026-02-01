"""Topic deduplication using embedding similarity."""

from collections import defaultdict

import numpy as np

from .database import get_db, get_topic_embeddings
from .embeddings import bytes_to_embedding, cosine_similarity_matrix


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


def find_duplicate_groups(threshold: float = 0.85) -> list[list[dict]]:
    """
    Find groups of topics that should be merged based on embedding similarity.

    Args:
        threshold: Cosine similarity threshold for merging (0.85 = very similar)

    Returns:
        List of groups, where each group is a list of topic dicts
        that should be merged together
    """
    # Get all topic embeddings
    topic_data = get_topic_embeddings()
    if not topic_data:
        return []

    topic_ids = [t[0] for t in topic_data]
    embeddings = np.array([bytes_to_embedding(t[1]) for t in topic_data])

    # Compute similarity matrix
    sim_matrix = cosine_similarity_matrix(embeddings)

    # Use Union-Find to group similar topics
    n = len(topic_ids)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                uf.union(i, j)

    # Get groups and fetch topic details
    groups = uf.groups()

    # Get topic details
    with get_db() as conn:
        cursor = conn.cursor()

        result = []
        for root, members in groups.items():
            if len(members) > 1:  # Only return groups with duplicates
                group_topics = []
                for idx in members:
                    topic_id = topic_ids[idx]
                    cursor.execute(
                        "SELECT id, label, occurrence_count FROM topics WHERE id = ?",
                        (topic_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        group_topics.append({
                            "id": row[0],
                            "label": row[1],
                            "occurrence_count": row[2],
                        })
                # Sort by occurrence count (keep most frequent as canonical)
                group_topics.sort(key=lambda x: x["occurrence_count"], reverse=True)
                result.append(group_topics)

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
            cursor.execute("""
                UPDATE chunk_topic_links
                SET topic_id = ?
                WHERE topic_id = ?
                AND chunk_id NOT IN (
                    SELECT chunk_id FROM chunk_topic_links WHERE topic_id = ?
                )
            """, (canonical["id"], dup["id"], canonical["id"]))

            # Delete duplicate links (where canonical already exists)
            cursor.execute(
                "DELETE FROM chunk_topic_links WHERE topic_id = ?",
                (dup["id"],)
            )

            # Update cooccurrences
            cursor.execute("""
                UPDATE topic_cooccurrences
                SET topic1_id = ?
                WHERE topic1_id = ?
            """, (canonical["id"], dup["id"]))

            cursor.execute("""
                UPDATE topic_cooccurrences
                SET topic2_id = ?
                WHERE topic2_id = ?
            """, (canonical["id"], dup["id"]))

            # Add occurrence count to canonical
            cursor.execute("""
                UPDATE topics
                SET occurrence_count = occurrence_count + ?
                WHERE id = ?
            """, (dup["occurrence_count"], canonical["id"]))

            # Delete duplicate topic
            cursor.execute("DELETE FROM topics WHERE id = ?", (dup["id"],))

        conn.commit()

    return {
        "canonical": canonical["label"],
        "merged": [d["label"] for d in duplicates],
        "dry_run": False,
    }


def deduplicate_topics(
    threshold: float = 0.85,
    dry_run: bool = False
) -> dict:
    """
    Deduplicate all topics based on embedding similarity.

    Args:
        threshold: Cosine similarity threshold for merging
        dry_run: If True, only report what would be merged

    Returns:
        Statistics about the deduplication
    """
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
        preview.append({
            "canonical": canonical["label"],
            "canonical_count": canonical["occurrence_count"],
            "duplicates": [
                {"label": d["label"], "count": d["occurrence_count"]}
                for d in duplicates
            ],
        })

    return preview
