"""Tests for topic deduplication functionality."""

from unittest.mock import MagicMock, patch

import numpy as np

from libtrails.deduplication import (
    UnionFind,
    deduplicate_topics,
    find_duplicate_groups_numpy,
    get_deduplication_preview,
    merge_groups_batch,
    merge_topic_group,
)


def _make_row(data):
    """Create a dict-like row that also supports integer indexing."""

    class Row(dict):
        def __getitem__(self, key):
            if isinstance(key, str):
                return super().__getitem__(key)
            return list(self.values())[key]

    return Row(data)


class TestFindDuplicateGroupsNumpy:
    """Tests for finding duplicate topic groups using numpy batch operations.

    Note: find_duplicate_groups_numpy() connects directly to the database and
    loads all embeddings into memory for fast batch processing.
    """

    @patch("libtrails.deduplication.get_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_no_duplicates_when_topics_orthogonal(self, mock_sqlite3, mock_get_db):
        """Test when no topics are similar (orthogonal embeddings)."""
        # Create 3 orthogonal unit vectors (no similarity)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        emb2 = np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes()
        emb3 = np.array([0.0, 0.0, 1.0], dtype=np.float32).tobytes()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            _make_row({"id": 1, "label": "topic1", "occurrence_count": 10, "embedding": emb1}),
            _make_row({"id": 2, "label": "topic2", "occurrence_count": 5, "embedding": emb2}),
            _make_row({"id": 3, "label": "topic3", "occurrence_count": 3, "embedding": emb3}),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        # Mock get_db for chunk_topic_links query (book-overlap gate)
        mock_db_conn = MagicMock()
        mock_db_cursor = MagicMock()
        mock_db_cursor.fetchall.return_value = []
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db_conn.__enter__ = MagicMock(return_value=mock_db_conn)
        mock_db_conn.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_db_conn

        groups = find_duplicate_groups_numpy(threshold=0.9, show_progress=False)

        # No groups should be returned when no topics are similar
        assert groups == []

    @patch("libtrails.deduplication.get_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_finds_similar_topics(self, mock_sqlite3, mock_get_db):
        """Test finding similar topics."""
        # Two similar embeddings (1 and 2), one different (3)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        emb2 = np.array([0.99, 0.14, 0.0], dtype=np.float32).tobytes()  # Similar to 1
        emb3 = np.array([0.0, 1.0, 0.0], dtype=np.float32).tobytes()  # Different

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # First call: load all topics
        # Second call: get topic details for group members
        mock_cursor.fetchall.return_value = [
            _make_row({"id": 1, "label": "philosophy", "occurrence_count": 10, "embedding": emb1}),
            _make_row({"id": 2, "label": "Philosophy", "occurrence_count": 5, "embedding": emb2}),
            _make_row({"id": 3, "label": "science", "occurrence_count": 3, "embedding": emb3}),
        ]
        # For fetchone calls when building groups
        mock_cursor.fetchone.side_effect = [
            (1, "philosophy", 10),
            (2, "Philosophy", 5),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        # Mock get_db — both topics share book 1 (needed for two-tier dedup)
        mock_db_conn = MagicMock()
        mock_db_cursor = MagicMock()
        mock_db_cursor.fetchall.return_value = [(1, 1), (2, 1)]
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db_conn.__enter__ = MagicMock(return_value=mock_db_conn)
        mock_db_conn.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_db_conn

        groups = find_duplicate_groups_numpy(threshold=0.9, show_progress=False)

        # Should find one group with topics 1 and 2
        assert len(groups) == 1
        assert len(groups[0]) == 2

    @patch("libtrails.deduplication.get_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_canonical_is_most_frequent(self, mock_sqlite3, mock_get_db):
        """Test that canonical topic has highest occurrence count."""
        # Two identical embeddings
        emb = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            _make_row({"id": 1, "label": "topic1", "occurrence_count": 5, "embedding": emb}),
            _make_row({"id": 2, "label": "topic2", "occurrence_count": 10, "embedding": emb}),
        ]
        # Topic 2 has higher count (10 > 5)
        mock_cursor.fetchone.side_effect = [
            (1, "topic1", 5),
            (2, "topic2", 10),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        # Mock get_db — both topics share book 1
        mock_db_conn = MagicMock()
        mock_db_cursor = MagicMock()
        mock_db_cursor.fetchall.return_value = [(1, 1), (2, 1)]
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db_conn.__enter__ = MagicMock(return_value=mock_db_conn)
        mock_db_conn.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_db_conn

        groups = find_duplicate_groups_numpy(threshold=0.9, show_progress=False)

        # First topic in group should be the most frequent (canonical)
        assert len(groups) == 1
        assert groups[0][0]["occurrence_count"] == 10


class TestGetDeduplicationPreview:
    """Tests for deduplication preview."""

    @patch("libtrails.deduplication.find_duplicate_groups_numpy")
    def test_preview_returns_groups(self, mock_find_groups):
        """Test that preview returns duplicate groups."""
        mock_find_groups.return_value = [
            [
                {"id": 1, "label": "philosophy", "occurrence_count": 10},
                {"id": 2, "label": "Philosophy", "occurrence_count": 5},
            ]
        ]

        preview = get_deduplication_preview(threshold=0.9, limit=10)

        assert isinstance(preview, list)
        assert len(preview) == 1
        assert preview[0]["canonical"] == "philosophy"
        assert len(preview[0]["duplicates"]) == 1

    @patch("libtrails.deduplication.find_duplicate_groups_numpy")
    def test_preview_empty_when_no_duplicates(self, mock_find_groups):
        """Test preview returns empty when no duplicates."""
        mock_find_groups.return_value = []

        preview = get_deduplication_preview(threshold=0.9, limit=10)

        assert preview == []


class TestDeduplicateTopics:
    """Tests for actual deduplication."""

    @patch("libtrails.deduplication.find_duplicate_groups_numpy")
    def test_dry_run_no_changes(self, mock_find_groups):
        """Test that dry run doesn't modify database."""
        mock_find_groups.return_value = [
            [
                {"id": 1, "label": "philosophy", "occurrence_count": 10},
                {"id": 2, "label": "Philosophy", "occurrence_count": 5},
            ]
        ]

        result = deduplicate_topics(threshold=0.9, dry_run=True)

        # Dry run should return stats without merging
        assert "topics_merged" in result
        assert result["dry_run"] is True
        assert result["duplicate_groups"] == 1
        assert result["topics_merged"] == 1

    @patch("libtrails.deduplication.merge_groups_batch")
    @patch("libtrails.deduplication.find_duplicate_groups_numpy")
    def test_actual_merge_calls_batch(self, mock_find_groups, mock_merge):
        """Test that actual deduplication uses batch merge."""
        mock_find_groups.return_value = [
            [
                {"id": 1, "label": "philosophy", "occurrence_count": 10},
                {"id": 2, "label": "Philosophy", "occurrence_count": 5},
            ]
        ]
        mock_merge.return_value = {
            "groups_merged": 1,
            "topics_merged": 1,
        }

        result = deduplicate_topics(threshold=0.9, dry_run=False)

        mock_merge.assert_called_once()
        assert result["dry_run"] is False

    @patch("libtrails.deduplication.find_duplicate_groups_numpy")
    def test_returns_merge_stats(self, mock_find_groups):
        """Test that deduplication returns statistics."""
        mock_find_groups.return_value = []

        result = deduplicate_topics(threshold=0.9, dry_run=False)

        assert "topics_merged" in result
        assert "duplicate_groups" in result
        assert result["topics_merged"] == 0


class TestUnionFind:
    """Tests for the UnionFind data structure."""

    def test_initial_state(self):
        """Each element is its own parent initially."""
        uf = UnionFind(5)
        for i in range(5):
            assert uf.find(i) == i

    def test_union_and_find(self):
        """Union merges groups, find returns same root."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.find(0) == uf.find(1) == uf.find(2)
        assert uf.find(3) != uf.find(0)

    def test_groups(self):
        """Groups returns connected components."""
        uf = UnionFind(5)
        uf.union(0, 1)
        uf.union(2, 3)
        groups = uf.groups()
        # Should have 3 groups: {0,1}, {2,3}, {4}
        group_sizes = sorted(len(v) for v in groups.values())
        assert group_sizes == [1, 2, 2]


class TestMergeTopicGroup:
    """Tests for merge_topic_group()."""

    def test_single_topic_no_merge(self):
        """Single topic group doesn't merge anything."""
        group = [{"id": 1, "label": "topic", "occurrence_count": 5}]
        result = merge_topic_group(group, dry_run=True)
        assert result["merged"] == 0

    def test_dry_run_returns_stats(self):
        """Dry run returns what would be merged without DB changes."""
        group = [
            {"id": 1, "label": "philosophy", "occurrence_count": 10},
            {"id": 2, "label": "Philosophy", "occurrence_count": 5},
        ]
        result = merge_topic_group(group, dry_run=True)
        assert result["canonical"] == "philosophy"
        assert "Philosophy" in result["merged"]
        assert result["dry_run"] is True


class TestMergeGroupsBatch:
    """Tests for merge_groups_batch()."""

    def test_empty_groups_returns_zero(self):
        """Empty groups list returns zero counts."""
        result = merge_groups_batch([])
        assert result["groups_merged"] == 0
        assert result["topics_merged"] == 0

    @patch("libtrails.vector_search.get_vec_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_creates_temp_table_and_merges(self, mock_sqlite3, mock_vec_db):
        """Creates temp table and performs bulk merge."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 5
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        mock_vec_conn = MagicMock()
        mock_vec_db.return_value = mock_vec_conn

        groups = [
            [
                {"id": 1, "label": "philosophy", "occurrence_count": 10},
                {"id": 2, "label": "Philosophy", "occurrence_count": 5},
                {"id": 3, "label": "PHILOSOPHY", "occurrence_count": 2},
            ]
        ]

        result = merge_groups_batch(groups)

        assert result["groups_merged"] == 1
        assert result["topics_merged"] == 2  # 2 duplicates merged into canonical

        # Verify temp table was created
        create_calls = [
            call
            for call in mock_cursor.execute.call_args_list
            if "CREATE TEMP TABLE dup_map" in str(call)
        ]
        assert len(create_calls) == 1

    @patch("libtrails.vector_search.get_vec_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_returns_correct_stats(self, mock_sqlite3, mock_vec_db):
        """Returns correct merge statistics."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.rowcount = 10
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        mock_vec_db.return_value = MagicMock()

        groups = [
            [
                {"id": 1, "label": "a", "occurrence_count": 10},
                {"id": 2, "label": "b", "occurrence_count": 5},
            ],
            [
                {"id": 3, "label": "c", "occurrence_count": 8},
                {"id": 4, "label": "d", "occurrence_count": 3},
                {"id": 5, "label": "e", "occurrence_count": 1},
            ],
        ]

        result = merge_groups_batch(groups)

        assert result["groups_merged"] == 2
        assert result["topics_merged"] == 3  # 1 + 2 duplicates


class TestTwoTierDedup:
    """Tests for two-tier deduplication threshold behavior."""

    @patch("libtrails.deduplication.get_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_high_confidence_merges_without_shared_book(self, mock_sqlite3, mock_get_db):
        """Topics above 0.95 merge even without shared books."""
        # Two nearly identical embeddings (>0.95 similarity)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        emb2 = np.array([0.999, 0.04, 0.0], dtype=np.float32).tobytes()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            _make_row({"id": 1, "label": "topic1", "occurrence_count": 10, "embedding": emb1}),
            _make_row({"id": 2, "label": "topic2", "occurrence_count": 5, "embedding": emb2}),
        ]
        mock_cursor.fetchone.side_effect = [
            (1, "topic1", 10),
            (2, "topic2", 5),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        # No shared books — but still should merge above 0.95
        mock_db_conn = MagicMock()
        mock_db_cursor = MagicMock()
        mock_db_cursor.fetchall.return_value = []  # No shared books
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db_conn.__enter__ = MagicMock(return_value=mock_db_conn)
        mock_db_conn.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_db_conn

        groups = find_duplicate_groups_numpy(threshold=0.85, show_progress=False)
        assert len(groups) == 1

    @patch("libtrails.deduplication.get_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_mid_tier_blocked_without_shared_book(self, mock_sqlite3, mock_get_db):
        """Topics between 0.85-0.95 do NOT merge without shared books."""
        # Similarity ~0.90 (between 0.85 threshold and 0.95 high-confidence)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        emb2 = np.array([0.90, 0.436, 0.0], dtype=np.float32).tobytes()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            _make_row({"id": 1, "label": "topic1", "occurrence_count": 10, "embedding": emb1}),
            _make_row({"id": 2, "label": "topic2", "occurrence_count": 5, "embedding": emb2}),
        ]
        # fetchone is called when building group details
        mock_cursor.fetchone.side_effect = [
            (1, "topic1", 10),
            (2, "topic2", 5),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.row_factory = None
        mock_sqlite3.connect.return_value = mock_conn
        mock_sqlite3.Row = MagicMock()

        # No shared books
        mock_db_conn = MagicMock()
        mock_db_cursor = MagicMock()
        mock_db_cursor.fetchall.return_value = []
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db_conn.__enter__ = MagicMock(return_value=mock_db_conn)
        mock_db_conn.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_db_conn

        groups = find_duplicate_groups_numpy(threshold=0.85, show_progress=False)
        # Should NOT merge because they don't share a book (mid-tier range)
        assert len(groups) == 0

    @patch("libtrails.deduplication.get_db")
    @patch("libtrails.deduplication.sqlite3")
    def test_below_threshold_never_merges(self, mock_sqlite3, mock_get_db):
        """Topics below 0.85 never merge."""
        # Low similarity (~0.70)
        emb1 = np.array([1.0, 0.0, 0.0], dtype=np.float32).tobytes()
        emb2 = np.array([0.7, 0.71, 0.0], dtype=np.float32).tobytes()

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            _make_row({"id": 1, "label": "topic1", "occurrence_count": 10, "embedding": emb1}),
            _make_row({"id": 2, "label": "topic2", "occurrence_count": 5, "embedding": emb2}),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_sqlite3.connect.return_value = mock_conn

        # Even with shared books, below threshold
        mock_db_conn = MagicMock()
        mock_db_cursor = MagicMock()
        mock_db_cursor.fetchall.return_value = [(1, 1), (2, 1)]
        mock_db_conn.cursor.return_value = mock_db_cursor
        mock_db_conn.__enter__ = MagicMock(return_value=mock_db_conn)
        mock_db_conn.__exit__ = MagicMock(return_value=False)
        mock_get_db.return_value = mock_db_conn

        groups = find_duplicate_groups_numpy(threshold=0.85, show_progress=False)
        assert len(groups) == 0
