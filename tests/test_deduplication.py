"""Tests for topic deduplication functionality."""

from unittest.mock import MagicMock, patch

import numpy as np

from libtrails.deduplication import (
    deduplicate_topics,
    find_duplicate_groups_numpy,
    get_deduplication_preview,
)


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

        def make_row(data):
            class Row(dict):
                def __getitem__(self, key):
                    if isinstance(key, str):
                        return super().__getitem__(key)
                    return list(self.values())[key]

            return Row(data)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            make_row({"id": 1, "label": "topic1", "occurrence_count": 10, "embedding": emb1}),
            make_row({"id": 2, "label": "topic2", "occurrence_count": 5, "embedding": emb2}),
            make_row({"id": 3, "label": "topic3", "occurrence_count": 3, "embedding": emb3}),
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

        def make_row(data):
            class Row(dict):
                def __getitem__(self, key):
                    if isinstance(key, str):
                        return super().__getitem__(key)
                    return list(self.values())[key]

            return Row(data)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # First call: load all topics
        # Second call: get topic details for group members
        mock_cursor.fetchall.return_value = [
            make_row({"id": 1, "label": "philosophy", "occurrence_count": 10, "embedding": emb1}),
            make_row({"id": 2, "label": "Philosophy", "occurrence_count": 5, "embedding": emb2}),
            make_row({"id": 3, "label": "science", "occurrence_count": 3, "embedding": emb3}),
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

        def make_row(data):
            class Row(dict):
                def __getitem__(self, key):
                    if isinstance(key, str):
                        return super().__getitem__(key)
                    return list(self.values())[key]

            return Row(data)

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            make_row({"id": 1, "label": "topic1", "occurrence_count": 5, "embedding": emb}),
            make_row({"id": 2, "label": "topic2", "occurrence_count": 10, "embedding": emb}),
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
