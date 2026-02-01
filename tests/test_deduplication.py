"""Tests for topic deduplication functionality."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from libtrails.deduplication import (
    find_duplicate_groups,
    get_deduplication_preview,
    deduplicate_topics,
)


class TestFindDuplicateGroups:
    """Tests for finding duplicate topic groups.

    Note: find_duplicate_groups() takes only a threshold parameter.
    It fetches embeddings internally via get_topic_embeddings().
    """

    @patch('libtrails.deduplication.bytes_to_embedding')
    @patch('libtrails.deduplication.get_topic_embeddings')
    @patch('libtrails.deduplication.get_db')
    def test_no_duplicates(self, mock_db, mock_get_embeddings, mock_bytes_to_embedding):
        """Test when no topics are similar."""
        # Very different embeddings (orthogonal)
        mock_get_embeddings.return_value = [
            (1, b'emb1'),
            (2, b'emb2'),
            (3, b'emb3'),
        ]
        mock_bytes_to_embedding.side_effect = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

        # Mock database for topic details
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchone.return_value = None  # No topics will be fetched (no duplicates)
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        groups = find_duplicate_groups(threshold=0.9)

        # No groups should be returned when no topics are similar
        assert groups == []

    @patch('libtrails.deduplication.bytes_to_embedding')
    @patch('libtrails.deduplication.get_topic_embeddings')
    @patch('libtrails.deduplication.get_db')
    def test_finds_duplicates(self, mock_db, mock_get_embeddings, mock_bytes_to_embedding):
        """Test finding similar topics."""
        # Two similar embeddings (1 and 2), one different (3)
        mock_get_embeddings.return_value = [
            (1, b'emb1'),
            (2, b'emb2'),
            (3, b'emb3'),
        ]
        mock_bytes_to_embedding.side_effect = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.99, 0.14, 0.0]),  # Similar to 1 (cosine > 0.9)
            np.array([0.0, 1.0, 0.0]),    # Different
        ]

        # Mock database for topic details
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Return topic details when queried
        mock_cursor.fetchone.side_effect = [
            (1, 'philosophy', 10),
            (2, 'Philosophy', 5),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        groups = find_duplicate_groups(threshold=0.9)

        # Should find one group with topics 1 and 2
        assert len(groups) == 1
        assert len(groups[0]) == 2

    @patch('libtrails.deduplication.bytes_to_embedding')
    @patch('libtrails.deduplication.get_topic_embeddings')
    @patch('libtrails.deduplication.get_db')
    def test_canonical_is_most_frequent(self, mock_db, mock_get_embeddings, mock_bytes_to_embedding):
        """Test that canonical topic has highest occurrence count."""
        mock_get_embeddings.return_value = [
            (1, b'emb1'),
            (2, b'emb2'),
        ]
        mock_bytes_to_embedding.side_effect = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.99, 0.14, 0.0]),  # Similar
        ]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Topic 2 has higher count (10 > 5)
        mock_cursor.fetchone.side_effect = [
            (1, 'topic1', 5),
            (2, 'topic2', 10),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        groups = find_duplicate_groups(threshold=0.9)

        # First topic in group should be the most frequent (canonical)
        assert len(groups) == 1
        assert groups[0][0]['occurrence_count'] == 10


class TestGetDeduplicationPreview:
    """Tests for deduplication preview."""

    @patch('libtrails.deduplication.find_duplicate_groups')
    def test_preview_returns_groups(self, mock_find_groups):
        """Test that preview returns duplicate groups."""
        mock_find_groups.return_value = [
            [
                {'id': 1, 'label': 'philosophy', 'occurrence_count': 10},
                {'id': 2, 'label': 'Philosophy', 'occurrence_count': 5},
            ]
        ]

        preview = get_deduplication_preview(threshold=0.9, limit=10)

        assert isinstance(preview, list)
        assert len(preview) == 1
        assert preview[0]['canonical'] == 'philosophy'
        assert len(preview[0]['duplicates']) == 1

    @patch('libtrails.deduplication.find_duplicate_groups')
    def test_preview_empty_when_no_duplicates(self, mock_find_groups):
        """Test preview returns empty when no duplicates."""
        mock_find_groups.return_value = []

        preview = get_deduplication_preview(threshold=0.9, limit=10)

        assert preview == []


class TestDeduplicateTopics:
    """Tests for actual deduplication."""

    @patch('libtrails.deduplication.find_duplicate_groups')
    @patch('libtrails.deduplication.get_db')
    def test_dry_run_no_changes(self, mock_db, mock_find_groups):
        """Test that dry run doesn't modify database."""
        mock_find_groups.return_value = [
            [
                {'id': 1, 'label': 'philosophy', 'occurrence_count': 10},
                {'id': 2, 'label': 'Philosophy', 'occurrence_count': 5},
            ]
        ]

        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        result = deduplicate_topics(threshold=0.9, dry_run=True)

        # Dry run should not execute any updates
        assert 'topics_merged' in result
        mock_conn.execute.assert_not_called()

    @patch('libtrails.deduplication.find_duplicate_groups')
    @patch('libtrails.deduplication.get_db')
    def test_returns_merge_stats(self, mock_db, mock_find_groups):
        """Test that deduplication returns statistics."""
        mock_find_groups.return_value = []

        mock_conn = MagicMock()
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        result = deduplicate_topics(threshold=0.9, dry_run=False)

        assert 'topics_merged' in result
        assert 'duplicate_groups' in result
