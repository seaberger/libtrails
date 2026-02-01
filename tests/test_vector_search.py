"""Tests for vector search functionality."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestVectorSearchSetup:
    """Tests for vector search initialization.

    Note: sqlite-vec is imported inside functions, so we test at a higher level.
    """

    def test_module_imports(self):
        """Test that vector_search module can be imported."""
        from libtrails import vector_search

        assert hasattr(vector_search, 'search_topics_semantic')
        assert hasattr(vector_search, 'search_books_by_topic_semantic')
        assert hasattr(vector_search, 'rebuild_vector_index')


class TestSearchTopicsSemantic:
    """Tests for semantic topic search."""

    @patch('libtrails.vector_search.embedding_to_bytes')
    @patch('libtrails.vector_search.embed_text')
    @patch('libtrails.vector_search.get_vec_db')
    def test_search_returns_results(self, mock_get_db, mock_embed, mock_to_bytes):
        """Test that search returns topic results."""
        from libtrails.vector_search import search_topics_semantic

        # Mock embedding
        mock_embed.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_to_bytes.return_value = b'\x00\x01\x02'

        # Mock database results with dict-like rows
        mock_row1 = {"topic_id": 1, "distance": 0.1, "label": "philosophy",
                     "occurrence_count": 10, "cluster_id": 0}
        mock_row2 = {"topic_id": 2, "distance": 0.2, "label": "ethics",
                     "occurrence_count": 5, "cluster_id": 0}

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [mock_row1, mock_row2]
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn

        results = search_topics_semantic("test query", limit=10)

        assert len(results) == 2
        assert results[0]['label'] == 'philosophy'
        mock_conn.close.assert_called()

    @patch('libtrails.vector_search.embedding_to_bytes')
    @patch('libtrails.vector_search.embed_text')
    @patch('libtrails.vector_search.get_vec_db')
    def test_search_empty_results(self, mock_get_db, mock_embed, mock_to_bytes):
        """Test search with no results."""
        from libtrails.vector_search import search_topics_semantic

        mock_embed.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_to_bytes.return_value = b'\x00\x01\x02'

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn

        results = search_topics_semantic("obscure query", limit=10)

        assert results == []


class TestSearchBooksByTopic:
    """Tests for searching books by topic."""

    @patch('libtrails.vector_search.embedding_to_bytes')
    @patch('libtrails.vector_search.embed_text')
    @patch('libtrails.vector_search.get_vec_db')
    def test_search_books_calls_semantic_search(self, mock_get_db, mock_embed, mock_to_bytes):
        """Test that book search uses semantic topic search."""
        from libtrails.vector_search import search_books_by_topic_semantic

        mock_embed.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        mock_to_bytes.return_value = b'\x00\x01\x02'

        # Mock database for topic search
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Return empty results to simplify test
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_get_db.return_value = mock_conn

        results = search_books_by_topic_semantic("philosophy", limit=10)

        # Should return empty list when no topics found
        assert isinstance(results, list)


class TestRebuildVectorIndex:
    """Tests for rebuilding vector index."""

    @patch('libtrails.vector_search.init_vector_search')
    def test_rebuild_returns_count(self, mock_init):
        """Test that rebuild returns indexed count."""
        from libtrails.vector_search import rebuild_vector_index

        # Mock connection with topics table
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [
            {'id': 1, 'embedding': b'\x00\x01\x02\x03'},
            {'id': 2, 'embedding': b'\x04\x05\x06\x07'},
        ]
        mock_conn.cursor.return_value = mock_cursor

        count = rebuild_vector_index(mock_conn, force_recreate=False)

        assert isinstance(count, int)
        assert count == 2

    @patch('libtrails.vector_search.init_vector_search')
    def test_force_recreate_calls_init(self, mock_init):
        """Test that force_recreate calls init with force_recreate=True."""
        from libtrails.vector_search import rebuild_vector_index

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor

        rebuild_vector_index(mock_conn, force_recreate=True)

        # Should have called init_vector_search with force_recreate=True
        mock_init.assert_called_with(mock_conn, force_recreate=True)
