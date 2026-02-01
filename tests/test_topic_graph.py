"""Tests for topic graph construction."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestComputeCooccurrences:
    """Tests for computing topic co-occurrences."""

    @patch('libtrails.topic_graph.get_db')
    def test_returns_stats(self, mock_db):
        """Test that co-occurrence computation returns statistics."""
        from libtrails.topic_graph import compute_cooccurrences

        mock_conn = MagicMock()
        mock_cursor = MagicMock()

        # Mock fetching chunk topic pairs
        mock_cursor.fetchall.side_effect = [
            # First call: get all chunk-topic pairs
            [(1, 1), (1, 2), (2, 1), (2, 3)],
            # Second call: existing cooccurrences (empty)
            [],
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        result = compute_cooccurrences()

        assert 'cooccurrence_pairs' in result

    @patch('libtrails.topic_graph.get_db')
    def test_handles_empty_data(self, mock_db):
        """Test handling when no chunk topics exist."""
        from libtrails.topic_graph import compute_cooccurrences

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        result = compute_cooccurrences()

        assert result['cooccurrence_pairs'] == 0


class TestBuildTopicGraph:
    """Tests for building the topic graph."""

    @patch('libtrails.topic_graph.get_topic_embeddings')
    @patch('libtrails.topic_graph.get_all_topics')
    @patch('libtrails.topic_graph.get_db')
    def test_returns_igraph(self, mock_db, mock_get_topics, mock_get_embeddings):
        """Test that build returns an igraph Graph."""
        from libtrails.topic_graph import build_topic_graph
        import igraph as ig

        mock_get_topics.return_value = [
            {'id': 1, 'label': 'topic1', 'occurrence_count': 10, 'cluster_id': None},
            {'id': 2, 'label': 'topic2', 'occurrence_count': 5, 'cluster_id': None},
        ]

        mock_get_embeddings.return_value = [
            (1, np.array([1.0, 0.0, 0.0]).astype(np.float32).tobytes()),
            (2, np.array([0.0, 1.0, 0.0]).astype(np.float32).tobytes()),
        ]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        graph = build_topic_graph()

        assert isinstance(graph, ig.Graph)
        assert graph.vcount() == 2

    @patch('libtrails.topic_graph.get_topic_embeddings')
    @patch('libtrails.topic_graph.get_all_topics')
    @patch('libtrails.topic_graph.get_db')
    def test_adds_similarity_edges(self, mock_db, mock_get_topics, mock_get_embeddings):
        """Test that similar topics get edges."""
        from libtrails.topic_graph import build_topic_graph

        mock_get_topics.return_value = [
            {'id': 1, 'label': 'topic1', 'occurrence_count': 10, 'cluster_id': None},
            {'id': 2, 'label': 'topic2', 'occurrence_count': 5, 'cluster_id': None},
        ]

        # Very similar embeddings (cosine similarity > 0.5)
        mock_get_embeddings.return_value = [
            (1, np.array([1.0, 0.0, 0.0]).astype(np.float32).tobytes()),
            (2, np.array([0.99, 0.1, 0.0]).astype(np.float32).tobytes()),
        ]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        # Note: actual parameter is embedding_threshold, not similarity_threshold
        graph = build_topic_graph(embedding_threshold=0.5)

        # Should have an edge between the similar topics
        assert graph.ecount() >= 1


class TestGetRelatedTopics:
    """Tests for finding related topics."""

    @patch('libtrails.topic_graph.build_topic_graph')
    def test_finds_neighbors(self, mock_build_graph):
        """Test finding topics related via graph."""
        from libtrails.topic_graph import get_related_topics
        import igraph as ig

        # Create a simple graph with all required attributes
        g = ig.Graph()
        g.add_vertices(3)
        g.vs["label"] = ["philosophy", "ethics", "science"]
        g.vs["topic_id"] = [1, 2, 3]
        g.vs["occurrence_count"] = [10, 5, 3]  # Required attribute
        g.add_edges([(0, 1)])  # philosophy connected to ethics
        g.es["weight"] = [0.8]
        g.es["type"] = ["similarity"]

        mock_build_graph.return_value = g

        results = get_related_topics("philosophy", limit=5)

        assert len(results) >= 1
        labels = [r['label'] for r in results]
        assert 'ethics' in labels

    @patch('libtrails.topic_graph.build_topic_graph')
    def test_not_found_returns_empty(self, mock_build_graph):
        """Test that nonexistent topic returns empty."""
        from libtrails.topic_graph import get_related_topics
        import igraph as ig

        g = ig.Graph()
        g.add_vertices(2)
        g.vs["label"] = ["philosophy", "ethics"]
        g.vs["topic_id"] = [1, 2]
        g.vs["occurrence_count"] = [10, 5]
        mock_build_graph.return_value = g

        results = get_related_topics("nonexistent", limit=5)

        assert results == []

    @patch('libtrails.topic_graph.build_topic_graph')
    def test_partial_match(self, mock_build_graph):
        """Test partial matching of topic names."""
        from libtrails.topic_graph import get_related_topics
        import igraph as ig

        g = ig.Graph()
        g.add_vertices(2)
        g.vs["label"] = ["self discovery", "personal growth"]
        g.vs["topic_id"] = [1, 2]
        g.vs["occurrence_count"] = [10, 5]  # Required attribute
        g.add_edges([(0, 1)])
        g.es["weight"] = [0.7]
        g.es["type"] = ["cooccurrence"]

        mock_build_graph.return_value = g

        # Search with partial match
        results = get_related_topics("discovery", limit=5)

        assert len(results) >= 1


class TestGetGraphStats:
    """Tests for graph statistics.

    Note: get_graph_stats(g: ig.Graph) requires a graph argument.
    """

    def test_returns_stats(self):
        """Test that stats include expected metrics."""
        from libtrails.topic_graph import get_graph_stats
        import igraph as ig

        g = ig.Graph()
        g.add_vertices(5)
        g.add_edges([(0, 1), (1, 2), (2, 3)])
        g.es["type"] = ["embedding", "cooccurrence", "embedding"]

        # Pass graph as argument
        stats = get_graph_stats(g)

        assert 'nodes' in stats
        assert 'edges' in stats
        assert stats['nodes'] == 5
        assert stats['edges'] == 3

    def test_empty_graph_stats(self):
        """Test stats for empty graph."""
        from libtrails.topic_graph import get_graph_stats
        import igraph as ig

        g = ig.Graph()

        # Pass graph as argument
        stats = get_graph_stats(g)

        assert stats['nodes'] == 0
        assert stats['edges'] == 0
