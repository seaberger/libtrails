"""Tests for topic clustering functionality."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np


class TestClusterTopics:
    """Tests for Leiden clustering."""

    @patch('libtrails.clustering.update_topic_cluster')
    @patch('libtrails.clustering.leidenalg')
    @patch('libtrails.topic_graph.build_topic_graph_cooccurrence_only')
    def test_cluster_returns_stats_cooccurrence_mode(self, mock_build_graph, mock_leidenalg, mock_update_cluster):
        """Test that clustering returns statistics in cooccurrence mode."""
        from libtrails.clustering import cluster_topics
        import igraph as ig

        # Create a real graph with proper attributes
        mock_graph = ig.Graph()
        mock_graph.add_vertices(10)
        mock_graph.vs["topic_id"] = list(range(10))
        mock_graph.add_edges([(0, 1), (1, 2), (2, 3)])
        mock_graph.es["type"] = ["cooccurrence", "cooccurrence", "cooccurrence"]
        mock_build_graph.return_value = mock_graph

        # Mock partition
        mock_partition = MagicMock()
        mock_partition.membership = [0, 0, 1, 1, 2, 2, 2, 3, 3, 3]
        mock_partition.quality.return_value = 0.5
        mock_leidenalg.find_partition.return_value = mock_partition
        mock_leidenalg.ModularityVertexPartition = MagicMock()

        result = cluster_topics(mode="cooccurrence")

        assert 'num_clusters' in result
        assert 'total_topics' in result
        assert 'modularity' in result
        assert result['mode'] == 'cooccurrence'

    @patch('libtrails.clustering.update_topic_cluster')
    @patch('libtrails.clustering.leidenalg')
    @patch('libtrails.topic_graph.build_topic_graph_cooccurrence_only')
    def test_cluster_with_partition_types(self, mock_build_graph, mock_leidenalg, mock_update_cluster):
        """Test clustering with different partition types."""
        from libtrails.clustering import cluster_topics
        import igraph as ig

        mock_graph = ig.Graph()
        mock_graph.add_vertices(5)
        mock_graph.vs["topic_id"] = list(range(5))
        mock_graph.add_edges([(0, 1), (1, 2)])
        mock_graph.es["type"] = ["cooccurrence", "cooccurrence"]
        mock_build_graph.return_value = mock_graph

        mock_partition = MagicMock()
        mock_partition.membership = [0, 0, 1, 1, 1]
        mock_partition.quality.return_value = 0.4
        mock_leidenalg.find_partition.return_value = mock_partition
        mock_leidenalg.ModularityVertexPartition = MagicMock()
        mock_leidenalg.SurpriseVertexPartition = MagicMock()
        mock_leidenalg.CPMVertexPartition = MagicMock()

        # Test modularity (default)
        result = cluster_topics(partition_type="modularity")
        assert result['partition_type'] == 'modularity'

        # Test surprise
        result = cluster_topics(partition_type="surprise")
        assert result['partition_type'] == 'surprise'

        # Test CPM with resolution
        result = cluster_topics(partition_type="cpm", resolution=0.5)
        assert result['partition_type'] == 'cpm'

    @patch('libtrails.clustering.update_topic_cluster')
    @patch('libtrails.clustering.leidenalg')
    @patch('libtrails.topic_graph.build_topic_graph_knn')
    def test_cluster_with_knn_mode(self, mock_build_graph, mock_leidenalg, mock_update_cluster):
        """Test clustering with k-NN mode."""
        from libtrails.clustering import cluster_topics
        import igraph as ig

        mock_graph = ig.Graph()
        mock_graph.add_vertices(5)
        mock_graph.vs["topic_id"] = list(range(5))
        mock_graph.add_edges([(0, 1), (1, 2), (2, 3)])
        mock_graph.es["type"] = ["embedding_knn", "embedding_knn", "cooccurrence"]
        mock_build_graph.return_value = mock_graph

        mock_partition = MagicMock()
        mock_partition.membership = [0, 0, 1, 1, 1]
        mock_partition.quality.return_value = 0.4
        mock_leidenalg.find_partition.return_value = mock_partition
        mock_leidenalg.ModularityVertexPartition = MagicMock()

        result = cluster_topics(mode="knn", knn_k=10)

        assert result['mode'] == 'knn'
        assert 'edge_types' in result
        assert 'embedding_knn' in result['edge_types']

    @patch('libtrails.topic_graph.build_topic_graph_cooccurrence_only')
    def test_handles_empty_graph_cooccurrence(self, mock_build_graph):
        """Test handling of empty graph in cooccurrence mode."""
        from libtrails.clustering import cluster_topics
        import igraph as ig

        mock_graph = ig.Graph()  # Empty graph
        mock_build_graph.return_value = mock_graph

        result = cluster_topics(mode="cooccurrence")

        assert 'error' in result

    def test_invalid_mode_returns_error(self):
        """Test that invalid mode returns an error."""
        from libtrails.clustering import cluster_topics

        result = cluster_topics(mode="invalid_mode")

        assert 'error' in result
        assert 'Unknown mode' in result['error']

    @patch('libtrails.topic_graph.build_topic_graph_cooccurrence_only')
    def test_invalid_partition_type_returns_error(self, mock_build_graph):
        """Test that invalid partition type returns an error."""
        from libtrails.clustering import cluster_topics
        import igraph as ig

        mock_graph = ig.Graph()
        mock_graph.add_vertices(5)
        mock_graph.vs["topic_id"] = list(range(5))
        mock_graph.add_edges([(0, 1)])
        mock_graph.es["type"] = ["cooccurrence"]
        mock_build_graph.return_value = mock_graph

        result = cluster_topics(partition_type="invalid_partition")

        assert 'error' in result
        assert 'Unknown partition type' in result['error']

    @patch('libtrails.clustering.update_topic_cluster')
    @patch('libtrails.clustering.leidenalg')
    @patch('libtrails.topic_graph.build_topic_graph_cooccurrence_only')
    def test_cluster_with_custom_cooccurrence_min(self, mock_build_graph, mock_leidenalg, mock_update_cluster):
        """Test clustering with custom cooccurrence minimum."""
        from libtrails.clustering import cluster_topics
        import igraph as ig

        mock_graph = ig.Graph()
        mock_graph.add_vertices(5)
        mock_graph.vs["topic_id"] = list(range(5))
        mock_graph.add_edges([(0, 1)])
        mock_graph.es["type"] = ["cooccurrence"]
        mock_build_graph.return_value = mock_graph

        mock_partition = MagicMock()
        mock_partition.membership = [0, 0, 1, 1, 1]
        mock_partition.quality.return_value = 0.4
        mock_leidenalg.find_partition.return_value = mock_partition
        mock_leidenalg.ModularityVertexPartition = MagicMock()

        result = cluster_topics(cooccurrence_min=10)
        # Verify the custom value was passed
        mock_build_graph.assert_called_once()
        call_args = mock_build_graph.call_args
        assert call_args.kwargs.get('cooccurrence_min') == 10


class TestGetClusterSummary:
    """Tests for cluster summary retrieval."""

    @patch('libtrails.clustering.get_db')
    def test_returns_cluster_list(self, mock_db):
        """Test that summary returns list of clusters."""
        from libtrails.clustering import get_cluster_summary

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # First call: get clusters with sizes
        # Second call (per cluster): get top topics
        mock_cursor.fetchall.side_effect = [
            [(0, 5), (1, 3), (2, 2)],  # cluster_id, count
            [('philosophy', 10), ('ethics', 5)],  # top topics for cluster 0
            [('science', 8)],  # top topics for cluster 1
            [('art', 3)],  # top topics for cluster 2
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        summary = get_cluster_summary()

        assert isinstance(summary, list)

    @patch('libtrails.clustering.get_db')
    def test_empty_when_no_clusters(self, mock_db):
        """Test returns empty when no clusters exist."""
        from libtrails.clustering import get_cluster_summary

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        summary = get_cluster_summary()

        assert summary == []


class TestGetTopicTree:
    """Tests for topic tree retrieval."""

    @patch('libtrails.clustering.get_cluster_summary')
    def test_returns_tree_structure(self, mock_summary):
        """Test that tree has correct structure."""
        from libtrails.clustering import get_topic_tree

        # Mock return must match actual get_cluster_summary format
        # which returns {'cluster_id', 'size', 'top_topics': [{'label': ..., 'count': ...}]}
        mock_summary.return_value = [
            {'cluster_id': 0, 'size': 5, 'top_topics': [{'label': 'topic1', 'count': 10}]},
            {'cluster_id': 1, 'size': 3, 'top_topics': [{'label': 'topic2', 'count': 5}]},
        ]

        tree = get_topic_tree()

        assert 'children' in tree
        assert len(tree['children']) == 2

    @patch('libtrails.clustering.get_cluster_summary')
    def test_empty_tree_when_no_clusters(self, mock_summary):
        """Test empty tree when no clusters."""
        from libtrails.clustering import get_topic_tree

        mock_summary.return_value = []

        tree = get_topic_tree()

        assert tree.get('children', []) == []


class TestGetClusterTopics:
    """Tests for getting topics in a cluster."""

    @patch('libtrails.clustering.get_db')
    def test_returns_topics_for_cluster(self, mock_db):
        """Test returning topics for a specific cluster."""
        from libtrails.clustering import get_cluster_topics

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        # Return tuples that get converted to dicts via list comprehension
        mock_cursor.fetchall.return_value = [
            (1, 'philosophy', 10),
            (2, 'ethics', 5),
        ]
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        topics = get_cluster_topics(0)

        assert len(topics) == 2
        assert topics[0]['label'] == 'philosophy'

    @patch('libtrails.clustering.get_db')
    def test_empty_for_nonexistent_cluster(self, mock_db):
        """Test empty result for nonexistent cluster."""
        from libtrails.clustering import get_cluster_topics

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.cursor.return_value = mock_cursor
        mock_db.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_db.return_value.__exit__ = MagicMock(return_value=False)

        topics = get_cluster_topics(999)

        assert topics == []
