"""Tests for domain generation, disparity filter, and outlier detection."""

import numpy as np
import pytest

import igraph as ig

from libtrails.domains import (
    compute_participation_coefficients,
    compute_robust_centroid,
    disparity_filter,
)


class TestDisparityFilter:
    """Tests for the disparity filter backbone extraction."""

    def test_empty_graph_unchanged(self):
        """Empty graph returns empty graph."""
        g = ig.Graph(n=3)
        g.vs["cluster_id"] = [0, 1, 2]
        result = disparity_filter(g, alpha=0.05)
        assert result.vcount() == 3
        assert result.ecount() == 0

    def test_single_edge_preserved(self):
        """A node's only edge is always preserved (degree=1 rule)."""
        g = ig.Graph(n=2, edges=[(0, 1)])
        g.vs["cluster_id"] = [0, 1]
        g.es["weight"] = [0.5]
        result = disparity_filter(g, alpha=0.01)  # very strict
        assert result.ecount() == 1

    def test_strong_edge_preserved(self):
        """A strong edge on a high-degree node is preserved."""
        # Star graph: node 0 connected to 1,2,3,4
        # Edge to 1 is very strong, others are weak
        g = ig.Graph(n=5, edges=[(0, 1), (0, 2), (0, 3), (0, 4)])
        g.vs["cluster_id"] = list(range(5))
        g.es["weight"] = [0.9, 0.01, 0.01, 0.01]
        result = disparity_filter(g, alpha=0.05)
        # Strong edge (0,1) should survive
        assert result.ecount() >= 1
        # Check that the strong edge is present
        strong_edges = [(e.source, e.target) for e in result.es if result.es[e.index]["weight"] > 0.5]
        assert len(strong_edges) >= 1

    def test_weak_edges_pruned_on_high_degree_nodes(self):
        """Weak edges on high-degree nodes are pruned more aggressively."""
        # Clique of 5 nodes, all edges equal weight
        g = ig.Graph.Full(5)
        g.vs["cluster_id"] = list(range(5))
        # All edges very weak
        g.es["weight"] = [0.01] * g.ecount()
        result = disparity_filter(g, alpha=0.01)  # very strict
        # With uniform weak weights on degree-4 nodes: alpha_ij = (1-0.01/total)^3
        # These should mostly be pruned
        assert result.ecount() < g.ecount()

    def test_vertex_attributes_preserved(self):
        """Vertex attributes are preserved in filtered graph."""
        g = ig.Graph(n=3, edges=[(0, 1), (1, 2)])
        g.vs["cluster_id"] = [10, 20, 30]
        g.es["weight"] = [0.9, 0.9]
        result = disparity_filter(g, alpha=0.05)
        assert result.vs["cluster_id"] == [10, 20, 30]

    def test_alpha_one_keeps_all_edges(self):
        """Alpha=1.0 should keep all edges (nothing is significant at alpha=1)."""
        # Actually, alpha_ij < alpha=1.0 is always true, so all edges kept
        g = ig.Graph(n=4, edges=[(0, 1), (1, 2), (2, 3)])
        g.vs["cluster_id"] = list(range(4))
        g.es["weight"] = [0.1, 0.1, 0.1]
        result = disparity_filter(g, alpha=1.0)
        assert result.ecount() == 3


class TestParticipationCoefficients:
    """Tests for participation coefficient computation."""

    def test_isolated_node(self):
        """Isolated node has participation=0, internal_frac=1."""
        g = ig.Graph(n=3, edges=[(0, 1)])
        g.vs["cluster_id"] = [0, 1, 2]
        membership = [0, 0, 1]
        coeffs = compute_participation_coefficients(g, membership)
        # Node 2 is isolated
        node2 = [c for c in coeffs if c["node_idx"] == 2][0]
        assert node2["participation"] == 0.0
        assert node2["internal_frac"] == 1.0

    def test_all_internal_edges(self):
        """Node with all edges in same community has participation=0."""
        g = ig.Graph(n=3, edges=[(0, 1), (0, 2)])
        g.vs["cluster_id"] = [0, 1, 2]
        membership = [0, 0, 0]  # all same community
        coeffs = compute_participation_coefficients(g, membership)
        node0 = [c for c in coeffs if c["node_idx"] == 0][0]
        assert node0["participation"] == 0.0
        assert node0["internal_frac"] == 1.0

    def test_bridge_node_high_participation(self):
        """Node with edges evenly split across communities has high participation."""
        # Node 0 connected to nodes in 2 different communities
        g = ig.Graph(n=3, edges=[(0, 1), (0, 2)])
        g.vs["cluster_id"] = [0, 1, 2]
        membership = [0, 1, 2]  # each in different community
        coeffs = compute_participation_coefficients(g, membership)
        node0 = [c for c in coeffs if c["node_idx"] == 0][0]
        # P = 1 - (1/2)^2 - (1/2)^2 = 0.5
        assert node0["participation"] == pytest.approx(0.5)
        assert node0["internal_frac"] == 0.0  # no edges to own community

    def test_sorted_by_participation_descending(self):
        """Results are sorted by participation descending."""
        g = ig.Graph(n=4, edges=[(0, 1), (0, 2), (0, 3), (1, 2)])
        g.vs["cluster_id"] = [0, 1, 2, 3]
        membership = [0, 1, 2, 3]
        coeffs = compute_participation_coefficients(g, membership)
        participations = [c["participation"] for c in coeffs]
        assert participations == sorted(participations, reverse=True)

    def test_three_way_bridge(self):
        """Node with edges to 3 communities has participation ~0.667."""
        g = ig.Graph(n=4, edges=[(0, 1), (0, 2), (0, 3)])
        g.vs["cluster_id"] = [0, 1, 2, 3]
        membership = [0, 1, 2, 3]
        coeffs = compute_participation_coefficients(g, membership)
        node0 = [c for c in coeffs if c["node_idx"] == 0][0]
        # P = 1 - 3*(1/3)^2 = 1 - 3/9 = 2/3
        assert node0["participation"] == pytest.approx(2.0 / 3.0)


class TestComputeRobustCentroid:
    """Tests for centroid computation with normalization."""

    def test_centroid_is_unit_norm(self):
        """Centroid should be L2 normalized."""
        topics = [
            {"label": "topic one", "embedding": np.array([1.0, 0.0, 0.0], dtype=np.float32), "occurrence_count": 10},
            {"label": "topic two", "embedding": np.array([0.0, 1.0, 0.0], dtype=np.float32), "occurrence_count": 5},
            {"label": "topic three", "embedding": np.array([0.0, 0.0, 1.0], dtype=np.float32), "occurrence_count": 3},
        ]
        centroid = compute_robust_centroid(topics)
        assert centroid is not None
        norm = np.linalg.norm(centroid)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_too_few_topics_returns_none(self):
        """Fewer than 3 topics returns None."""
        topics = [
            {"label": "topic one", "embedding": np.array([1.0, 0.0], dtype=np.float32), "occurrence_count": 10},
            {"label": "topic two", "embedding": np.array([0.0, 1.0], dtype=np.float32), "occurrence_count": 5},
        ]
        centroid = compute_robust_centroid(topics)
        assert centroid is None

    def test_short_labels_filtered(self):
        """Topics with short labels are filtered out."""
        topics = [
            {"label": "a", "embedding": np.array([1.0, 0.0], dtype=np.float32), "occurrence_count": 100},
            {"label": "be", "embedding": np.array([0.0, 1.0], dtype=np.float32), "occurrence_count": 50},
            {"label": "topic one", "embedding": np.array([1.0, 0.0], dtype=np.float32), "occurrence_count": 10},
            {"label": "topic two", "embedding": np.array([0.0, 1.0], dtype=np.float32), "occurrence_count": 5},
        ]
        # Only 2 valid topics after filtering (less than 3) → None
        centroid = compute_robust_centroid(topics)
        assert centroid is None
