"""Tests for multi-resolution Leiden CPM sweep."""

import igraph as ig
import leidenalg
import numpy as np
import pytest

from libtrails.sweep import (
    Plateau,
    SweepResult,
    SweepSummary,
    compute_stability,
    find_stable_plateaus,
    leiden_sweep,
    log_spaced_resolutions,
    recommend_resolution,
    run_sweep,
)


class TestLogSpacedResolutions:
    def test_basic_output(self):
        res = log_spaced_resolutions(0.001, 1.0, 10)
        assert len(res) == 10
        assert res[0] == pytest.approx(0.001, rel=1e-3)
        assert res[-1] == pytest.approx(1.0, rel=1e-3)

    def test_monotonically_increasing(self):
        res = log_spaced_resolutions(0.0001, 1.0, 20)
        for i in range(len(res) - 1):
            assert res[i] < res[i + 1]

    def test_single_point(self):
        res = log_spaced_resolutions(0.5, 0.5, 1)
        assert len(res) == 1
        assert res[0] == pytest.approx(0.5, rel=1e-3)

    def test_log_spacing(self):
        """Values should be log-spaced, not linearly spaced."""
        res = log_spaced_resolutions(0.001, 1.0, 4)
        # In log space, the ratios between consecutive values should be equal
        ratios = [res[i + 1] / res[i] for i in range(len(res) - 1)]
        for i in range(len(ratios) - 1):
            assert ratios[i] == pytest.approx(ratios[i + 1], rel=1e-3)


class TestComputeStability:
    def test_identical_memberships(self):
        """Identical partitions should have NMI = 1.0."""
        results = [
            SweepResult(0.1, 3, 1.0, [0, 0, 1, 1, 2, 2], 0.01),
            SweepResult(0.2, 3, 0.9, [0, 0, 1, 1, 2, 2], 0.01),
        ]
        nmi = compute_stability(results)
        assert len(nmi) == 1
        assert nmi[0] == pytest.approx(1.0)

    def test_different_memberships(self):
        """Completely different partitions should have low NMI."""
        results = [
            SweepResult(0.1, 3, 1.0, [0, 0, 0, 1, 1, 1], 0.01),
            SweepResult(0.2, 2, 0.9, [0, 1, 0, 1, 0, 1], 0.01),
        ]
        nmi = compute_stability(results)
        assert len(nmi) == 1
        assert nmi[0] < 0.5

    def test_output_length(self):
        results = [
            SweepResult(r, 3, 1.0, [0, 1, 2, 0, 1, 2], 0.01)
            for r in [0.1, 0.2, 0.3, 0.4]
        ]
        nmi = compute_stability(results)
        assert len(nmi) == 3  # n-1

    def test_empty_results(self):
        nmi = compute_stability([])
        assert nmi == []

    def test_single_result(self):
        results = [SweepResult(0.1, 3, 1.0, [0, 1, 2], 0.01)]
        nmi = compute_stability(results)
        assert nmi == []


class TestFindStablePlateaus:
    def test_single_plateau(self):
        nmi_scores = [0.95, 0.93, 0.91, 0.5, 0.3]
        resolutions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        plateaus = find_stable_plateaus(nmi_scores, resolutions, threshold=0.90, min_length=2)
        assert len(plateaus) == 1
        assert plateaus[0].length == 3
        assert plateaus[0].start_resolution == 0.01
        assert plateaus[0].end_resolution == 0.04

    def test_no_plateau(self):
        nmi_scores = [0.5, 0.4, 0.3]
        resolutions = [0.01, 0.02, 0.03, 0.04]
        plateaus = find_stable_plateaus(nmi_scores, resolutions, threshold=0.90, min_length=2)
        assert len(plateaus) == 0

    def test_too_short_plateau(self):
        """A single point above threshold shouldn't count as a plateau."""
        nmi_scores = [0.5, 0.95, 0.5]
        resolutions = [0.01, 0.02, 0.03, 0.04]
        plateaus = find_stable_plateaus(nmi_scores, resolutions, threshold=0.90, min_length=2)
        assert len(plateaus) == 0

    def test_multiple_plateaus(self):
        nmi_scores = [0.95, 0.93, 0.5, 0.3, 0.92, 0.94]
        resolutions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        plateaus = find_stable_plateaus(nmi_scores, resolutions, threshold=0.90, min_length=2)
        assert len(plateaus) == 2

    def test_sorted_by_mean_nmi(self):
        """Plateaus should be sorted by mean NMI descending."""
        nmi_scores = [0.91, 0.91, 0.5, 0.98, 0.97]
        resolutions = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
        plateaus = find_stable_plateaus(nmi_scores, resolutions, threshold=0.90, min_length=2)
        assert len(plateaus) == 2
        assert plateaus[0].mean_nmi >= plateaus[1].mean_nmi


class TestRecommendResolution:
    def test_with_plateau(self):
        results = [
            SweepResult(0.01, 100, 10.0, [0] * 100, 0.01),
            SweepResult(0.02, 90, 9.0, [0] * 100, 0.01),
            SweepResult(0.03, 80, 8.0, [0] * 100, 0.01),
            SweepResult(0.04, 50, 5.0, [0] * 100, 0.01),
        ]
        nmi_scores = [0.95, 0.93, 0.5]
        plateaus = [
            Plateau(
                start_resolution=0.01,
                end_resolution=0.03,
                start_index=0,
                end_index=2,
                mean_nmi=0.94,
                length=2,
            )
        ]
        res, idx = recommend_resolution(results, nmi_scores, plateaus)
        assert res == 0.02  # midpoint of plateau (index 1)
        assert idx == 1

    def test_no_plateau_falls_back_to_max_nmi(self):
        results = [
            SweepResult(0.01, 100, 10.0, [0] * 10, 0.01),
            SweepResult(0.02, 90, 9.0, [0] * 10, 0.01),
            SweepResult(0.03, 80, 8.0, [0] * 10, 0.01),
        ]
        nmi_scores = [0.7, 0.85]  # index 1 is highest
        plateaus = []
        res, idx = recommend_resolution(results, nmi_scores, plateaus)
        assert res == 0.02  # resolution at index of max NMI
        assert idx == 1

    def test_empty_results(self):
        res, idx = recommend_resolution([], [], [])
        assert res is None
        assert idx is None


def _make_test_graph(n_nodes: int = 30, n_communities: int = 3) -> ig.Graph:
    """Create a test graph with planted community structure."""
    # Create a planted partition graph
    nodes_per_community = n_nodes // n_communities
    g = ig.Graph(n=n_nodes)

    edges = []
    weights = []
    for c in range(n_communities):
        start = c * nodes_per_community
        end = start + nodes_per_community
        # Dense intra-community edges
        for i in range(start, end):
            for j in range(i + 1, end):
                edges.append((i, j))
                weights.append(1.0)

    # Sparse inter-community edges
    rng = np.random.RandomState(42)
    for c1 in range(n_communities):
        for c2 in range(c1 + 1, n_communities):
            start1 = c1 * nodes_per_community
            start2 = c2 * nodes_per_community
            # Just a few cross-edges
            for _ in range(2):
                i = rng.randint(start1, start1 + nodes_per_community)
                j = rng.randint(start2, start2 + nodes_per_community)
                edges.append((i, j))
                weights.append(0.1)

    g.add_edges(edges)
    g.es["weight"] = weights
    return g


class TestLeidenSweep:
    def test_basic_sweep(self):
        g = _make_test_graph(30, 3)
        resolutions = [0.001, 0.01, 0.1, 0.5, 1.0]
        results = leiden_sweep(g, resolutions=resolutions, seed=42)

        assert len(results) == 5
        for r in results:
            assert r.num_clusters >= 1
            assert r.elapsed >= 0
            assert len(r.membership) == 30

    def test_deterministic_with_seed(self):
        g = _make_test_graph(30, 3)
        resolutions = [0.01, 0.1]
        r1 = leiden_sweep(g, resolutions=resolutions, seed=42)
        r2 = leiden_sweep(g, resolutions=resolutions, seed=42)
        assert r1[0].membership == r2[0].membership
        assert r1[1].membership == r2[1].membership

    def test_more_clusters_at_higher_resolution(self):
        """Higher CPM resolution should generally produce more clusters."""
        g = _make_test_graph(60, 3)
        resolutions = [0.001, 1.0]
        results = leiden_sweep(g, resolutions=resolutions, seed=42)
        # At very low resolution, fewer or equal clusters than at high
        assert results[0].num_clusters <= results[1].num_clusters

    def test_multiple_iterations(self):
        g = _make_test_graph(30, 3)
        results = leiden_sweep(g, resolutions=[0.1], seed=42, n_iterations=3)
        assert len(results) == 1
        assert results[0].num_clusters >= 1


class TestRunSweep:
    def test_end_to_end(self):
        g = _make_test_graph(30, 3)
        summary = run_sweep(
            g,
            resolutions=[0.001, 0.01, 0.05, 0.1, 0.5],
            seed=42,
        )

        assert isinstance(summary, SweepSummary)
        assert len(summary.results) == 5
        assert len(summary.nmi_scores) == 4
        assert summary.recommended_resolution is not None
        assert summary.recommended_index is not None

    def test_sweep_with_planted_communities(self):
        """A graph with clear communities should produce stable plateaus."""
        g = _make_test_graph(60, 3)
        # Use a range that brackets the expected structure
        summary = run_sweep(
            g,
            resolutions=log_spaced_resolutions(0.001, 1.0, 15),
            seed=42,
            plateau_threshold=0.85,
            min_plateau_length=2,
        )

        # Should find the planted structure
        assert summary.recommended_resolution is not None
        # At the recommended resolution, should find ~3 communities
        rec_idx = summary.recommended_index
        assert summary.results[rec_idx].num_clusters >= 2
