"""Experiment: resolution_profile() + Significance-guided CPM on demo library.

Run with:
    LIBTRAILS_DB=demo CALIBRE_LIBRARY_PATH=~/Calibre_Demo_Library uv run python experiments/resolution_experiment.py
"""

import sys
import time

import igraph as ig
import leidenalg as la
import numpy as np


def build_graph():
    """Build the topic graph (same as clustering.py knn mode)."""
    from libtrails.config import COOCCURRENCE_MIN_COUNT, CLUSTER_KNN_K, PMI_MIN_THRESHOLD
    from libtrails.topic_graph import build_topic_graph_knn

    print("Building topic graph (knn mode)...")
    t0 = time.time()
    g = build_topic_graph_knn(
        cooccurrence_min=COOCCURRENCE_MIN_COUNT,
        pmi_min=PMI_MIN_THRESHOLD,
        k=CLUSTER_KNN_K,
    )
    print(f"  {g.vcount()} nodes, {g.ecount()} edges in {time.time() - t0:.1f}s")
    return g


def experiment_resolution_profile(g: ig.Graph):
    """Use leidenalg's bisection-based resolution_profile to find all change points."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: resolution_profile() — bisection change points")
    print("=" * 70)

    weights = g.es["weight"] if "weight" in g.es.attributes() else None

    opt = la.Optimiser()
    print("Running resolution_profile (bisection)... this may take a while.")
    t0 = time.time()

    profile = opt.resolution_profile(
        g,
        la.CPMVertexPartition,
        resolution_range=(0.0001, 1.0),
        weights="weight" if weights else None,
        min_diff_resolution=0.1,  # log-scale diff; wider to keep it tractable
        number_iterations=-1,  # run until stable
    )

    elapsed = time.time() - t0
    print(f"  Found {len(profile)} distinct partitions in {elapsed:.1f}s\n")

    print(f"{'Resolution':>12}  {'Clusters':>8}  {'Quality':>12}")
    print("-" * 36)
    for p in profile:
        res = p.resolution_parameter
        n_clusters = len(set(p.membership))
        quality = p.quality()
        print(f"{res:>12.6f}  {n_clusters:>8}  {quality:>12.1f}")

    return profile


def experiment_significance_scoring(g: ig.Graph, profile=None):
    """Score CPM partitions using Significance on an unweighted graph."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Significance-guided CPM scoring")
    print("=" * 70)

    # Create unweighted version for Significance scoring
    g_uw = g.copy()
    if "weight" in g_uw.es.attributes():
        del g_uw.es["weight"]

    # Build resolutions to test: mix of log-spaced + profile change points
    resolutions = sorted(set(
        list(np.logspace(-4, 0, 30))
        + ([p.resolution_parameter for p in profile] if profile else [])
    ))

    weights = g.es["weight"] if "weight" in g.es.attributes() else None

    print(f"Testing {len(resolutions)} resolutions...")
    print(f"\n{'Resolution':>12}  {'Clusters':>8}  {'CPM Quality':>12}  {'Significance':>14}")
    print("-" * 52)

    best_sig = -float("inf")
    best_res = None
    best_n = None
    results = []

    for res in resolutions:
        # Run CPM on weighted graph
        part = la.find_partition(
            g,
            la.CPMVertexPartition,
            weights=weights,
            resolution_parameter=res,
            seed=42,
        )
        n_clusters = len(set(part.membership))
        cpm_quality = part.quality()

        # Score with Significance on unweighted graph
        sig_part = la.SignificanceVertexPartition(g_uw, initial_membership=part.membership)
        sig_score = sig_part.quality()

        results.append({
            "resolution": res,
            "n_clusters": n_clusters,
            "cpm_quality": cpm_quality,
            "significance": sig_score,
        })

        marker = ""
        if sig_score > best_sig:
            best_sig = sig_score
            best_res = res
            best_n = n_clusters
            marker = " *"

        print(f"{res:>12.6f}  {n_clusters:>8}  {cpm_quality:>12.1f}  {sig_score:>14.1f}{marker}")

    print(f"\nBest significance: {best_sig:.1f} at resolution {best_res:.6f} ({best_n} clusters)")

    # Also show top 5 by significance
    results.sort(key=lambda r: r["significance"], reverse=True)
    print("\nTop 5 resolutions by Significance:")
    for r in results[:5]:
        print(f"  res={r['resolution']:.6f}  clusters={r['n_clusters']}  sig={r['significance']:.1f}")

    return results


def main():
    g = build_graph()

    # Experiment 1: resolution_profile
    profile = experiment_resolution_profile(g)

    # Experiment 2: Significance scoring
    results = experiment_significance_scoring(g, profile)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Graph: {g.vcount()} nodes, {g.ecount()} edges")
    print(f"Global density: {2 * g.ecount() / (g.vcount() * (g.vcount() - 1)):.6f}")
    print(f"resolution_profile found {len(profile)} distinct partitions")
    best = max(results, key=lambda r: r["significance"])
    print(f"Best significance: res={best['resolution']:.6f}, {best['n_clusters']} clusters")
    print(f"Old default (0.001): ~295 clusters")


if __name__ == "__main__":
    main()
