"""Multi-resolution Leiden CPM sweep with NMI stability and significance evaluation.

Finds natural community structure by sweeping CPM resolution and
scoring partitions with both NMI stability (adjacent partition similarity)
and Significance (statistical quality vs. random graph).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import igraph as ig
import leidenalg
import numpy as np

if TYPE_CHECKING:
    from rich.table import Table
from sklearn.metrics import normalized_mutual_info_score

from .config import (
    SWEEP_MIN_PLATEAU_LENGTH,
    SWEEP_N_RESOLUTIONS,
    SWEEP_PLATEAU_THRESHOLD,
    SWEEP_RESOLUTION_RANGE,
    SWEEP_SEED,
)


@dataclass
class PartitionStats:
    """Distribution statistics for a partition's cluster sizes."""

    max_size: int
    median_size: int
    mean_size: float
    p10_size: int
    p90_size: int
    singletons: int
    modularity_q: float


@dataclass
class SweepResult:
    """Result from a single Leiden run at one resolution."""

    resolution: float
    num_clusters: int
    quality: float
    membership: list[int]
    elapsed: float
    stats: PartitionStats | None = None


@dataclass
class Plateau:
    """A stable region in the resolution sweep."""

    start_resolution: float
    end_resolution: float
    start_index: int
    end_index: int
    mean_nmi: float
    length: int


@dataclass
class SweepSummary:
    """Complete results from a multi-resolution sweep."""

    results: list[SweepResult]
    nmi_scores: list[float]
    plateaus: list[Plateau]
    recommended_resolution: float | None
    recommended_index: int | None
    resolutions: list[float] = field(default_factory=list)
    significance_scores: list[float] = field(default_factory=list)


def compute_partition_stats(
    g: ig.Graph,
    membership: list[int],
) -> PartitionStats:
    """Compute distribution statistics and modularity Q for a partition."""
    from collections import Counter

    sizes = list(Counter(membership).values())
    sizes_arr = np.array(sizes)

    # Modularity Q via igraph (works on weighted graphs)
    modularity_q = g.modularity(
        membership, weights="weight" if "weight" in g.es.attributes() else None
    )

    return PartitionStats(
        max_size=int(sizes_arr.max()),
        median_size=int(np.median(sizes_arr)),
        mean_size=float(np.mean(sizes_arr)),
        p10_size=int(np.percentile(sizes_arr, 10)),
        p90_size=int(np.percentile(sizes_arr, 90)),
        singletons=int(np.sum(sizes_arr == 1)),
        modularity_q=float(modularity_q),
    )


def log_spaced_resolutions(
    low: float = SWEEP_RESOLUTION_RANGE[0],
    high: float = SWEEP_RESOLUTION_RANGE[1],
    n: int = SWEEP_N_RESOLUTIONS,
) -> list[float]:
    """Generate log-spaced resolution values for the sweep."""
    return list(np.logspace(np.log10(low), np.log10(high), n))


def leiden_at_resolution(
    g: ig.Graph,
    resolution: float,
    seed: int | None = SWEEP_SEED,
    weights: list[float] | None = None,
) -> SweepResult:
    """Run Leiden CPM at a single resolution.

    Args:
        g: The igraph Graph to cluster.
        resolution: CPM resolution parameter.
        seed: Random seed for determinism.
        weights: Edge weights (computed once, passed in).

    Returns:
        SweepResult with membership and quality.
    """
    start = time.time()
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights=weights,
        resolution_parameter=resolution,
        seed=seed,
    )
    elapsed = time.time() - start

    return SweepResult(
        resolution=resolution,
        num_clusters=len(set(partition.membership)),
        quality=partition.quality(),
        membership=list(partition.membership),
        elapsed=elapsed,
    )


def leiden_sweep(
    g: ig.Graph,
    resolutions: list[float] | None = None,
    seed: int | None = SWEEP_SEED,
    n_iterations: int = 1,
) -> list[SweepResult]:
    """Run Leiden at multiple resolutions on the same graph.

    Args:
        g: The igraph Graph to cluster.
        resolutions: List of resolution values to try.
        seed: Random seed for determinism.
        n_iterations: Number of runs per resolution (for robustness).
            When > 1, keeps the best quality run per resolution.

    Returns:
        List of SweepResult, one per resolution.
    """
    if resolutions is None:
        resolutions = log_spaced_resolutions()

    weights = g.es["weight"] if "weight" in g.es.attributes() else None
    results = []

    for res in resolutions:
        best = None
        for i in range(n_iterations):
            iter_seed = seed + i if seed is not None else None
            result = leiden_at_resolution(g, res, seed=iter_seed, weights=weights)
            if best is None or result.quality > best.quality:
                best = result
        best.stats = compute_partition_stats(g, best.membership)
        results.append(best)

    return results


def compute_stability(results: list[SweepResult]) -> list[float]:
    """Compute NMI between adjacent resolution partitions.

    Returns:
        List of NMI scores (length = len(results) - 1).
        nmi_scores[i] is the NMI between results[i] and results[i+1].
    """
    nmi_scores = []
    for i in range(len(results) - 1):
        nmi = normalized_mutual_info_score(
            results[i].membership,
            results[i + 1].membership,
        )
        nmi_scores.append(float(nmi))
    return nmi_scores


def compute_significance(
    g: ig.Graph,
    results: list[SweepResult],
) -> list[float]:
    """Score each partition using Significance on an unweighted graph.

    Significance measures statistical quality of community structure
    compared to a random graph. It's resolution-free — the peak indicates
    the most statistically significant partition.

    Only works on unweighted graphs, so we strip edge weights before scoring.

    Args:
        g: The original igraph Graph (may be weighted).
        results: Sweep results containing memberships to score.

    Returns:
        List of significance scores, one per result.
    """
    # Create unweighted copy for Significance scoring
    g_uw = g.copy()
    if "weight" in g_uw.es.attributes():
        del g_uw.es["weight"]

    scores = []
    for r in results:
        sig_part = leidenalg.SignificanceVertexPartition(g_uw, initial_membership=r.membership)
        scores.append(sig_part.quality())
    return scores


def find_stable_plateaus(
    nmi_scores: list[float],
    resolutions: list[float],
    threshold: float = SWEEP_PLATEAU_THRESHOLD,
    min_length: int = SWEEP_MIN_PLATEAU_LENGTH,
) -> list[Plateau]:
    """Find consecutive regions where NMI stays above threshold.

    Args:
        nmi_scores: NMI between adjacent partitions (len = len(resolutions) - 1).
        resolutions: Resolution values corresponding to sweep results.
        threshold: Minimum NMI to consider stable.
        min_length: Minimum number of consecutive NMI scores above threshold.

    Returns:
        List of Plateau objects, sorted by mean NMI descending.
    """
    plateaus = []
    i = 0

    while i < len(nmi_scores):
        if nmi_scores[i] >= threshold:
            start = i
            while i < len(nmi_scores) and nmi_scores[i] >= threshold:
                i += 1
            end = i  # exclusive index into nmi_scores

            length = end - start
            if length >= min_length:
                mean_nmi = float(np.mean(nmi_scores[start:end]))
                plateaus.append(
                    Plateau(
                        start_resolution=resolutions[start],
                        end_resolution=resolutions[end],  # end index in results
                        start_index=start,
                        end_index=end,
                        mean_nmi=mean_nmi,
                        length=length,
                    )
                )
        else:
            i += 1

    plateaus.sort(key=lambda p: p.mean_nmi, reverse=True)
    return plateaus


def recommend_resolution(
    results: list[SweepResult],
    nmi_scores: list[float],
    plateaus: list[Plateau],
    significance_scores: list[float] | None = None,
) -> tuple[float | None, int | None]:
    """Pick the best resolution from sweep results.

    Strategy (in priority order):
    1. If significance scores are available, use the resolution with peak significance.
    2. If plateaus exist, use midpoint of the best (highest mean NMI) plateau.
    3. Fall back to the resolution with highest pairwise NMI.

    Returns:
        (resolution, index) or (None, None) if results are empty.
    """
    if not results:
        return None, None

    # Significance peak is the most principled criterion
    if significance_scores:
        best_idx = int(np.argmax(significance_scores))
        return results[best_idx].resolution, best_idx

    if plateaus:
        best = plateaus[0]
        mid_index = (best.start_index + best.end_index) // 2
        return results[mid_index].resolution, mid_index

    # Fallback: resolution adjacent to the highest NMI score
    if nmi_scores:
        best_idx = int(np.argmax(nmi_scores))
        return results[best_idx].resolution, best_idx

    return results[0].resolution, 0


def run_sweep(
    g: ig.Graph,
    resolutions: list[float] | None = None,
    seed: int | None = SWEEP_SEED,
    n_iterations: int = 1,
    plateau_threshold: float = SWEEP_PLATEAU_THRESHOLD,
    min_plateau_length: int = SWEEP_MIN_PLATEAU_LENGTH,
) -> SweepSummary:
    """Main entry point: sweep resolutions and find stable partitions.

    Args:
        g: The igraph Graph.
        resolutions: Resolution values to try (default: log-spaced).
        seed: Random seed.
        n_iterations: Runs per resolution for robustness.
        plateau_threshold: NMI threshold for stability.
        min_plateau_length: Minimum consecutive stable NMI scores.

    Returns:
        SweepSummary with results, NMI, significance, plateaus, and recommendation.
    """
    if resolutions is None:
        resolutions = log_spaced_resolutions()

    results = leiden_sweep(g, resolutions=resolutions, seed=seed, n_iterations=n_iterations)
    nmi_scores = compute_stability(results)
    significance_scores = compute_significance(g, results)
    plateaus = find_stable_plateaus(
        nmi_scores,
        resolutions,
        threshold=plateau_threshold,
        min_length=min_plateau_length,
    )
    rec_res, rec_idx = recommend_resolution(results, nmi_scores, plateaus, significance_scores)

    return SweepSummary(
        results=results,
        nmi_scores=nmi_scores,
        plateaus=plateaus,
        recommended_resolution=rec_res,
        recommended_index=rec_idx,
        resolutions=resolutions,
        significance_scores=significance_scores,
    )


def format_sweep_table(summary: SweepSummary) -> "Table":
    """Build a Rich table showing sweep results.

    Columns: Resolution | Clusters | Quality | Significance | NMI | Markers
    """
    from rich.table import Table

    has_sig = bool(summary.significance_scores)

    has_stats = summary.results and summary.results[0].stats is not None

    table = Table(title="Leiden CPM Resolution Sweep")
    table.add_column("Resolution", justify="right", style="cyan", width=12)
    table.add_column("Clusters", justify="right", width=10)
    table.add_column("Q", justify="right", width=8)
    table.add_column("Quality", justify="right", width=12)
    if has_sig:
        table.add_column("Significance", justify="right", width=14)
    if has_stats:
        table.add_column("Max", justify="right", width=8)
        table.add_column("Median", justify="right", width=8)
        table.add_column("Singles", justify="right", width=8)
    table.add_column("NMI", justify="right", width=8)
    table.add_column("Time", justify="right", width=6)
    table.add_column("", width=16)  # markers

    # Build plateau membership set for marking
    plateau_indices = set()
    for p in summary.plateaus:
        for i in range(p.start_index, p.end_index + 1):
            plateau_indices.add(i)

    # Find peak significance index for marking
    peak_sig_idx = int(np.argmax(summary.significance_scores)) if has_sig else -1

    for i, r in enumerate(summary.results):
        nmi_str = f"{summary.nmi_scores[i]:.3f}" if i < len(summary.nmi_scores) else ""

        markers = []
        if i in plateau_indices:
            markers.append("[green]plateau[/green]")
        if i == summary.recommended_index:
            markers.append("[bold yellow]<< BEST[/bold yellow]")

        q_str = f"{r.stats.modularity_q:.4f}" if r.stats else ""
        row = [
            f"{r.resolution:.6f}",
            str(r.num_clusters),
            q_str,
            f"{r.quality:.2f}",
        ]
        if has_sig:
            sig_str = f"{summary.significance_scores[i]:.1f}"
            if i == peak_sig_idx:
                sig_str = f"[bold]{sig_str}[/bold]"
            row.append(sig_str)
        if has_stats and r.stats:
            row.extend(
                [
                    str(r.stats.max_size),
                    str(r.stats.median_size),
                    str(r.stats.singletons),
                ]
            )
        row.extend(
            [
                nmi_str,
                f"{r.elapsed:.0f}s",
                " ".join(markers),
            ]
        )
        table.add_row(*row)

    return table


def save_sweep_json(summary: SweepSummary, path: Path) -> None:
    """Export sweep results to JSON (excludes memberships for size)."""
    data = {
        "resolutions": [r.resolution for r in summary.results],
        "num_clusters": [r.num_clusters for r in summary.results],
        "quality": [r.quality for r in summary.results],
        "elapsed": [r.elapsed for r in summary.results],
        "nmi_scores": summary.nmi_scores,
        "significance_scores": summary.significance_scores,
        "plateaus": [
            {
                "start_resolution": p.start_resolution,
                "end_resolution": p.end_resolution,
                "mean_nmi": p.mean_nmi,
                "length": p.length,
            }
            for p in summary.plateaus
        ],
        "recommended_resolution": summary.recommended_resolution,
    }
    # Add partition distribution stats if available
    if summary.results and summary.results[0].stats is not None:
        data["modularity_q"] = [r.stats.modularity_q for r in summary.results]
        data["max_size"] = [r.stats.max_size for r in summary.results]
        data["median_size"] = [r.stats.median_size for r in summary.results]
        data["mean_size"] = [r.stats.mean_size for r in summary.results]
        data["p10_size"] = [r.stats.p10_size for r in summary.results]
        data["p90_size"] = [r.stats.p90_size for r in summary.results]
        data["singletons"] = [r.stats.singletons for r in summary.results]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
