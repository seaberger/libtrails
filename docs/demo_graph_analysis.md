# Demo Library Graph Analysis

Analysis of the topic graph built from the demo library (100 books, ~31,849 chunks).

## Graph Construction

- **Method**: KNN (k=10 embedding neighbors + co-occurrence edges)
- **Date**: Feb 16, 2026
- **Database**: `data/demo_library.db`

## Basic Topology

| Metric | Value |
|--------|-------|
| Vertices (n) | 121,118 |
| Edges (m) | 913,896 |
| Density (p) | 0.000125 |
| Avg degree ⟨k⟩ | 15.09 |
| Connected components | 18 |
| Giant component | 121,101 (100.0%) |

## Community Structure: Erdős-Rényi Comparison

Following Girvan and Newman (2002), we compare the clustering coefficient of the real graph to the expected value for a random graph G(n, p) with the same size and density.

| Metric | Value |
|--------|-------|
| Global clustering C (actual) | 0.2456 |
| Avg local clustering C (actual) | 0.2728 |
| Expected C for ER G(n, p) | 0.000125 |
| **Ratio C_actual / C_random** | **1,971x** |

The graph has ~2,000x higher clustering than an Erdős-Rényi random graph of the same size — clear community structure.

## Degree Distribution

| Metric | Value |
|--------|-------|
| Min degree | 0 |
| Median degree | 13 |
| Mean degree | 15.1 |
| P95 degree | 27 |
| P99 degree | 38 |
| Max degree | 95 |
| Std dev | 6.1 |
| Skewness | 2.54 |

Expected std for ER: 3.9 — actual is 1.6x higher. Moderately heterogeneous but not strongly power-law. The KNN construction (k=10) caps embedding neighbors per node, constraining the tail. The right-skew comes from co-occurrence edges.

## Modularity (Leiden Partition)

| Metric | Value |
|--------|-------|
| Clusters | 2,468 |
| Modularity Q | 0.2282 |

Below the conventional Q > 0.3 threshold. Likely due to the very fine-grained partition (2,468 clusters over 121K nodes). A coarser resolution would score higher. Modularity also has a known resolution limit that penalizes many small communities.

---

## Experiment 1: KNN k-Value Sweep (Feb 16, 2026)

How does the number of embedding neighbors (k) affect graph topology and partition quality?

### Method

Built 7 graphs with k = {3, 5, 7, 10, 20, 30, 50}, holding all other parameters constant (cooccurrence_min=5, pmi_min=0.0, min_similarity=0.3). Measured topology metrics and modularity of the existing Surprise partition against each graph.

### Results

| k | Edges | Avg Degree | C_global | C/C_random | Modularity Q | Components | Giant% | Max Degree |
|---|-------|-----------|----------|------------|-------------|-----------|--------|-----------|
| **3** | 281,303 | 4.6 | 0.2069 | **5,394x** | **0.2694** | 50 | 99.8% | 45 |
| 5 | 462,767 | 7.6 | 0.2315 | 3,669x | 0.2539 | 18 | 100.0% | 62 |
| 7 | 643,477 | 10.6 | 0.2405 | 2,742x | 0.2421 | 18 | 100.0% | 76 |
| 10 | 913,896 | 15.1 | 0.2456 | 1,971x | 0.2282 | 18 | 100.0% | 95 |
| 20 | 1,811,701 | 29.9 | 0.2482 | 1,005x | 0.1935 | 18 | 100.0% | 238 |
| 30 | 2,704,063 | 44.7 | 0.2462 | 668x | 0.1702 | 18 | 100.0% | 366 |
| 50 | 4,480,022 | 74.0 | 0.2405 | 394x | 0.1417 | 18 | 100.0% | 603 |

### Observations

1. **Modularity increases monotonically as k decreases** — k=3 achieves the highest Q=0.2694. Sparser graphs are more cleanly separable.

2. **Clustering coefficient varies modestly** — C_global ranges from 0.207 (k=3) to 0.248 (k=20). The triangle density is largely a property of the embedding space, but very sparse graphs (k=3) have fewer triangles.

3. **k=3 fragments the graph slightly** — 50 components (vs 18 at k>=5), though the giant component still holds 99.8% of nodes. The 32 extra fragments are negligible.

4. **Co-occurrence edges are negligible** — Only 501 co-occurrence edges at every k vs 281K–4.5M KNN edges. The graph is almost entirely determined by embedding similarity.

5. **Build time is stable** — ~105–112s per graph regardless of k. The sklearn KNN computation dominates; edge insertion is fast.

### Conclusion

**k=3 gives the highest modularity** (Q=0.2694), but the tradeoff is lower clustering coefficient and more disconnected fragments. For topic-level Leiden clustering where within-cluster richness matters, **k=5–10** provides a good balance. For macro-level super-cluster analysis, **k=3 may produce cleaner domain boundaries** and is worth experimenting with (see Future Experiments below).

---

## Experiment 2: Gamma (CPM Resolution) Sensitivity (Feb 16, 2026)

How does the CPM resolution parameter (gamma) affect partition granularity and quality? Compared at k=10 (original) and k=5 (sparser graph).

### Method

Ran Leiden with `CPMVertexPartition` at 25 log-spaced gamma values from 0.0001 to 1.0 (seed=42 for reproducibility). Measured cluster count, modularity Q, max cluster size, median cluster size, and singleton count.

### Results: k=10

| gamma | Clusters | Q | Max Size | Median | Singletons |
|-------|---------|------|---------|--------|-----------|
| 0.00010 | 72 | **0.7530** | 7,927 | 601 | 18 |
| 0.00015 | 91 | 0.7488 | 4,817 | 965 | 17 |
| 0.00022 | 135 | 0.7422 | 3,848 | 663 | 17 |
| 0.00032 | 179 | 0.7351 | 3,050 | 531 | 17 |
| 0.00046 | 239 | 0.7261 | 2,675 | 385 | 17 |
| 0.00068 | 321 | 0.7155 | 2,023 | 309 | 17 |
| 0.00100 | 441 | 0.7025 | 1,357 | 220 | 20 |
| 0.00147 | 589 | 0.6909 | 846 | 169 | 19 |
| 0.00215 | 788 | 0.6784 | 699 | 123 | 21 |
| 0.00316 | 1,038 | 0.6651 | 636 | 96 | 18 |
| 0.00464 | 1,344 | 0.6509 | 404 | 77 | 23 |
| 0.00681 | 1,735 | 0.6370 | 329 | 58 | 27 |
| **0.01000** | **2,262** | **0.6201** | **263** | **45** | **26** |
| 0.01468 | 2,871 | 0.6032 | 215 | 36 | 28 |
| 0.02154 | 3,722 | 0.5821 | 142 | 28 | 32 |
| 0.03162 | 4,768 | 0.5596 | 135 | 21 | 38 |
| 0.04642 | 6,128 | 0.5345 | 97 | 17 | 47 |
| 0.06813 | 7,952 | 0.5057 | 78 | 13 | 101 |
| 0.10000 | 10,441 | 0.4717 | 55 | 10 | 231 |
| 0.14678 | 13,711 | 0.4331 | 44 | 7 | 412 |
| 0.21544 | 18,103 | 0.3881 | 31 | 5 | 829 |
| 0.31623 | 23,563 | 0.3365 | 24 | 4 | 1,726 |
| 0.46416 | 31,761 | 0.2752 | 17 | 3 | 4,183 |
| 0.68129 | 38,634 | 0.2124 | 13 | 3 | 6,524 |
| 1.00000 | 120,868 | 0.0005 | 13 | 1 | 120,788 |

### Results: k=5

| gamma | Clusters | Q | Max Size | Median | Singletons |
|-------|---------|------|---------|--------|-----------|
| 0.00010 | 157 | **0.7925** | 3,380 | 605 | 17 |
| 0.00015 | 208 | 0.7877 | 2,618 | 442 | 17 |
| 0.00022 | 288 | 0.7806 | 2,084 | 330 | 17 |
| 0.00032 | 380 | 0.7714 | 1,401 | 271 | 17 |
| 0.00046 | 500 | 0.7642 | 1,115 | 202 | 17 |
| 0.00068 | 676 | 0.7553 | 774 | 151 | 17 |
| 0.00100 | 884 | 0.7460 | 563 | 116 | 17 |
| 0.00147 | 1,164 | 0.7354 | 489 | 91 | 17 |
| 0.00215 | 1,518 | 0.7254 | 378 | 71 | 17 |
| 0.00316 | 1,944 | 0.7142 | 292 | 54 | 17 |
| 0.00464 | 2,484 | 0.7026 | 233 | 43 | 17 |
| **0.00681** | **3,151** | **0.6893** | **187** | **34** | **18** |
| 0.01000 | 4,006 | 0.6751 | 141 | 27 | 18 |
| 0.01468 | 5,059 | 0.6580 | 104 | 21 | 20 |
| 0.02154 | 6,344 | 0.6401 | 79 | 17 | 18 |
| 0.03162 | 7,941 | 0.6187 | 60 | 13 | 20 |
| 0.04642 | 9,944 | 0.5942 | 49 | 11 | 26 |
| 0.06813 | 12,554 | 0.5662 | 35 | 9 | 59 |
| 0.10000 | 15,974 | 0.5317 | 28 | 7 | 136 |
| 0.14678 | 20,493 | 0.4911 | 24 | 5 | 351 |
| 0.21544 | 26,143 | 0.4446 | 21 | 4 | 1,234 |
| 0.31623 | 32,754 | 0.3918 | 19 | 3 | 2,886 |
| 0.46416 | 43,035 | 0.3240 | 13 | 2 | 8,118 |
| 0.68129 | 49,064 | 0.2701 | 13 | 2 | 11,577 |
| 1.00000 | 120,868 | 0.0010 | 13 | 1 | 120,788 |

### Comparison: k=5 vs k=10

| Metric | k=10 | k=5 |
|--------|------|-----|
| Peak Q | 0.7530 | **0.7925** |
| Clusters at peak Q | 72 | 157 |
| Q at ~2,500 clusters | 0.620 (gamma=0.01) | **0.703** (gamma=0.0046) |
| Q > 0.3 range | gamma < 0.32 | gamma < **0.46** |
| Singleton-free range | gamma < 0.007 | gamma < **0.007** |

k=5 produces higher modularity at every gamma value, with a wider range of significant partitions (Q > 0.3). The sparser graph has cleaner community boundaries.

### Observations

1. **k=5 dominates k=10 at every resolution** — Peak Q=0.793 (k=5) vs 0.753 (k=10). The sparser graph has sharper community boundaries, which holds across all gamma values.

2. **k=5 finds more communities at the same gamma** — At gamma=0.0001, k=5 finds 157 clusters vs k=10's 72. The finer-grained graph resolves more structure at the macro level.

3. **Current Surprise partition (~2,500 clusters) corresponds to gamma ≈ 0.005 at k=5** — With k=5 at gamma=0.0046, we get 2,484 clusters at Q=0.703. This is substantially higher than k=10's Q=0.620 at the same cluster count.

4. **No stability plateau at either k** — Cluster count increases continuously with gamma. This is characteristic of hierarchical community structure: there is no single "right" resolution, just different valid scales.

5. **Singleton explosion above gamma=0.1** — Consistent across both k values. CPM with gamma=1.0 essentially finds only cliques.

6. **Hierarchical structure is clear** — The graph supports meaningful partitions at multiple scales: ~70–157 macro-communities, ~2,500 meso-communities, and ~10,000+ micro-communities. This is ideal for the domain → cluster → topic hierarchy used in LibTrails.

---

## Experiment 3: Choosing the Right Number of Clusters (Feb 16, 2026)

Modularity Q alone is insufficient for choosing a partition — it rewards very coarse groupings. A practical partition must balance statistical quality with usability.

### Interpreting the Metrics

| Metric | What it measures | What "good" looks like |
|--------|-----------------|----------------------|
| **Q (modularity)** | How well-separated clusters are vs a random null model | > 0.3 is significant; higher is better, but peaks at very coarse resolutions |
| **max_size** | The largest cluster | < ~300 topics — no single cluster should be unmanageable for browsing or labeling |
| **median** | The typical cluster size | 20–80 topics — enough to see patterns, few enough to scan |
| **singletons** | Topics that ended up alone | < 1% of total topics — a few genuine outliers are fine, thousands means over-fragmentation |
| **mean / p10 / p90** | Distribution shape | Balanced clusters; large spread between p10 and p90 indicates heterogeneity |

Q has a known resolution limit (Fortunato & Barthélemy, 2007): it cannot detect communities smaller than ~sqrt(2m). For our k=5 graph (m=462K), that's ~960 nodes. This means Q is structurally biased toward partitions with fewer than ~120 clusters, explaining why peak Q always occurs at the coarsest gamma.

### Fine-Grained Sweep: k=5 Sweet Spot (gamma 0.004–0.009)

| gamma | Clusters | Q | Max Size | Median | Singletons | Mean | P10 | P90 |
|-------|---------|------|---------|--------|-----------|------|-----|-----|
| 0.0040 | 2,268 | 0.7068 | 228 | 47 | 17 | 53 | 22 | 92 |
| 0.0046 | 2,484 | 0.7026 | 233 | 43 | 17 | — | — | — |
| **0.0050** | **2,620** | **0.7001** | **200** | **41** | **17** | **46** | **18** | **81** |
| 0.0060 | 2,921 | 0.6937 | 192 | 36 | 18 | 41 | 17 | 72 |
| 0.0068 | 3,151 | 0.6893 | 187 | 34 | 18 | — | — | — |
| 0.0070 | 3,223 | 0.6884 | 178 | 33 | 18 | 38 | 16 | 65 |
| 0.0080 | 3,514 | 0.6830 | 138 | 31 | 19 | 34 | 15 | 59 |
| 0.0090 | 3,753 | 0.6787 | 141 | 29 | 18 | 32 | 14 | 55 |

### Recommended Operating Point

**gamma ≈ 0.005 at k=5** (~2,500–2,600 clusters) is the sweet spot for the demo library:

- **Q = 0.700** — well above the 0.3 significance threshold
- **max_size = 200** — no mega-clusters; largest group is browsable
- **median = 41** — typical cluster has ~40 topics, easy to scan and label
- **singletons = 17** — negligible (0.01%), essentially all topics find a home
- **p10=18, p90=81** — reasonable spread; even the small clusters are meaningful

This aligns closely with where the current Surprise partition lands (~2,468 clusters), suggesting Surprise's implicit resolution is well-calibrated for this dataset. Switching to CPM at gamma=0.005 would achieve similar granularity with the advantage of being an explicit, tunable parameter.

### Scaling Considerations

For the full V2 library (837K topics, ~7x the demo), maintaining a median cluster size of ~40 topics would require ~20,000 clusters. The gamma value that produces this will differ because the graph density and edge weight distribution change with scale. The gamma sweep should be repeated on V2 data.

---

## Future Experiments

- **k=3 for super-cluster/domain analysis**: k=3 produced the highest modularity (Q=0.269) with the existing Surprise partition. Worth testing whether a k=3 graph with CPM at gamma ≈ 0.0001 produces cleaner domain boundaries than the current K-means-on-cluster-centroids approach. This could simplify the domain pipeline by deriving macro-communities directly from the graph rather than as a post-processing step.

- **Reproduce on V2 dataset**: Once the V2 graph build completes (~837K topics), repeat the k-sweep and gamma analysis to verify findings generalize beyond the demo library.

- **GPU-accelerated KNN (#49)**: Use FAISS-GPU on RTX 3090 to reduce graph build time from ~90 min to seconds for V2-scale datasets, enabling rapid iteration on k and gamma parameters.

## References

- Erdős, P. and Rényi, A. (1959). "On random graphs." *Publicationes Mathematicae*, 6, 290–297.
- Girvan, M. and Newman, M.E.J. (2002). "Community structure in social and biological networks." *Proceedings of the National Academy of Sciences*, 99(12), 7821–7826.
