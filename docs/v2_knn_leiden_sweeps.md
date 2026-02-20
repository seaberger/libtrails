# V2 KNN & Leiden Sweep Results

## Database: `ipad_library_v2.db`
- **837,320 topics** with embeddings (384-dim BGE-small)
- **937 books**, 335K chunks
- **2M cooccurrence pairs**
- Previous clustering (sklearn k=10): 3,330 clusters (median 188, mean 251)

---

## Phase 1: KNN k Sweep

**Goal**: Find optimal graph density. Higher k = more edges = denser graph = fewer, larger clusters.

**Fixed parameters**:
- Resolution: 0.001 (CPM)
- min_similarity: 0.65
- Cooccurrences: skip (use existing 2M pairs)

| k | GPU time | Graph edges | Clusters | Largest | Leiden time | Memory | Quality |
|---|----------|-------------|----------|---------|-------------|--------|---------|
| 3 | 34.5s | 1.34M | 16,262 | 264 | 16.9s | 1.5 GB | 2,002,907 |
| 4 | 34.9s | 1.98M | 10,965 | 405 | 24.3s | 1.9 GB | 2,513,463 |
| 5 | 35.4s | 2.62M | 8,236 | 594 | 29.8s | 2.2 GB | 3,029,940 |
| **6** | **35.5s** | **3.25M** | **6,534** | **748** | **31.8s** | **2.5 GB** | **3,545,096** |
| 8 | 36.5s | 4.52M | 4,728 | 1,344 | 43.6s | 3.0 GB | 4,561,491 |
| 11 | 37.8s | 6.42M | 3,343 | 2,141 | 57.4s | 4.3 GB | 6,068,982 |
| 15 | 39.3s | 8.96M | 2,451 | 2,409 | 75.1s | 5.2 GB | 8,042,370 |
| 21 | 41.7s | 12.75M | 1,775 | 3,233 | 98.3s | 7.8 GB | 10,926,421 |
| 31 | 45.3s | 19.08M | 1,237 | 4,644 | 135.4s | 10.0 GB | 15,621,224 |

### Observations

- **k=3-5**: Very sparse graphs → too many small clusters (8K-16K). Topics fragment.
- **k=6-8**: Sweet spot zone. k=6 gives 6,534 clusters (largest 748) with manageable memory.
- **k=11**: Matches the previous V2 clustering (~3,330 clusters). This was the baseline.
- **k=15+**: Clusters get too large (2,400+ topics). Top clusters become mega-blobs.
- **k=31**: Only 1,237 clusters but largest has 4,644 topics — too coarse for navigation.
- **GPU time**: Nearly constant (~35-45s) regardless of k. The bottleneck is upload (~50-65s).
- **Memory**: Scales linearly with k. k=31 uses 10 GB — approaching MacBook limits.

### Recommendation

**k=6** looks optimal for V2:
- 6,534 clusters — enough granularity for a 936-book library
- Largest cluster (748) is manageable — not a mega-blob
- Fast: 32s Leiden, 2.5 GB memory
- ~2x more clusters than k=11, which may improve domain generation

**Next**: Run Leiden resolution sweep at k=6 to fine-tune cluster count.

---

## Phase 2: Leiden Resolution Sweep

**Goal**: Fine-tune cluster granularity at k=6. Higher resolution = more, smaller clusters.

**Fixed parameters**:
- k=6 (from Phase 1)
- min_similarity: 0.65
- Cooccurrences: skip (use existing 2M pairs)
- Graph: cached (837,320 nodes, 3,252,347 edges)

### Automated Sweep (30 log-spaced resolutions, 0.0001–0.01)

| Resolution | Clusters | Leiden time | Quality |
|-----------|----------|-------------|---------|
| 0.000100 | 1,083 | 36.2s | 3,898,969 |
| 0.000117 | 1,266 | 37.4s | 3,875,828 |
| 0.000137 | 1,430 | 36.8s | 3,853,942 |
| 0.000161 | 1,626 | 36.9s | 3,832,192 |
| 0.000189 | 1,856 | 36.7s | 3,810,105 |
| 0.000221 | 2,111 | 36.8s | 3,786,889 |
| 0.000259 | 2,398 | 36.4s | 3,764,593 |
| 0.000304 | 2,730 | 36.9s | 3,740,372 |
| 0.000356 | 3,007 | 35.4s | 3,715,902 |
| 0.000418 | 3,405 | 34.7s | 3,690,795 |
| 0.000489 | 3,870 | 34.6s | 3,664,982 |
| 0.000574 | 4,356 | 34.8s | 3,638,889 |
| 0.000672 | 4,911 | 34.2s | 3,611,710 |
| 0.000788 | 5,524 | 34.0s | 3,586,110 |
| **0.000924** | **6,234** | **33.9s** | **3,558,363** |
| 0.001083 | 6,967 | 34.1s | 3,531,204 |
| 0.001269 | 7,783 | 34.9s | 3,502,208 |
| 0.001487 | 8,725 | 36.2s | 3,472,793 |
| 0.001743 | 9,732 | 35.8s | 3,442,119 |
| 0.002043 | 10,865 | 34.9s | 3,410,139 |
| 0.002395 | 12,112 | 31.5s | 3,377,038 |
| 0.002807 | 13,528 | 30.8s | 3,342,981 |
| 0.003290 | 15,061 | 30.8s | 3,308,836 |
| 0.003857 | 16,797 | 30.8s | 3,272,389 |
| 0.004520 | 18,575 | 30.3s | 3,234,984 |
| 0.005298 | 20,595 | 29.5s | 3,196,223 |
| 0.006210 | 22,794 | 29.0s | 3,155,692 |
| 0.007279 | 25,296 | 28.6s | 3,113,472 |
| 0.008532 | 27,927 | 28.7s | 3,069,177 |
| 0.010000 | 30,886 | 28.7s | 3,022,192 |

NMI plateau starts at resolution 0.000924 (NMI=0.923, length=15). Total sweep time: 1,027s.

### Sweet Spot Sweep: Partition Distribution Metrics (0.0005–0.003)

10 log-spaced resolutions with full distribution stats (modularity Q, max/median/mean size, P10/P90, singletons).

| gamma | Clusters | Q | Max | Median | Mean | P10 | P90 | Singles |
|-------|---------|------|-----|--------|------|-----|-----|---------|
| 0.000500 | 3,944 | 0.6701 | 1,709 | 171 | 212 | 79 | 401 | 13 |
| 0.000610 | 4,585 | 0.6649 | 1,301 | 147 | 183 | 71 | 337 | 13 |
| 0.000745 | 5,316 | 0.6600 | 1,233 | 129 | 158 | 62 | 287 | 13 |
| **0.000909** | **6,138** | **0.6549** | **903** | **113** | **136** | **56** | **243** | **13** |
| **0.001109** | **7,087** | **0.6500** | **698** | **98** | **118** | **50** | **209** | **13** |
| 0.001353 | 8,169 | 0.6450 | 624 | 86 | 103 | 44 | 178 | 13 |
| 0.001651 | 9,339 | 0.6398 | 520 | 75 | 90 | 40 | 156 | 13 |
| 0.002015 | 10,767 | 0.6342 | 457 | 65 | 78 | 35 | 135 | 13 |
| 0.002458 | 12,339 | 0.6287 | 400 | 57 | 68 | 31 | 118 | 13 |
| 0.003000 | 14,152 | 0.6227 | 357 | 50 | 59 | 28 | 101 | 14 |

NMI plateau starts at resolution 0.001109 (NMI=0.911, length=5). Singletons essentially constant (13–14) across all resolutions.

### Manual Resolution Probes (with top-5 cluster sizes)

| Resolution | Clusters | Top 5 cluster sizes |
|-----------|----------|---------------------|
| 0.0001 | 1,091 | 4,766 / 4,451 / 4,209 / 3,974 / 3,557 |
| 0.0003 | 2,663 | 2,218 / 2,018 / 1,951 / 1,867 / 1,845 |
| 0.0005 | 3,940 | 1,704 / 1,338 / 1,302 / 1,299 / 1,261 |
| 0.0008 | 5,614 | 1,092 / 949 / 895 / 876 / 799 |
| 0.001 | 6,534 | 748 (from Phase 1 at this resolution) |
| 0.002 | 10,680 | 466 / 457 / 457 / 436 / 435 |
| 0.003 | 14,158 | 375 / 344 / 342 / 330 / 326 |
| 0.005 | 19,900 | 248 / 235 / 228 / 226 / 218 |

### Observations

- **Resolution vs. clusters is nearly linear on a log-log scale**: every ~3x increase in resolution roughly doubles the cluster count.
- **Modularity Q decreases slowly** across the sweet spot — from 0.670 to 0.623. All values are well above the 0.3 significance threshold. Q is not a strong differentiator here.
- **NMI plateau starts at ~0.001**: From this point up, the partitioning is "stable" (similar structure just more fragmented). Consistent across both the 30-point and 10-point sweeps.
- **Singletons are negligible** — only 13–14 across the entire range. Virtually all topics find a home. (Compare to the demo library where singletons exploded above gamma=0.1.)
- **Largest cluster size drops fast**: From 1,709 at gamma=0.0005 to 698 at gamma=0.001, then to 357 at gamma=0.003.
- **Median cluster size**: 171 at gamma=0.0005 → 98 at gamma=0.001 → 50 at gamma=0.003.
- **Leiden time is nearly constant** (~33-37s) regardless of resolution.

### Analysis: Choosing the Resolution

Following the same criteria from the demo library analysis (Experiment 3):

| Metric | What "good" looks like | gamma=0.0009 | gamma=0.001 | gamma=0.0015 |
|--------|----------------------|-------------|-------------|-------------|
| **Q (modularity)** | > 0.3 significant | 0.655 | 0.650 | 0.640 |
| **max_size** | < ~1,000 (browsable) | 903 | ~700 | 520 |
| **median** | 50–150 (scannable) | 113 | ~100 | 75 |
| **singletons** | < 1% | 13 (0.002%) | 13 | 13 |
| **P10 / P90** | Balanced spread | 56 / 243 | ~50 / ~210 | 40 / 156 |

**Recommendation**: **Resolution 0.001** (CPM) at k=6.
- **~6,500 clusters** for 837K topics (~128 topics/cluster average)
- **Q = 0.650** — well above significance threshold
- **Max cluster ~700** — no mega-blobs, largest cluster is browsable
- **Median ~100** — typical cluster has ~100 topics, easy to scan and label
- **Singletons = 13** — negligible (0.002%)
- **P10=50, P90=210** — reasonable spread; even small clusters are meaningful
- Right at the NMI plateau onset, meaning it captures meaningful structure at the point where further splitting becomes structurally redundant
- ~2x more clusters than V1 baseline (3,330) — better granularity for the domain/super-cluster layer
- Closely matches the demo library sweet spot (gamma=0.005 at k=5, which similarly landed at the NMI plateau onset with Q=0.700, median=41, max=200)

---

## Phase 3: Domain (Super-Cluster) Hyperparameter Optimization

**Goal**: Group the 6,575 Leiden clusters into ~25-30 browsable high-level domains using Leiden CPM on a cluster-level k-NN graph. Two hyperparameters: graph density (k) and Leiden resolution (gamma).

**Method**: Build a k-NN graph over Leiden cluster centroids (384-dim BGE embeddings), then run Leiden CPM to find macro-communities. This is more principled than K-means because it respects graph topology — clusters that co-occur and have strong KNN connections get grouped together, rather than just being geometrically close in embedding space. The demo library analysis (`docs/demo_graph_analysis.md`) suggested k=3 with gamma ≈ 0.0001 as a starting point, but V2 scale (6,562 clusters vs ~300 in demo) required its own sweep.

**Fixed parameters**:
- Cluster graph nodes: 6,562 (clusters with valid centroids)
- min_similarity: 0.3 (for cluster graph edges)
- Outlier reassignment: enabled (participation coefficient > 0.7)

---

### 3a. Cluster Graph k Sweep

**Goal**: Find optimal graph density for the cluster-level graph. Tested k = 2, 3, 4, 5, 6, 8, 10.

| k | Edges | Density |
|---|-------|---------|
| 2 | 11,027 | 0.0005 |
| 3 | 16,287 | 0.0008 |
| 4 | 21,550 | 0.0010 |
| 5 | 26,746 | 0.0012 |
| 6 | 31,985 | 0.0015 |
| 8 | 42,443 | 0.0020 |
| 10 | 52,905 | 0.0025 |

**Initial comparison at ~25 domains** (selecting gamma per k to hit closest to 25):

| k | gamma | Domains | Q | Max | Median | P10 | P90 |
|---|-------|---------|------|------|--------|-----|-----|
| 2 | 0.000300 | 38 | 0.810 | 1,179 | 123 | 4 | 309 |
| **3** | **0.000516** | **24** | **0.764** | **1,269** | **202** | **105** | **367** |
| **4** | **0.000676** | **24** | **0.735** | **1,375** | **238** | **96** | **335** |
| 5 | 0.001163 | 25 | 0.718 | 1,227 | 214 | 83 | 412 |
| 6 | 0.001332 | 25 | 0.702 | 1,362 | 172 | 52 | 523 |
| 8 | 0.001747 | 23 | 0.668 | 1,455 | 216 | 77 | 378 |
| 10 | 0.002000 | 20 | 0.661 | 1,433 | 249 | 69 | 660 |

**Key observations**:
- **k=2** has highest Q (0.81) but P10=4 — some domains have only 4-5 clusters, barely viable
- **k=3-4** sweet spot: Q ≈ 0.73-0.76, P10 > 90, reasonable max sizes
- **k≥6** too dense: Q drops below 0.70, low gammas produce mega-domains (3,000-6,000 clusters)
- Eliminated k=2 (sparse fragmentation), k≥5 (too dense). Focused on k=3 and k=4.

---

### 3b. Full Resolution Sweep: k=2, 3, 4

25 log-spaced resolutions from 0.0001 to 0.005 for each k value.

#### Domain Count

| gamma | k=2 | k=3 | k=4 |
|-------|-----|-----|-----|
| 0.000100 | 14 | 7 | 3 |
| 0.000118 | 19 | 6 | 4 |
| 0.000139 | 23 | 8 | 5 |
| 0.000163 | 21 | 7 | 6 |
| 0.000192 | 25 | 11 | 7 |
| 0.000226 | 30 | 12 | 7 |
| 0.000266 | 33 | 15 | 10 |
| 0.000313 | 39 | 18 | 11 |
| 0.000368 | 39 | 20 | 14 |
| 0.000434 | 46 | 21 | 17 |
| 0.000510 | 52 | 24 | 16 |
| 0.000601 | 60 | 29 | 21 |
| 0.000707 | 70 | 32 | 24 |
| 0.000832 | 74 | 41 | **27** |
| 0.000980 | 86 | 45 | 29 |
| 0.001153 | 89 | 53 | 35 |
| 0.001357 | 101 | 61 | 40 |
| 0.001597 | 116 | 67 | 49 |
| 0.001880 | 131 | 75 | 57 |
| 0.002213 | 146 | 86 | 62 |
| 0.002605 | 166 | 100 | 67 |
| 0.003066 | 180 | 115 | 81 |
| 0.003609 | 204 | 127 | 95 |
| 0.004248 | 227 | 140 | 108 |
| 0.005000 | 251 | 160 | 124 |

#### Modularity Q

| gamma | k=2 | k=3 | k=4 |
|-------|-----|-----|-----|
| 0.000100 | 0.761 | 0.307 | 0.068 |
| 0.000118 | 0.781 | 0.535 | 0.071 |
| 0.000139 | 0.767 | 0.577 | 0.445 |
| 0.000163 | 0.784 | 0.657 | 0.508 |
| 0.000192 | 0.794 | 0.706 | 0.615 |
| 0.000226 | 0.804 | 0.726 | 0.641 |
| 0.000266 | 0.808 | 0.747 | 0.696 |
| 0.000313 | 0.807 | 0.751 | 0.702 |
| 0.000368 | 0.812 | 0.756 | 0.711 |
| 0.000434 | 0.821 | 0.758 | 0.722 |
| 0.000510 | 0.821 | 0.756 | 0.726 |
| 0.000601 | 0.820 | 0.770 | 0.736 |
| 0.000707 | 0.817 | 0.770 | 0.738 |
| 0.000832 | 0.818 | 0.773 | **0.738** |
| 0.000980 | 0.813 | 0.773 | 0.740 |
| 0.001153 | 0.810 | 0.768 | 0.737 |
| 0.001357 | 0.809 | 0.765 | 0.741 |
| 0.001597 | 0.799 | 0.764 | 0.735 |
| 0.001880 | 0.794 | 0.758 | 0.728 |
| 0.002213 | 0.790 | 0.749 | 0.728 |
| 0.002605 | 0.780 | 0.742 | 0.725 |
| 0.003066 | 0.778 | 0.735 | 0.714 |
| 0.003609 | 0.772 | 0.728 | 0.710 |
| 0.004248 | 0.764 | 0.721 | 0.695 |
| 0.005000 | 0.758 | 0.710 | 0.684 |

#### Max Domain Size (clusters)

| gamma | k=2 | k=3 | k=4 |
|-------|-----|-----|-----|
| 0.000100 | 1,628 | 5,222 | 6,304 |
| 0.000118 | 1,532 | 3,293 | 6,292 |
| 0.000139 | 1,722 | 2,915 | 3,863 |
| 0.000163 | 1,537 | 2,304 | 3,045 |
| 0.000192 | 1,431 | 2,134 | 2,373 |
| 0.000226 | 1,300 | 1,835 | 2,352 |
| 0.000266 | 1,183 | 1,639 | 1,877 |
| 0.000313 | 1,191 | 1,611 | 1,751 |
| 0.000368 | 1,078 | 1,538 | 1,784 |
| 0.000434 | 574 | 1,504 | 1,690 |
| 0.000510 | 557 | 1,490 | 1,578 |
| 0.000601 | 563 | 1,163 | 1,406 |
| 0.000707 | 509 | 1,111 | 1,317 |
| 0.000832 | 513 | 682 | **1,185** |
| 0.000980 | 428 | 692 | 1,116 |
| 0.001153 | 497 | 673 | 1,002 |
| 0.001357 | 410 | 621 | 676 |
| 0.001597 | 373 | 513 | 584 |
| 0.001880 | 222 | 521 | 631 |
| 0.002213 | 252 | 374 | 511 |
| 0.002605 | 171 | 346 | 462 |
| 0.003066 | 166 | 324 | 399 |
| 0.003609 | 193 | 270 | 396 |
| 0.004248 | 123 | 243 | 299 |
| 0.005000 | 92 | 210 | 262 |

#### Median Domain Size

| gamma | k=2 | k=3 | k=4 |
|-------|-----|-----|-----|
| 0.000100 | 412 | 223 | 253 |
| 0.000192 | 192 | 442 | 426 |
| 0.000313 | 119 | 222 | 353 |
| 0.000510 | 102 | 191 | 293 |
| 0.000707 | 81 | 163 | 180 |
| 0.000832 | 73 | 113 | **180** |
| 0.000980 | 65 | 106 | 182 |
| 0.001357 | 54 | 87 | 114 |
| 0.001880 | 43 | 65 | 84 |
| 0.002605 | 34 | 53 | 73 |
| 0.005000 | 24 | 38 | 43 |

#### Significance

| gamma | k=2 | k=3 | k=4 |
|-------|-----|-----|-----|
| 0.000100 | 11,561 | 5,227 | 1,746 |
| 0.000313 | 19,927 | 21,052 | 20,335 |
| 0.000601 | 24,625 | 26,177 | 29,148 |
| 0.000832 | 26,127 | 30,052 | **31,834** |
| 0.001153 | 27,822 | 32,530 | 35,128 |
| 0.001880 | 31,031 | 35,567 | 40,751 |
| 0.005000 | 35,468 | 41,910 | 48,476 |

Significance climbs monotonically (no peak in this range) — it favors finer partitions. Not useful as a selection criterion for domains; used only to confirm all configs are statistically significant.

#### Singletons

Zero across all k values and gammas. Every Leiden cluster finds a domain home.

---

### 3c. Seed Stability Analysis

**Goal**: Test robustness — do the same hyperparameters produce the same partition across random seeds?

Ran 30 random seeds (42–71) at 8 candidate configurations. For each config, computed all pairwise NMI scores (435 pairs).

| Config | Mean domains | Std | Range | Mean NMI | Min NMI | Max NMI |
|--------|-------------|-----|-------|----------|---------|---------|
| k=3, γ=0.0006 | 29.7 | 1.3 | 26–33 | 0.772 | 0.720 | 0.819 |
| k=3, γ=0.0007 | 34.1 | 1.3 | 32–36 | 0.787 | 0.735 | 0.842 |
| k=3, γ=0.0008 | 38.2 | 1.7 | 35–42 | 0.793 | 0.736 | 0.845 |
| k=4, γ=0.0006 | 20.5 | 0.7 | 19–22 | 0.785 | 0.717 | 0.848 |
| k=4, γ=0.0007 | 23.0 | 0.9 | 21–25 | 0.782 | 0.717 | 0.854 |
| **k=4, γ=0.0008** | **25.3** | **0.9** | **24–27** | **0.783** | **0.720** | **0.833** |
| k=4, γ=0.0009 | 28.0 | 1.2 | 24–30 | 0.785 | 0.741 | 0.837 |
| k=4, γ=0.0010 | 30.7 | 1.1 | 29–33 | 0.789 | 0.741 | 0.836 |

**Key findings**:
- **Stability is comparable across all configs** — pairwise NMI ≈ 0.77–0.79 everywhere. No config is dramatically more stable. Domain boundaries have ~20% variability across seeds, which is normal for community detection at this coarse scale.
- **k=4 has lower variance in domain count** (std 0.7–0.9) vs k=3 (std 1.3–1.7). The denser graph anchors the partition better.
- **Cross-k NMI ≈ 0.68–0.72**: k=3 and k=4 produce fundamentally different partitions. Within the same k, adjacent gammas are highly correlated (NMI 0.80–0.85).
- Stability does not differentiate the candidates — both are equally robust. Decision rests on practical domain quality.

---

### 3d. Final Recommendation

**Selected: k=4, γ=0.0008** (with mega-domain splitting)

| Metric | k=3, γ=0.0006 | **k=4, γ=0.0008** | Rationale |
|--------|---------------|-------------------|-----------|
| Domains | 29 | **25** | Fewer starting domains → less manual curation |
| Q (modularity) | **0.770** | 0.738 | Both well above 0.3 significance threshold |
| Significance | 26,177 | **31,834** | Higher = more statistically significant structure |
| Max domain | 1,163 | **1,185** | Comparable; handled by mega-domain split |
| Median | 175 | **180** | Comparable; good for browsing |
| P10 | 84 | **89** | k=4 slightly better — smallest domains more viable |
| P90 | 336 | **331** | Comparable |
| Singletons | 0 | **0** | Perfect — all clusters assigned |
| Domain count std | 1.3 | **0.9** | k=4 more stable across random seeds |
| Outliers reassigned | 26 | **87** | More outlier correction = cleaner boundaries |

**Rationale**:
1. **Fewer domains to curate**: 25 vs 29 — since we manually merge/split/label domains anyway, a tighter starting set is preferred
2. **Tighter count stability**: std=0.9 (range 24–27) vs std=1.3 (range 26–33) — less randomness in the starting partition
3. **Higher significance**: 31,834 vs 26,177 — the denser k=4 graph captures more meaningful cluster relationships
4. **Better P10**: 89 vs 84 — even the smallest domains are substantial enough to be meaningful
5. **Mega-domain handled by splitting**: Domain #0 (1,239 clusters) is split into 5 sub-groups using K-means on cluster centroids, producing a final ~29 domain starting set
6. **Q tradeoff acceptable**: 0.738 vs 0.770 is a minor difference; both indicate strong community structure
7. **Consistent with demo library finding**: The demo analysis noted k=3 for super-clusters, but V2's 20x larger cluster set (6,562 vs ~300) benefits from the denser k=4 graph to maintain cohesion

### Applied Configuration

```
k=4, gamma=0.0008, min_similarity=0.3, outlier_reassignment=True (threshold=0.7)
→ 25 domains (87 outlier clusters reassigned)
→ Mega-domain #0 (1,239 clusters) split into 5 sub-groups
→ Final: ~29 domains for manual curation
```

**Mega-domain split (domain #0 → 5 sub-groups)**:

| Sub-domain | Clusters | Auto-label |
|-----------|----------|------------|
| 25 | 409 | cugel's arrival in smolod / alethiometer's significance / bene gesserit |
| 27 | 307 | protomolecule experimentation / úrsula's business acumen |
| 28 | 209 | devon's protective instincts / lilith's aspirations for social mobility |
| 29 | 164 | american expatriate experience / gersbach's public persona / magister ludi |
| 26 | 150 | quidditch team tryouts / america shaftoe's personal history |

---

## Phase 4: min_similarity Sweep (Topic-Level)

*Skipped — diminishing returns. The KNN graph at min_similarity=0.65 already produces well-distributed clusters with no pathologies (zero singletons, no mega-blobs). Sweeping min_similarity 0.5–0.8 would require rebuilding the GPU KNN graph + reclustering (~47 min) with minimal expected impact on cluster quality.*

---

## Phase 5: Domain Membership — The Differentiation Problem

### The Problem

After generating 29 domains (k=4, γ=0.0008, mega-split) and loading them into V2, we measured how many books appear in each domain using the current "any topic link" membership rule. The results reveal a **fundamental differentiation failure**:

| Domain | Clusters | Books | % of 937 | Label |
|-------:|:--------:|------:|---------:|-------|
| 1 | 710 | 936 | **99.9%** | Mind & Potential |
| 4 | 330 | 926 | 98.8% | Analytical & Creative Systems |
| 2 | 394 | 923 | 98.5% | Advanced Scientific Concepts |
| 6 | 288 | 922 | 98.4% | Literary Study & Writing |
| 3 | 333 | 918 | 98.0% | Modern Conflict & Warfare |
| 11 | 196 | 916 | 97.8% | Hidden Lives & Lore |
| ... | ... | ... | 90-97% | *(22 of 29 domains above 90%)* |
| 26 | 150 | 748 | 79.8% | Royal History & Intrigue |
| 24 | 32 | 426 | **45.5%** | Russian Literature & History |

**Every domain contains 80-100% of the library.** The domains are not differentiating books at all.

### Root Cause

The membership chain is: `book → chunks (~200/book) → topics (~5/chunk) → clusters → domains`. A typical 300-page book produces ~1,000 topic mentions spread across hundreds of clusters. Those clusters inevitably span most or all 29 domains. Binary "any link" membership is meaningless at this granularity.

### Case Study: Russian Literature & History (Domain 24)

The most specific domain (32 clusters, only 45% of library) still includes 426 books — but the vast majority have trivial connections:

| Concentration | Books | % of 426 | Examples |
|:-------------:|------:|---------:|----------|
| ≥10% | 2 | 0.5% | Reading Chekhov (15.9%), Life and Fate (10.7%) |
| ≥5% | 7 | 1.6% | + Blowout, Complete Chekhov, Fall of Giants, Sketches from a Hunter's Album |
| ≥3% | 18 | 4.2% | + Doctor Zhivago, Crime and Punishment, Tinker Tailor |
| ≥1% | 57 | 13.4% | + A Swim in a Pond in the Rain, How Fiction Works |
| <1% | **369** | **86.6%** | Mistborn, Game of Thrones, On Cooking (!), Norton Anthology |

**86% of the "Russian Literature" domain's books have <1% of their topics in this domain.** On Cooking and Mistborn are counted as "Russian Literature" because of a single incidental topic mention.

The mean concentration is **0.63%** and median is **0.25%** — noise-level membership.

### Solution Direction: Weighted Membership + Two-Tier Hierarchy

Two complementary changes are needed:

**1. Weighted domain membership (primary domain assignment)**

Instead of binary membership, compute each book's **topic concentration** per domain. Assign each book to its top 1-3 domains by weight. This turns the flat many-to-many into meaningful rankings.

For Russian Literature at a ≥3% threshold, the domain would contain **18 actually-relevant books** instead of 426 — a 24x improvement in precision.

**2. Two-tier domain hierarchy (Galaxy → Constellation)**

- **Galaxy** (~5-8): Broad browsing categories (Fiction, Science & Technology, History & Politics, Arts & Culture, Food & Craft, etc.)
- **Constellation** (~40-80): Finer topical groupings within each galaxy (the current 29 domains are too coarse for the second tier, too fine for the first)

Books assigned to their **primary galaxy** by topic weight, then ranked within constellations. This gives meaningful differentiation at the top level and browsable depth within each galaxy.

### Implications

- **Domain counts in the LLM labels table are unreliable** — the reported "book_count" uses the broken any-link rule
- **The `cluster_books` bridge table** already stores `topic_count` and `book_total_topics` — everything needed for weighted membership is already materialized
- **`book_cluster_relevance()`** in `stats.py` already implements concentration + BM25 + PPMI scoring — this can be extended to domain-level aggregation
- **App display changes needed**: The themes page currently shows all books per domain; needs to rank by concentration and apply a minimum threshold
- **More domains may be needed**: 29 is too few for constellations; may need γ ≈ 0.002 (50-60 domains) for the constellation tier, with a separate galaxy grouping above

---

## Phase 6: Demo Library Community Sweep

**Goal**: Find optimal k and γ for community-level Leiden clustering on the demo library, matching the V2 methodology (Phase 2 of `topic_hierarchy_architecture.md`). The demo library has 7x fewer topics than V2, so it needs its own hyperparameter tuning.

**Database**: `demo_library.db`
- **121,118 topics** with embeddings (384-dim BGE-small)
- **100 books**, 31,727 chunks
- **462,873 co-occurrence pairs**
- **26 domains** (existing)
- **2,468 clusters** (existing, from earlier Leiden at k=10, γ=0.001)

**V2 reference**: k=4, γ=4.86e-5 → ~230 communities (from 837K topics)

---

### 6a. KNN k Sweep

**Goal**: Find optimal graph density for the demo scale. Tested k=2,3,4,5,6 at fixed γ=0.0001.

| k | Edges | Clusters | Quality | Largest | Leiden time |
|---|-------|----------|---------|---------|-------------|
| 2 | 197K | 761 | 454,172 | 2,533 | 1.7s |
| 3 | 288K | 270 | 559,462 | 2,050 | 2.3s |
| **4** | **379K** | **186** | **665,094** | **3,145** | **2.7s** |
| 5 | 470K | 140 | 770,528 | 3,531 | 3.2s |
| 6 | 560K | 107 | 877,066 | 4,969 | 3.6s |

**Observations**:
- **k=2**: Very sparse → 761 fragments with large mega-clusters. Too noisy.
- **k=3**: 270 clusters at this fixed γ — in the right ballpark, denser graph starts consolidating.
- **k=4**: 186 clusters — close to target. Matches V2 methodology.
- **k=5-6**: Too dense — only 107-140 mega-clusters with 3,500-5,000 topic maximums.

**Candidates**: k=3 and k=4, consistent with V2 domain sweep findings (Phase 3) where k=3 and k=4 were the finalists.

---

### 6b. Full Resolution Sweep: k=3 and k=4

25 log-spaced resolutions from 1e-6 to 5e-4 for each k value.

#### k=3 (288K edges)

| γ | Clusters | Q | Quality | Significance | Max | Median | Singles | NMI |
|---|----------|------|---------|-------------|------|--------|---------|------|
| 1.00e-06 | 47 | 0.005 | 644,794 | 6,805 | 120,683 | 5 | 17 | 0.867 |
| 1.68e-06 | 98 | 0.021 | 635,012 | 23,584 | 119,329 | 5 | 17 | 0.711 |
| 2.82e-06 | 161 | 0.054 | 619,332 | 42,268 | 116,655 | 6 | 17 | 0.649 |
| 3.65e-06 | 235 | 0.081 | 608,221 | 64,939 | 114,335 | 6 | 17 | 0.325 |
| 4.73e-06 | 139 | 0.425 | 601,308 | 135,280 | 84,199 | 6 | 17 | 0.467 |
| 7.94e-06 | 88 | 0.750 | 590,476 | 284,294 | 32,832 | 6 | 17 | 0.542 |
| 1.03e-05 | 76 | 0.809 | 586,406 | 387,973 | 19,298 | 7 | 17 | 0.573 |
| 1.73e-05 | 84 | 0.849 | 578,830 | 563,904 | 9,475 | 26 | 17 | 0.625 |
| 2.90e-05 | 105 | 0.857 | 572,743 | 686,889 | 5,226 | 720 | 17 | 0.694 |
| 4.86e-05 | 157 | 0.857 | 566,973 | 811,389 | 4,276 | 714 | 17 | 0.729 |
| **6.30e-05** | **196** | **0.857** | **564,311** | **871,591** | **2,750** | **506** | **17** | **0.754** |
| **8.16e-05** | **237** | **0.855** | **561,757** | **918,349** | **2,279** | **443** | **17** | **0.771** |
| **1.06e-04** | **281** | **0.852** | **558,781** | **960,615** | **2,308** | **381** | **17** | **0.782** |
| 1.37e-04 | 345 | 0.850 | 556,062 | 1,006,292 | 1,968 | 304 | 17 | 0.801 |
| 2.30e-04 | 514 | 0.845 | 550,461 | 1,094,684 | 1,260 | 199 | 17 | 0.834 |
| 5.00e-04 | 886 | 0.836 | 541,333 | 1,205,346 | 1,053 | 117 | 17 | — |

#### k=4 (379K edges)

| γ | Clusters | Q | Quality | Significance | Max | Median | Singles | NMI |
|---|----------|------|---------|-------------|------|--------|---------|------|
| 1.00e-06 | 26 | 0.003 | 796,530 | 4,230 | 120,890 | 1 | 17 | 0.796 |
| 1.68e-06 | 41 | 0.005 | 786,635 | 8,790 | 120,680 | 5 | 17 | 0.699 |
| 2.82e-06 | 59 | 0.021 | 770,213 | 25,346 | 119,446 | 6 | 17 | 0.594 |
| 4.73e-06 | 92 | 0.075 | 744,172 | 63,062 | 115,363 | 6 | 17 | 0.327 |
| 6.13e-06 | 72 | 0.356 | 731,247 | 112,746 | 87,474 | 6 | 17 | 0.240 |
| 1.03e-05 | 59 | 0.712 | 710,634 | 310,776 | 34,600 | 7 | 17 | 0.496 |
| 1.73e-05 | 60 | 0.805 | 696,127 | 536,245 | 14,606 | 11 | 17 | 0.612 |
| 2.90e-05 | 70 | 0.828 | 686,588 | 736,762 | 7,362 | 799 | 17 | 0.682 |
| 4.86e-05 | 103 | 0.831 | 677,992 | 883,832 | 5,185 | 865 | 17 | 0.729 |
| 8.16e-05 | 152 | 0.827 | 668,584 | 1,033,157 | 3,990 | 671 | 17 | 0.757 |
| **1.06e-04** | **185** | **0.826** | **664,421** | **1,080,034** | **2,825** | **527** | **17** | **0.780** |
| **1.37e-04** | **235** | **0.822** | **660,261** | **1,156,499** | **2,708** | **455** | **17** | **0.802** |
| **1.77e-04** | **287** | **0.819** | **655,905** | **1,207,531** | **2,028** | **356** | **17** | **0.820** |
| 2.30e-04 | 345 | 0.816 | 652,062 | 1,268,061 | 1,511 | 305 | 17 | 0.829 |
| 3.86e-04 | 512 | 0.808 | 642,816 | 1,370,885 | 1,046 | 199 | 17 | 0.854 |
| 5.00e-04 | 619 | 0.805 | 638,165 | 1,413,690 | 983 | 168 | 17 | — |

#### Observations

Both k values exhibit the **same non-monotonic phase transition** seen in V2 (topic_hierarchy_architecture.md):
- Cluster count rises, drops sharply (γ ≈ 5-6e-6), then recovers monotonically.
- The phase transition is a CPM landscape feature — the algorithm discovers a qualitatively different partition structure at that scale.
- Below the transition: many small fragments + one mega-cluster (Max > 100K). Above: well-distributed communities.
- **Q stabilizes at ~0.82-0.86** in the post-transition region for both k values.
- **Singletons constant at 17** across all resolutions and k values — minor.

#### Comparison at ~235 Communities

| Metric | k=3 (γ=8.16e-5) | k=4 (γ=1.37e-4) |
|--------|-----------------|-----------------|
| Clusters | 237 | 235 |
| Q (modularity) | **0.855** | 0.822 |
| Significance | 918,349 | **1,156,499** |
| Max size | **2,279** | 2,708 |
| Median | **443** | 455 |
| NMI | 0.771 | **0.802** |
| Singletons | 17 | 17 |

**k=3** has slightly better Q (0.855 vs 0.822) and smaller max cluster.
**k=4** has 26% higher significance and better NMI stability (0.802 vs 0.771).

This mirrors the V2 domain-level finding (Phase 3d) where k=4 won over k=3 for its higher significance and tighter stability, despite k=3 having marginally better Q.

---

### 6c. Recommendation

**Selected: k=4, γ=1.37e-4** for demo community clustering.

| Metric | Demo (k=4, γ=1.37e-4) | V2 (k=4, γ=4.86e-5) |
|--------|------------------------|----------------------|
| Communities | 235 | ~230 |
| Q | 0.822 | 0.831 |
| Significance | 1,156,499 | 883,832 |
| Max | 2,708 | 5,185 |
| Median | 455 | 865 |
| NMI | 0.802 | 0.729 |
| Singletons | 17 | 17 |

*V2 metrics are from the full V2 dataset (837K topics). Demo metrics are from the demo library (121K topics). Each γ was independently tuned to produce ~230 communities at its respective scale.*

**Rationale**:
1. **Matches V2 community count** (~235 vs ~230) — same user experience
2. **Higher resolution** than V2 (1.37e-4 vs 4.86e-5) is expected — smaller corpus needs stronger resolution to produce the same number of communities
3. **k=4 consistent with V2** — same graph density choice for the same reasons (significance, stability)
4. **Q well above significance threshold** — 0.822 >> 0.3
5. **Singletons negligible** — 17 out of 121K topics

**Scaling relationship**: Demo needs γ ~2.8x higher than V2 to produce the same community count at k=4. This makes sense: with 7x fewer topics but only 2.8x higher γ, the demo's communities are proportionally larger relative to corpus size — each community covers more of the 100-book library.

---

## Phase 7: Higher-Resolution Domain Sweep (55-75 Domains)

**Goal**: Instead of the current approach (k=4, γ=0.0008 → 25 domains + K-means mega-domain splitting), find parameters for a single Leiden pass that directly produces 55-75 domains. The 929-book library spans too many genres to compress into 25 themes — the mega-domain (#0 with 1,239 clusters) and cross-domain contamination (e.g. baseball in "Modern Conflict & Warfare") suggest the library needs finer-grained top-level categories.

**Motivation**: Second-pass splitting (Leiden or K-means on mega-domain subgraphs) is a workaround for insufficient initial resolution. A single well-tuned pass is cleaner and avoids hybrid methodology issues.

**Method**: Same infrastructure as Phase 3 — `build_cluster_graph()` + `leiden_sweep()` from sweep.py. Swept k=3 (16,287 edges) and k=4 (21,550 edges) across 20 log-spaced resolutions each.

---

### 7a. k=3 Sweep (γ = 0.0008–0.0025)

| γ | Domains | Q | Sig | Max | Median | Mean | P10 | P90 | Singles | NMI |
|---|---------|-------|--------|-----|--------|------|-----|-----|---------|------|
| 8.00e-04 | 39 | 0.775 | 29,892 | 646 | 135 | 168.3 | 59 | 269 | 0 | 0.833 |
| 8.49e-04 | 43 | 0.773 | 30,260 | 681 | 108 | 152.6 | 56 | 286 | 0 | 0.825 |
| 9.02e-04 | 41 | 0.772 | 30,417 | 643 | 143 | 160.0 | 55 | 292 | 0 | 0.853 |
| 9.58e-04 | 45 | 0.773 | 31,304 | 648 | 108 | 145.8 | 62 | 261 | 0 | 0.793 |
| 1.02e-03 | 46 | 0.767 | 30,560 | 937 | 107 | 142.7 | 48 | 241 | 0 | 0.808 |
| 1.08e-03 | 52 | 0.770 | 32,091 | 680 | 97 | 126.2 | 35 | 236 | 0 | 0.855 |
| 1.15e-03 | 53 | 0.770 | 32,647 | 613 | 98 | 123.8 | 49 | 226 | 0 | 0.861 |
| 1.22e-03 | 54 | 0.769 | 32,582 | 616 | 87 | 121.5 | 48 | 231 | 0 | 0.859 |
| **1.29e-03** | **58** | **0.767** | **33,103** | **581** | **86** | **113.1** | **37** | **216** | **0** | **0.841** |
| 1.37e-03 | 58 | 0.767 | 33,207 | 589 | 87 | 113.1 | 42 | 222 | 0 | 0.854 |
| **1.46e-03** | **61** | **0.767** | **33,886** | **573** | **90** | **107.6** | **37** | **203** | **0** | **0.863** |
| **1.55e-03** | **68** | **0.765** | **34,639** | **572** | **78** | **96.5** | **28** | **182** | **0** | **0.861** |
| **1.64e-03** | **67** | **0.764** | **34,772** | **537** | **80** | **97.9** | **34** | **163** | **0** | **0.879** |
| **1.74e-03** | **71** | **0.764** | **35,109** | **541** | **71** | **92.4** | **28** | **180** | **0** | **0.876** |
| 1.85e-03 | 77 | 0.760 | 35,639 | 519 | 67 | 85.2 | 25 | 155 | 0 | 0.860 |
| 1.97e-03 | 80 | 0.758 | 35,747 | 545 | 68 | 82.0 | 23 | 147 | 0 | 0.845 |
| 2.09e-03 | 84 | 0.753 | 36,551 | 394 | 63 | 78.1 | 27 | 144 | 0 | 0.841 |
| 2.22e-03 | 87 | 0.749 | 36,853 | 370 | 62 | 75.4 | 27 | 130 | 0 | 0.856 |
| 2.35e-03 | 93 | 0.746 | 37,649 | 339 | 55 | 70.6 | 25 | 118 | 0 | 0.838 |
| 2.50e-03 | 94 | 0.745 | 37,577 | 395 | 56 | 69.8 | 26 | 128 | 0 | — |

### 7b. k=4 Sweep (γ = 0.0012–0.0035)

| γ | Domains | Q | Sig | Max | Median | Mean | P10 | P90 | Singles | NMI |
|---|---------|-------|--------|------|--------|------|-----|-----|---------|------|
| 1.20e-03 | 40 | 0.737 | 36,210 | 1,035 | 109 | 164.1 | 52 | 279 | 0 | 0.831 |
| 1.27e-03 | 36 | 0.738 | 35,804 | 971 | 149 | 182.3 | 55 | 270 | 0 | 0.805 |
| 1.34e-03 | 40 | 0.741 | 37,380 | 629 | 127 | 164.1 | 50 | 262 | 0 | 0.850 |
| 1.42e-03 | 44 | 0.737 | 38,446 | 655 | 115 | 149.1 | 44 | 259 | 0 | 0.858 |
| 1.50e-03 | 45 | 0.739 | 38,591 | 619 | 105 | 145.8 | 39 | 256 | 0 | 0.846 |
| 1.59e-03 | 46 | 0.737 | 38,883 | 622 | 105 | 142.7 | 54 | 252 | 0 | 0.816 |
| 1.68e-03 | 49 | 0.733 | 39,453 | 647 | 99 | 133.9 | 48 | 242 | 0 | 0.873 |
| **1.78e-03** | **55** | **0.730** | **40,642** | **640** | **89** | **119.3** | **40** | **225** | **0** | **0.879** |
| **1.88e-03** | **57** | **0.728** | **40,803** | **594** | **80** | **115.1** | **33** | **210** | **0** | **0.843** |
| **1.99e-03** | **58** | **0.729** | **41,399** | **591** | **90** | **113.1** | **38** | **187** | **0** | **0.866** |
| **2.11e-03** | **58** | **0.728** | **41,248** | **495** | **88** | **113.1** | **41** | **206** | **0** | **0.870** |
| **2.23e-03** | **60** | **0.729** | **41,753** | **481** | **86** | **109.4** | **41** | **205** | **0** | **0.879** |
| **2.36e-03** | **65** | **0.727** | **42,252** | **480** | **80** | **101.0** | **29** | **193** | **0** | **0.877** |
| **2.50e-03** | **68** | **0.725** | **42,716** | **482** | **75** | **96.5** | **34** | **198** | **0** | **0.876** |
| **2.64e-03** | **72** | **0.723** | **43,321** | **475** | **64** | **91.1** | **35** | **165** | **0** | **0.869** |
| 2.79e-03 | 74 | 0.721 | 43,618 | 475 | 65 | 88.7 | 32 | 165 | 0 | 0.864 |
| 2.96e-03 | 77 | 0.719 | 44,069 | 429 | 66 | 85.2 | 23 | 171 | 0 | 0.859 |
| 3.13e-03 | 79 | 0.716 | 44,130 | 423 | 64 | 83.1 | 22 | 148 | 0 | 0.875 |
| 3.31e-03 | 84 | 0.712 | 44,947 | 409 | 61 | 78.1 | 26 | 139 | 0 | 0.860 |
| 3.50e-03 | 95 | 0.707 | 46,044 | 420 | 57 | 69.1 | 19 | 125 | 0 | — |

### 7c. k=5 Sweep (γ = 0.0015–0.006)

Graph: 6,562 nodes, 26,746 edges.

| γ | Domains | Q | Sig | Max | Median | Mean | P10 | P90 | Singles | NMI |
|---|---------|-------|--------|-----|--------|------|-----|-----|---------|------|
| 1.50e-03 | 35 | 0.715 | 41,701 | 991 | 156 | 187.5 | 65 | 309 | 0 | 0.789 |
| 1.61e-03 | 39 | 0.719 | 43,665 | 628 | 141 | 168.3 | 50 | 283 | 0 | 0.863 |
| 1.74e-03 | 40 | 0.716 | 44,608 | 607 | 138 | 164.1 | 49 | 264 | 0 | 0.873 |
| 1.87e-03 | 41 | 0.716 | 44,799 | 607 | 120 | 160.0 | 58 | 278 | 0 | 0.867 |
| 2.01e-03 | 42 | 0.713 | 45,316 | 553 | 128 | 156.2 | 56 | 269 | 0 | 0.850 |
| 2.16e-03 | 49 | 0.712 | 46,855 | 540 | 102 | 133.9 | 35 | 226 | 0 | 0.878 |
| 2.32e-03 | 50 | 0.709 | 46,946 | 517 | 103 | 131.2 | 38 | 236 | 0 | 0.855 |
| 2.50e-03 | 52 | 0.709 | 47,512 | 487 | 95 | 126.2 | 46 | 221 | 0 | 0.852 |
| **2.69e-03** | **56** | **0.706** | **48,195** | **467** | **88** | **117.2** | **32** | **217** | **0** | **0.889** |
| **2.89e-03** | **64** | **0.703** | **49,593** | **469** | **75** | **102.5** | **31** | **205** | **0** | **0.889** |
| **3.11e-03** | **63** | **0.704** | **49,624** | **438** | **77** | **104.2** | **33** | **207** | **0** | **0.875** |
| **3.35e-03** | **67** | **0.698** | **50,498** | **433** | **75** | **97.9** | **28** | **190** | **0** | **0.883** |
| **3.60e-03** | **68** | **0.697** | **50,634** | **429** | **73** | **96.5** | **31** | **183** | **0** | **0.866** |
| **3.87e-03** | **72** | **0.693** | **51,459** | **429** | **73** | **91.1** | **32** | **180** | **0** | **0.863** |
| 4.17e-03 | 79 | 0.690 | 52,325 | 412 | 71 | 83.1 | 26 | 148 | 0 | 0.872 |
| 4.48e-03 | 85 | 0.683 | 53,026 | 395 | 56 | 77.2 | 26 | 159 | 0 | 0.877 |
| 4.82e-03 | 91 | 0.684 | 53,748 | 382 | 57 | 72.1 | 24 | 128 | 1 | 0.875 |
| 5.19e-03 | 95 | 0.679 | 54,051 | 366 | 54 | 69.1 | 24 | 110 | 0 | 0.882 |
| 5.58e-03 | 99 | 0.676 | 54,592 | 418 | 52 | 66.3 | 24 | 111 | 0 | 0.880 |
| 6.00e-03 | 109 | 0.668 | 55,453 | 332 | 49 | 60.2 | 21 | 90 | 0 | — |

### 7d. k=6 Sweep (γ = 0.002–0.008)

Graph: 6,562 nodes, 31,985 edges.

| γ | Domains | Q | Sig | Max | Median | Mean | P10 | P90 | Singles | NMI |
|---|---------|-------|--------|-----|--------|------|-----|-----|---------|------|
| 2.00e-03 | 34 | 0.702 | 48,604 | 664 | 158 | 193.0 | 48 | 375 | 0 | 0.848 |
| 2.15e-03 | 40 | 0.699 | 50,438 | 565 | 137 | 164.1 | 48 | 330 | 0 | 0.838 |
| 2.31e-03 | 40 | 0.698 | 50,857 | 604 | 135 | 164.1 | 46 | 299 | 0 | 0.828 |
| 2.49e-03 | 44 | 0.694 | 52,102 | 544 | 125 | 149.1 | 35 | 278 | 0 | 0.858 |
| 2.68e-03 | 46 | 0.693 | 52,495 | 534 | 103 | 142.7 | 48 | 292 | 0 | 0.870 |
| 2.88e-03 | 50 | 0.691 | 53,627 | 547 | 93 | 131.2 | 35 | 248 | 0 | 0.862 |
| 3.10e-03 | 51 | 0.689 | 54,031 | 509 | 92 | 128.7 | 36 | 280 | 0 | 0.888 |
| 3.33e-03 | 51 | 0.690 | 54,118 | 524 | 82 | 128.7 | 36 | 244 | 0 | 0.869 |
| **3.59e-03** | **58** | **0.684** | **55,509** | **485** | **80** | **113.1** | **32** | **234** | **0** | **0.876** |
| **3.86e-03** | **62** | **0.682** | **56,447** | **483** | **73** | **105.8** | **28** | **208** | **0** | **0.862** |
| **4.15e-03** | **67** | **0.678** | **57,443** | **455** | **71** | **97.9** | **27** | **192** | **0** | **0.875** |
| **4.46e-03** | **70** | **0.674** | **57,897** | **442** | **67** | **93.7** | **28** | **187** | **0** | **0.858** |
| **4.80e-03** | **74** | **0.671** | **59,174** | **408** | **71** | **88.7** | **27** | **181** | **0** | **0.881** |
| 5.16e-03 | 77 | 0.668 | 59,254 | 417 | 65 | 85.2 | 29 | 157 | 0 | 0.885 |
| 5.55e-03 | 82 | 0.662 | 60,149 | 380 | 63 | 80.0 | 27 | 141 | 0 | 0.883 |
| 5.98e-03 | 89 | 0.658 | 60,974 | 384 | 58 | 73.7 | 23 | 131 | 0 | 0.883 |
| 6.43e-03 | 89 | 0.657 | 60,992 | 359 | 58 | 73.7 | 27 | 126 | 0 | 0.890 |
| 6.91e-03 | 102 | 0.649 | 62,446 | 361 | 54 | 64.3 | 18 | 106 | 0 | 0.904 |
| 7.44e-03 | 104 | 0.646 | 62,303 | 331 | 52 | 63.1 | 16 | 100 | 0 | 0.897 |
| 8.00e-03 | 112 | 0.640 | 62,885 | 334 | 50 | 58.6 | 16 | 103 | 3 | — |

---

### 7e. Cross-k Analysis

**All four k values produce viable 55-75 domain partitions with zero singletons.**

Comparison at ~67 domains (closest point for each k):

| Metric | k=3 (γ=1.64e-3) | k=4 (γ=2.36e-3) | k=5 (γ=3.35e-3) | k=6 (γ=4.15e-3) |
|--------|-----------------|-----------------|-----------------|-----------------|
| Domains | 67 | 65 | 67 | 67 |
| Q (modularity) | **0.764** | 0.727 | 0.698 | 0.678 |
| Significance | 34,772 | 42,252 | 50,498 | **57,443** |
| Max domain | 537 | **480** | 433 | 455 |
| Median | **80** | **80** | 75 | 71 |
| Mean | 97.9 | 101.0 | 97.9 | 97.9 |
| P10 | 34 | 29 | 28 | 27 |
| P90 | 163 | 193 | 190 | 192 |
| NMI | **0.879** | 0.877 | 0.883 | 0.875 |
| Singletons | 0 | 0 | 0 | 0 |

#### Trends across k

- **Q decreases monotonically** with k: 0.764 → 0.727 → 0.698 → 0.678. Denser graphs have lower modularity because edges bridge more communities.
- **Significance increases monotonically** with k: 34,772 → 42,252 → 50,498 → 57,443. More edges = more statistical signal to differentiate from random graphs. However, significance is not directly comparable across different graph densities — it naturally grows with edge count.
- **Max domain is similar** across all k values (433–537 at ~67 domains). No mega-domains at any k.
- **Median domain size converges** around 71–80 for all k values at this domain count.
- **NMI stability is uniformly high** (0.875–0.883) — all partitions are robust.
- **P10 decreases slightly** with k (34 → 27), meaning the smallest domains get a bit smaller with denser graphs.

#### k=3 vs k=4 vs k=5 vs k=6: Which to choose?

The key tradeoff is **Q vs. domain cohesion**:

- **k=3** has the highest Q (0.764) and best P10 (34), but significance is lowest. The sparser graph may under-connect related clusters.
- **k=4** balances Q (0.727) with higher significance and the smallest max domain (480). This was the winner in Phase 3 for 25 domains.
- **k=5** gives even higher significance (50,498) and lower max (433), with only modest Q loss (0.698). The additional edges may help connect clusters that k=4 misses.
- **k=6** maximizes significance (57,443) but Q drops to 0.678 and P10 falls to 27. At higher k, denser connectivity can over-merge and blur domain boundaries.

**Significance scaling caveat**: Higher k means more edges, which mechanically inflates significance scores. The ~65% increase from k=3 to k=6 partly reflects the ~2x edge count increase (16K → 32K), not purely better structure. Comparing significance across different k values requires normalizing by edge count or using a k-independent metric.

**Improvement over Phase 3 (25-domain) approach:**

| Aspect | Phase 3 (25 domains) | Phase 7 (65 domains) |
|--------|---------------------|---------------------|
| Domains | 25 (+ 5 from mega-split = 29) | 60–70 |
| Max domain | 1,185 clusters (mega!) | 430–540 clusters |
| Methodology | Two-pass (Leiden + K-means split) | Single Leiden pass |
| Contamination risk | High (diverse topics forced together) | Lower (finer granularity) |
| Manual curation | Heavy (29 labels + merge/split) | More labels, but cleaner groups |

---

#### Quick Reference

| k | Q | Sig | Max | Median | P10 | NMI | Verdict |
|---|-------|--------|-----|--------|-----|------|---------|
| 3 | **0.764** | 34,772 | 537 | **80** | **34** | 0.879 | Highest Q, but sparser graph may under-connect |
| 4 | 0.727 | 42,252 | **480** | **80** | 29 | 0.877 | Validated in Phase 3, good all-around |
| **5** | **0.698** | **50,498** | **433** | **75** | **28** | **0.883** | **Lowest max domain, best connectivity** |
| 6 | 0.678 | 57,443 | 455 | 71 | 27 | 0.875 | Diminishing returns, Q drops further |

- **Q drops steadily** with higher k (0.764 → 0.678) — sparser graphs have cleaner modularity
- **Significance climbs** with k but partly mechanical (more edges = more signal, not directly comparable across k)
- **Max domain smallest at k=5** (433) — denser graph connects more clusters without over-merging
- **P10 favors lower k** — smallest domains more viable at k=3 (34 clusters) vs k=6 (27)
- **NMI uniformly high** (0.875–0.883) — all partitions are robust regardless of k

---

### 7f. Recommendation

**Selected: k=5, γ=0.0035** for ~67 domains.

k=5 offers the best balance of connectivity and domain cohesion. The denser graph (26,746 edges vs k=4's 21,550) helps related clusters find each other without the over-merging seen at k=6. Easy to fall back to k=3 or k=4 if domain inspection reveals issues.

| k | γ | Domains | Q | Max | Median | P10 | Notes |
|---|---|---------|-------|-----|--------|-----|-------|
| 5 | 0.0029 | 64 | 0.703 | 469 | 75 | 31 | Conservative |
| **5** | **0.0035** | **67** | **0.698** | **433** | **75** | **28** | **Selected** |
| 5 | 0.0039 | 72 | 0.693 | 429 | 73 | 32 | Fine-grained |

**Next step**: Run `regenerate-domains` at k=5, γ=0.0035, inspect domain contents, then decide if the granularity is right before proceeding to LLM labeling.

### 7g. Domain Generation & Labeling

**Generated**: `regenerate-domains --method leiden --cluster-knn-k 5 --resolution 0.0035`
- **Result**: 66 domains from 6,562 Leiden clusters
- **185 outlier clusters** reassigned (threshold=0.7)
- **Largest domain**: 463 clusters (7% of total — down from 1,239 / 19% in old 25-domain setup)

**Cross-domain contamination check** (old setup had baseball/fitness mixed into military):
- Domain 12 "Modern Warfare & Geopolitics": Clean — 75 military topics, 0 baseball, 2 fitness
- Domain 18 "Games, Sports & Competition": Absorbed the games/sports/fitness content into its own domain

**LLM auto-labeling**: gemma-3-27b via LM Studio (localhost:1234) generated initial labels for all 66 domains from top-20 aggregated topics per domain. Several labels were misleading — the 27b model often latched onto surface-level keywords rather than understanding the domain's true content.

**Manual curation**: Claude Opus 4.6 reviewed top-30 topics for all 66 domains and produced curated labels. Key fixes:

| Domain | gemma-3-27b (misleading) | Curated (accurate) | Why |
|--------|--------------------------|-------------------|-----|
| 9 | Advanced Military Science | **Sci-Fi Technology & Space** | Soft blades, lightspeed ships, space elevators — nothing military |
| 18 | Strategic Conflict & Survival | **Games, Sports & Competition** | Baseball, card games, pull-ups, tennis, chess |
| 34 | Security & Intrusion | **Buildings & Physical Spaces** | House/palace architecture, hotels, log cabins |
| 58 | Political Intrigue & Power | **Royal Courts & Dynasties** | 90% ASOIAF + historical monarchy |
| 25 | European Family & Society | **European Literary Fiction** | Buddenbrooks, Glass Bead Game, Man Without Qualities |
| 2 | Science & Technology | **Technical & Quantitative** | ML/programming + physics + trading — not just "science" |
| 30 | Global History & Fiction | **Colonialism & the Tropics** | Cholera, Panama Canal, Congo reform, García Márquez |
| 28 | Hidden Histories & Upheaval | **Coming of Age & Hidden Pasts** | WWI childhood, concealed identity, generational trauma |
| 48 | Social & Economic Dislocation | **Migration & Displacement** | Reunions, housing crisis, La Bestia, refugees |
| 20 | Power, Intrigue & Perception | **Philosophy & Character Study** | Cynic philosophy, Stoic indifferents, Belbo, Aschenbach |

**Full curated domain list** (66 domains, sorted by cluster count):

| ID | Clusters | Label |
|----|----------|-------|
| 0 | 463 | Speculative Fiction |
| 1 | 455 | Character Drama & Emotion |
| 2 | 252 | Technical & Quantitative |
| 3 | 220 | Business & Investing |
| 4 | 217 | Psychology & Self-Improvement |
| 5 | 202 | Religion & Classical Learning |
| 6 | 200 | Culinary Arts |
| 7 | 189 | Politics & Governance |
| 8 | 178 | Writing Craft & Publishing |
| 9 | 154 | Sci-Fi Technology & Space |
| 10 | 153 | Natural Environments & Wilderness |
| 11 | 144 | Literary Figures & Criticism |
| 12 | 133 | Modern Warfare & Geopolitics |
| 13 | 128 | Society & Social Order |
| 14 | 127 | Early American History |
| 15 | 126 | Fear & Anxiety |
| 16 | 120 | Survival & Animal Encounters |
| 17 | 115 | Family & Parenthood |
| 18 | 111 | Games, Sports & Competition |
| 19 | 108 | Espionage & Secrecy |
| 20 | 107 | Ancient & Medieval History |
| 21 | 106 | Philosophy & Character Study |
| 22 | 104 | Weapons & Authority |
| 23 | 102 | Health, Medicine & Biology |
| 24 | 95 | Philosophy & Epistemology |
| 25 | 87 | European Literary Fiction |
| 26 | 87 | Education & Training |
| 27 | 86 | Coming of Age & Hidden Pasts |
| 28 | 84 | Civil Rights & World History |
| 29 | 84 | Crime & Investigation |
| 30 | 82 | Love & Relationships |
| 31 | 80 | Art, Treasure & Archaeology |
| 32 | 79 | Medicine & Drug Trade |
| 33 | 77 | Buildings & Physical Spaces |
| 34 | 76 | Colonialism & the Tropics |
| 35 | 75 | Economic Theory & Systems |
| 36 | 73 | Cities & Urban Life |
| 37 | 69 | Catastrophe & Destruction |
| 38 | 68 | Death & Mortality |
| 39 | 68 | Land, Agriculture & Forestry |
| 40 | 68 | Communication & Messaging |
| 41 | 63 | Maritime & Naval |
| 42 | 56 | Law & Justice |
| 43 | 54 | Employment & Institutional Power |
| 44 | 53 | Travel & Transportation |
| 45 | 53 | Identity & Fate |
| 46 | 52 | Migration & Displacement |
| 47 | 52 | Music & Performance |
| 48 | 51 | Crafts & Trades |
| 49 | 50 | Vices & Indulgence |
| 50 | 49 | Sexuality & Gender |
| 51 | 47 | Language & Linguistics |
| 52 | 43 | Celebrations & Gatherings |
| 53 | 42 | Eastern Philosophy & Asian History |
| 54 | 41 | Film, Photography & Theatre |
| 55 | 38 | Media & Journalism |
| 56 | 37 | Light, Darkness & Color |
| 57 | 37 | Climate & Energy |
| 58 | 35 | Royal Courts & Dynasties |
| 59 | 35 | Time & Temporality |
| 60 | 34 | Psychic Powers & Perception |
| 61 | 26 | AI & Technological Progress |
| 62 | 24 | Magic & Fairy Tales |
| 63 | 16 | Wounds & Battlefield Medicine |
| 64 | 15 | Historiography |
| 65 | 7 | Fire & Flames |

**Files**:
- `experiments/super_clusters_k5_g0035.json` — raw 66-domain Leiden output
- `experiments/domain_labels_k5_g0035.json` — gemma-3-27b auto-labels (superseded)
- `experiments/domain_labels_final.py` — curated label mapping + JSON generator
- `experiments/domain_labels_final.json` — final curated labels for `load-domains`

**Verdict**: The higher-resolution 66-domain Leiden clustering (k=5, γ=0.0035) produces clean, thematically distinct domains without the mega-domain problem or cross-domain contamination of the old 25-domain approach. The library's 921 books span genuinely diverse subject matter that warrants this granularity.

---

## Phase 8: Community-Level Sweep (Multi-k, Targeting 500-700)

**Date**: Feb 19, 2026
**Goal**: Find optimal k and γ for ~600 communities (intermediate tier between 6,575 clusters and 66 domains).
**Motivation**: The existing 253 communities (k=6, γ=5.5e-5) are too few for 837K topics — 65% had zero primary books, communities averaged 3,310 topics (6.4x larger than demo library communities), and the max community had 20,290 topics.

**Script**: `experiments/community_sweep_v2.py`
**Data**: `experiments/community_sweep_v2_results.json`

### Parameters

| k | GPU cache k | Graph edges | γ range | Leiden time/res | Total sweep |
|---|-------------|-------------|---------|-----------------|-------------|
| 3 | 4 | 1,976,070 | 5e-5 to 5e-4 | 23.7s | 474s |
| 4 | 5 | 2,611,909 | 4e-5 to 4e-4 | 29.5s | 591s |
| 5 | 6 | 3,247,209 | 3e-5 to 3e-4 | 33.4s | 669s |
| 7 | 8 | 4,514,774 | 2e-5 to 2.5e-4 | 44.2s | 884s |
| 10 | 11 | 6,416,705 | 1.5e-5 to 2e-4 | 60.0s | 1200s |

20 log-spaced resolutions per k value. Total runtime: ~63 min on Mac M-series CPU.

### Cross-k Comparison at ~600 Communities

| k | γ | N | Q | Sig (M) | Max | Med | Mean | P10 | P90 | NMI | Singletons |
|---|---|---|---|---------|-----|-----|------|-----|-----|-----|------------|
| 3 | 5.0e-5 | 1,243* | 0.754 | 8.2 | 3,956 | 524 | 674 | 207 | 1,352 | 0.77 | 13 |
| 4 | 4.0e-5 | 648 | 0.734 | 9.1 | 11,617 | 938 | 1,292 | 333 | 2,777 | 0.77 | 13 |
| 5 | 4.9e-5 | 579 | 0.715 | 10.7 | 10,714 | 975 | 1,446 | 321 | 3,160 | 0.79 | 13 |
| 7 | 7.6e-5 | 585 | 0.685 | 14.1 | 12,369 | 1,002 | 1,431 | 267 | 3,207 | 0.82 | 13 |
| 10 | 1.2e-4 | 566 | 0.658 | 18.9 | 16,207 | 1,017 | 1,479 | 250 | 3,133 | 0.85 | 15 |

*k=3 can't reach 600 — its sparsest graph already fragments into 1,243+ communities.

### Per-k Results in 500-700 Range

**k=4** (2.6M edges):

| γ | N | Q | Sig (M) | Max | Med | Mean | P10 | P90 | NMI |
|---|---|---|---------|-----|-----|------|-----|-----|-----|
| 4.00e-5 | 648 | 0.734 | 9.1 | 11,617 | 938 | 1,292 | 333 | 2,777 | 0.77 |

Only one point in range. Max cluster very large (11.6K).

**k=5** (3.2M edges):

| γ | N | Q | Sig (M) | Max | Med | Mean | P10 | P90 | NMI |
|---|---|---|---------|-----|-----|------|-----|-----|-----|
| 4.32e-5 | 507 | 0.719 | 10.5 | 17,593 | 1,126 | 1,652 | 360 | 3,556 | 0.79 |
| 4.87e-5 | 579 | 0.715 | 10.7 | 10,714 | 975 | 1,446 | 321 | 3,160 | 0.79 |
| 5.50e-5 | 652 | 0.713 | 10.9 | 14,332 | 891 | 1,284 | 311 | 2,770 | 0.80 |

**k=7** (4.5M edges):

| γ | N | Q | Sig (M) | Max | Med | Mean | P10 | P90 | NMI |
|---|---|---|---------|-----|-----|------|-----|-----|-----|
| 6.62e-5 | 518 | 0.690 | 13.8 | 16,835 | 1,081 | 1,616 | 289 | 3,344 | 0.81 |
| 7.56e-5 | 585 | 0.685 | 14.1 | 12,369 | 1,002 | 1,431 | 267 | 3,207 | 0.82 |
| 8.63e-5 | 656 | 0.683 | 14.4 | 9,494 | 877 | 1,276 | 274 | 2,795 | 0.83 |

**k=10** (6.4M edges):

| γ | N | Q | Sig (M) | Max | Med | Mean | P10 | P90 | NMI |
|---|---|---|---------|-----|-----|------|-----|-----|-----|
| 1.16e-4 | 566 | 0.658 | 18.9 | 16,207 | 1,017 | 1,479 | 250 | 3,133 | 0.85 |
| 1.33e-4 | 643 | 0.653 | 19.3 | 8,717 | 873 | 1,302 | 241 | 2,918 | 0.86 |

### Key Observations

1. **Significance increases monotonically with k** (8M → 19M): denser graphs detect more statistically significant community structure. This is the strongest quality signal.

2. **NMI stability increases with k** (0.77 → 0.85): denser graphs produce more reproducible partitions across runs.

3. **Modularity Q decreases with k** (0.75 → 0.66): expected — denser random graphs have higher baseline modularity, so absolute Q values drop. This is a normalization artifact, not a quality decline.

4. **All k values produce mega-communities**: Max cluster ranges from 9K-17K at ~600 communities. This is a persistent property of the topic graph — some topics form very dense cliques (e.g., speculative fiction worldbuilding).

5. **Size distributions are similar across k at ~600N**: median ~900-1000, mean ~1300-1500, P90 ~2800-3200. The choice of k doesn't dramatically change the distribution shape.

6. **Singletons are constant** (~13-15): these are isolated nodes in the topic graph regardless of k.

### Analysis: k=7 vs k=10 for Book Discovery

The metrics don't tell the full story. What matters is the user experience: **coherence** (does the community grouping make sense?) and **surprise** (do I discover connections I didn't expect?).

**Why k=10 wins for exploration:**

1. **More connections = more discovery paths.** When a topic connects to 10 nearest neighbors instead of 7, books get linked through more semantic pathways. A book about cooking that discusses food chemistry connects to science books through more routes. This captures the "surprising but valid" connections that make exploration valuable.

2. **Communities are more "earned."** To split a denser graph (6.4M edges) into ~600 communities, Leiden uses a higher γ (1.33e-4 vs 8.6e-5). Each community must have substantially higher internal density than the surrounding graph. The 19.3M significance score means these communities are statistically very "real" — not artifacts of graph sparsity.

3. **Higher NMI (0.86 vs 0.83) = reproducibility over time.** As new books are added to the library and topics reprocessed, k=10 communities absorb new topics into existing communities more consistently rather than reshuffling. This matters for a growing library — community links shared today should still work after the next extraction run.

4. **Significance validates the structure.** 19.3M vs 14.4M means k=10's communities are 34% more statistically significant. Users are seeing genuine thematic clusters, not noise.

**The counterargument for k=7:** Sparser graphs create tighter semantic boundaries. A community about "maritime & naval" won't accidentally absorb "travel & transportation" through bridging connections. For precise categorization (like library shelving), k=7's tighter boundaries would win — but that's what the 6,575 fine-grained clusters are for. Communities serve a different purpose: broad thematic exploration.

**The mega-community problem exists at both k values** and needs a separate solution (see below).

### Decision: k=10, γ=1.33e-4 → ~643 Communities

**Selected**: k=10 with second-pass splitting of mega-communities.

**Rationale**: The three-tier hierarchy uses different graph densities at each level:
- **Fine clusters** (6,575): k=6, γ=0.001 — tight, specific topic groupings
- **Communities** (~600): k=10, γ=1.33e-4 — broad thematic exploration with maximal semantic connectivity
- **Domains** (66): k=5, γ=0.0035 on cluster centroids — high-level categorization

Each tier uses the graph density appropriate to its granularity. Communities are the discovery tier — maximizing connections serves their purpose.

### Two-Pass Mega-Community Splitting Plan

At k=10, γ=1.33e-4, the partition has 643 communities but the largest is 8,717 topics — too broad for coherent browsing. The size distribution (P90=2,918) means roughly the top 10-15% of communities need splitting.

**Approach: Surprise (resolution-free) within subgraphs**

1. Run initial community clustering: k=10, γ=1.33e-4
2. Identify mega-communities above a size threshold (~2,500 topics)
3. For each mega-community: extract the topic subgraph and run `SurpriseVertexPartition` (no γ parameter — Surprise naturally finds the best partition based on the subgraph's own structure)
4. Replace each mega-community with its sub-communities
5. Merge and renumber: final community IDs, update all mapping tables
6. Auto-label with gemma-3-27b (top-30 topics per community → label)
7. Manual review pass: refine labels, merge any that are too similar

**Why Surprise for the second pass?** Surprise is resolution-free — it adapts to each mega-community's internal density without manual γ tuning. A dense speculative fiction mega-community might split into 4-5 sub-communities, while a sparser one might only split into 2. This is the same quality function already used for fine-grained clusters, so it's well-tested on this topic graph.

k=3 and k=4 are ruled out — too sparse, poor significance, and k=3 can't even reach 600.

---

## Phase 9: Community Clustering with `max_comm_size` — Results & Analysis

**Date**: Feb 19, 2026

### Approach: Single-Pass with `max_comm_size`

The two-pass splitting plan from Phase 8 was abandoned after testing revealed fundamental issues:

1. **Surprise (resolution-free) over-splits**: On dense KNN subgraphs, `SurpriseVertexPartition` finds very fine-grained structure — 400-800+ sub-communities per mega-community instead of the expected 3-5.
2. **CPM sub-splitting is hard to tune**: Tested 2x, 5x, and 10x the pass-1 resolution on V2. Results ranged from ineffective (2x: largest sub-community 87% of parent) to extreme (10x: 2,268 total communities).

**Solution**: `leidenalg.find_partition()` has a built-in `max_comm_size` parameter that constrains the optimizer during partitioning. No second pass needed — Leiden respects the cap during its own refinement loop.

```
libtrails cluster --tier community --knn-k 10 --resolution 1.33e-4 --max-community-size 2500
```

### Results: k=10, γ=1.33e-4, max_comm_size=2500

| Metric | Unconstrained | With max_comm_size=2500 |
|--------|---------------|------------------------|
| Communities | 626 | **728** |
| Max size | 10,469 | **2,500** |
| Quality | 6,784,073 | **6,683,630** (1.5% lower) |
| Leiden time | 59s | 57s |
| At cap (2500) | — | 7 communities |

**Size distribution**:

| Percentile | Value |
|------------|-------|
| P10 | 314 |
| P25 | 533 |
| Median | 986 |
| P75 | 1,783 |
| P90 | 2,414 |
| Max | 2,500 |
| Mean | 1,171 |
| Singletons | 13 (junk extraction artifacts) |

### Structural Analysis

**What works well:**
- Size distribution is excellent — no mega-communities, `max_comm_size` cap barely constrains (only 7 at cap)
- All 936 books covered with community assignments, clean primary assignments (no duplicates)
- All 66 domains have communities mapped (median 9 per domain)
- Quality loss from the constraint is negligible (1.5%)
- Several spot-checked communities show good thematic coherence (cooking, writing craft, Dune, romance)

**Critical finding — cluster-community misalignment:**

| Metric | Value |
|--------|-------|
| Clusters in exactly 1 community | 29 (0.4%) |
| Clusters split across 2 communities | 20 (0.3%) |
| Clusters split across 3+ communities | **6,526 (99.3%)** |
| Max communities per cluster | 65 |
| Median majority-community fraction | 62.1% |
| Clusters with ≥50% in majority community | 67.4% |

**Root cause**: Clusters were built on a k=6 KNN graph (3.2M edges) while communities use k=10 (6.4M edges). Different graph densities carve the topic space along fundamentally different boundaries. Clusters cannot nest inside communities because they live on different graphs.

**Downstream effects:**

| Metric | Value | Impact |
|--------|-------|--------|
| Domain coherence (median) | 53.3% | Community→domain mapping is noisy (issue #57) |
| Communities with <30% domain coherence | 127 (17.8%) | Significant minority are poorly assigned |
| Communities with 0 primary books | 345 (47.4%) | Many communities lack a "flagship" book |
| Communities per book (median) | 252 | Books spread very broadly across communities |
| Communities per book (P10) | 90 | Even niche books appear in many communities |
| Primary books per community (median) | 1 | Most communities have ≤1 primary book |

**13 unmapped communities**: All singletons (1 topic each) from orphan clusters 6562-6574 with no domain assignment. Extraction artifacts — malformed passage quotes, misspelled names, place names. Not real communities.

### Recommendation: Re-run at k=6

The structural misalignment is the dominant quality issue. Running communities on the same k=6 graph as clusters would:

1. **Enable clean nesting** — clusters become subsets of communities (same graph, different resolution)
2. **Improve domain coherence** — the hierarchy becomes a clean chain: domains → communities → clusters
3. **Reduce "zero primary books"** — community boundaries aligned with cluster structure better capture book-level patterns
4. **Simplify the architecture** — one graph for both tiers instead of two

**Tradeoffs**: k=6 has lower significance (~11M vs 19M) and NMI stability (0.80 vs 0.86). However:
- Significance is not comparable across graph densities (the doc notes this in Phase 8)
- NMI 0.80 is still robust
- The structural benefit of nesting outweighs the statistical metrics

**Next step**: Dry run at k=6 with `--max-community-size 2500` to find the right γ for ~600-700 communities and verify nesting improvement. Based on Phase 8 sweep data, k=5 at γ=5.5e-5 gave 652 communities, so k=6 should be in a similar range (γ≈6-7e-5).
