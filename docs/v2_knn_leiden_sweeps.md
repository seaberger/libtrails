# V2 KNN & Leiden Sweep Results

## Database: `ipad_library_v2.db`
- **837,320 topics** with embeddings (384-dim BGE-small)
- **936 books**, 335K chunks
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
