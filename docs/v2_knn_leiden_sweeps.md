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

## Phase 3: min_similarity Sweep

*TODO: Run at best k + resolution*
