# Domain (Super-Cluster) Generation Methodology

This document describes the process used to generate thematic domains from Leiden topic clusters. The methodology evolved through several iterations to address quality issues.

## Pipeline Overview

```
Topics (153,040)
    ↓ Leiden clustering (with hub removal)
Leiden Clusters (1,284)
    ↓ K-means on cluster centroids
Super-clusters (34)
    ↓ Sub-clustering catch-alls + merging similar domains
Final Domains (29)
```

## Problem: The Mega-Cluster Issue

### Initial Attempt (No Hub Removal)

Running Leiden clustering with `resolution=0.001` produced 669 clusters, but **cluster 0 contained 10,100 topics (6.6% of all topics)** with extremely diverse content:

| Topic | Occurrences |
|-------|-------------|
| relationships | 35,458 |
| conflict | 18,118 |
| space exploration | 4,175 |
| artificial intelligence | 3,054 |
| philosophy | 4,359 |

This mega-cluster was mapped to "Personal Journeys", causing severe misclassification. Topics like "space exploration" and "artificial intelligence" ended up in the wrong domain.

### Root Cause: Hub Topics

High-degree "hub" topics (like "relationships", "conflict", "family") create transitivity chains in the graph, pulling unrelated content together. The topic "relationships" connects to almost everything, creating shortcuts that merge disparate clusters.

## Solution: Hub Removal

### Approach

1. **Identify hubs**: Topics in the top 5% by degree (>32 edges)
2. **Remove hubs** before running Leiden clustering
3. **Cluster non-hub topics** to get coherent groups
4. **Reassign hubs** to their most common neighbor cluster post-hoc

### Command Used

```bash
uv run libtrails cluster --remove-hubs --hub-percentile 95 --resolution 0.002
```

### Results

| Metric | Before (no hubs) | After (hub removal) |
|--------|------------------|---------------------|
| Resolution | 0.001 | 0.002 |
| Clusters | 669 | 1,284 |
| Largest cluster | 10,100 (6.6%) | 493 (0.3%) |
| Hubs removed | 0 | 7,546 |

The mega-cluster problem was eliminated.

## Super-Cluster Generation

### Initial K-means (30 clusters)

With 1,284 coherent Leiden clusters, we applied K-means to group them into super-clusters based on robust centroids.

**Robust Centroid Calculation:**
- Filter topics with labels < 4 characters (noise like "a", "the")
- Take top 15 topics by occurrence count
- Weight by `log1p(occurrence_count)` for stability

```bash
uv run libtrails regenerate-domains -n 30
```

### Catch-All Detection

Two super-clusters emerged as catch-alls (mixed content):

| Super-cluster | Size | Content Analysis |
|---------------|------|------------------|
| 24 | 132 | Fantasy + Philosophy + Classics + Noise |
| 29 | 73 | Tech + Cooking + Games + Physics |

**Super-cluster 24 sample topics:**
- allomancy (Mistborn fantasy)
- nietzsche (philosophy)
- dostoevsky (literature)
- scallops (random noise)

**Super-cluster 29 sample topics:**
- fire, grilling (cooking)
- chess strategy (games)
- robotics, automation (technology)
- time, energy (physics)

These were not coherent domains but residual catch-alls.

## Sub-Clustering Catch-Alls

Rather than globally increasing super-clusters (which might shuffle good clusters), we surgically split the two catch-alls:

```python
catchall_splits = {
    24: 4,  # Fantasy+Philosophy → 4 sub-groups
    29: 3   # Tech+Cooking → 3 sub-groups
}

new_super_clusters = split_catchall_superclusters(
    super_clusters,
    catchall_splits
)
```

### Results of Sub-Clustering

**Super-cluster 24 split into:**
| New ID | Size | Auto-label |
|--------|------|------------|
| 30 | 55 | glyphs / allomancy / nietzsche |
| 31 | 36 | warriors / hell / home |
| 32 | 40 | aes sedai / oasis / data collection |
| 33 | 1 | holograms (tiny outlier) |

**Super-cluster 29 split into:**
| New ID | Size | Auto-label |
|--------|------|------------|
| 34 | 29 | fire / grilling / communication |
| 35 | 23 | time / human interaction / typography |
| 36 | 21 | robotics / engineering / construction |

### Manual Adjustment

Cluster 33 (1 cluster, holograms) was merged into cluster 36 (Engineering & Robotics) as it was too small to be a standalone domain and thematically related.

**Final: 34 super-clusters**

## Domain Label Refinement

### Strategic Merges

To reduce 34 super-clusters to a manageable ~25-30 domains, similar clusters were merged:

| Domain | Merged Super-clusters | Total Clusters |
|--------|----------------------|----------------|
| **Fantasy & Speculative** | 20, 30, 31, 32 | 167 |
| **Culinary Arts** | 13, 34 | 87 |
| **Warfare & Military** | 14, 28 | 78 |

### Final REFINED_LABELS Mapping

```python
REFINED_LABELS = {
    # Core domains (no merge)
    0: "History & Archaeology",
    1: "Logic & Mathematics",
    2: "Politics & Power",
    5: "AI & Machine Learning",
    9: "Space & Science",
    # ... (26 more)

    # Merged domains
    13: "Culinary Arts",
    34: "Culinary Arts",  # fire/grilling → merge

    14: "Warfare & Military",
    28: "Warfare & Military",  # combat → merge

    20: "Fantasy & Speculative",
    30: "Fantasy & Speculative",
    31: "Fantasy & Speculative",
    32: "Fantasy & Speculative",
}
```

## Final Domain Structure

**29 domains** covering 1,284 Leiden clusters:

| ID | Clusters | Domain |
|----|----------|--------|
| 0 | 167 | Fantasy & Speculative |
| 1 | 87 | Culinary Arts |
| 2 | 78 | Warfare & Military |
| 3 | 62 | AI & Machine Learning |
| 4 | 60 | Historical Drama |
| 5 | 58 | Financial Strategy |
| 6 | 49 | Education & Class |
| 7 | 48 | Religion & Philosophy |
| 8 | 48 | Nature & Travel |
| 9 | 47 | Technology & Data |
| 10 | 44 | Family & Relationships |
| 11 | 42 | History & Archaeology |
| 12 | 41 | Politics & Power |
| 13 | 39 | Inner Landscapes |
| 14 | 38 | Literature & Poetry |
| 15 | 37 | Leadership & Strategy |
| 16 | 36 | Arts & Society |
| 17 | 36 | Identity & Dreams |
| 18 | 36 | Conflict & Emotion |
| 19 | 35 | Survival & Mortality |
| 20 | 33 | World Cultures |
| 21 | 32 | Nature & Agriculture |
| 22 | 31 | Logic & Mathematics |
| 23 | 23 | Time & Communication |
| 24 | 22 | Engineering & Robotics |
| 25 | 19 | Crime & Suspense |
| 26 | 18 | Space & Science |
| 27 | 9 | Espionage & Security |
| 28 | 9 | Architecture & Design |

## Validation

Key topics now correctly assigned:

| Topic | Occurrences | Domain |
|-------|-------------|--------|
| space exploration | 4,175 | Space & Science ✓ |
| artificial intelligence | 3,054 | AI & Machine Learning ✓ |
| cooking techniques | 1,746 | Culinary Arts ✓ |
| harry potter | 1,369 | Fantasy & Speculative ✓ |
| robotics | 1,035 | Engineering & Robotics ✓ |
| allomancy | 309 | Fantasy & Speculative ✓ |

**Before** (no hub removal): All were misclassified to "Personal Journeys"
**After**: Each in its proper thematic domain

## Key Lessons

1. **Hub removal is essential** for meaningful clustering at scale
2. **Resolution 0.002** with hub removal produces coherent clusters without mega-clusters
3. **K-means on centroids** works well for super-clustering, but produces 2-3 catch-all groups
4. **Sub-clustering catch-alls** is more surgical than globally increasing K
5. **Manual domain merging** reduces redundancy while preserving semantic coherence

## Reproducing This Pipeline

```bash
# 1. Run Leiden with hub removal
uv run libtrails cluster --remove-hubs --hub-percentile 95 --resolution 0.002

# 2. Generate super-clusters
uv run libtrails regenerate-domains -n 30

# 3. Inspect for catch-alls (look for clusters >100 with mixed content)
# 4. Sub-cluster catch-alls in Python (see split_catchall_superclusters)
# 5. Update REFINED_LABELS in experiments/domain_labels_final.py
# 6. Generate final domains
uv run python experiments/domain_labels_final.py

# 7. Load into database
uv run libtrails load-domains
```

## Files

| File | Purpose |
|------|---------|
| `src/libtrails/domains.py` | Domain generation functions |
| `src/libtrails/clustering.py` | Leiden with hub removal |
| `experiments/super_clusters_robust.json` | Super-cluster assignments |
| `experiments/domain_labels_final.py` | REFINED_LABELS mapping |
| `experiments/domain_labels_final.json` | Final domain structure |
