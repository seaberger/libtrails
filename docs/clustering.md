# Topic Clustering in LibTrails

LibTrails uses Leiden clustering (CPM variant) with hub removal to organize 108K+ topics into coherent, browsable themes.

## The Problem: Hub Transitivity

Generic topics like "relationships", "technology", and "conflict" appear across many books and create **transitivity chains** that merge unrelated themes:

```
AI ethics → technology → smartphones → relationships → romance novels
furniture → interior design → decoration → python decorators → programming
```

At low resolution, the clustering algorithm says "these are all reachable, must be one community" - creating mega-clusters with 2,000+ unrelated topics.

## The Solution: Hub Removal

We identify and temporarily remove high-degree "hub" topics before clustering:

1. **Build the topic graph** (co-occurrence + k-NN embedding edges)
2. **Identify hubs** - topics in the top 5% by connection count
3. **Cluster without hubs** - run Leiden on the remaining 95%
4. **Reassign hubs** - each hub joins its most-connected cluster

This breaks transitivity chains while preserving hub topics in results.

### Hub Detection Methods

```bash
# By degree (top N% most connected)
libtrails cluster --remove-hubs --hub-percentile 95

# By generic term patterns
libtrails cluster --remove-hubs --hub-method generic

# Both methods combined
libtrails cluster --remove-hubs --hub-method both
```

### Diagnosing Hubs

```bash
libtrails diagnose-hubs --top-n 50
```

Shows degree distribution and top hub topics. Example output:

```
Degree Distribution (connections per topic)
 Minimum          10
 Median           13
 Mean             17.3
 95th percentile  31
 Maximum          2045   # "relationships" - 157x median!

Top Hub Topics (by degree)
  2045 edges: relationships
  1928 edges: family relationships
  1405 edges: travel
  1378 edges: conflict
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--resolution` | 0.002 | CPM resolution (higher = more clusters) |
| `--hub-percentile` | 95 | Remove top N% by degree |
| `--partition-type` | cpm | Leiden partition type |
| `--mode` | knn | Graph construction (knn, cooccurrence, full) |

### Resolution Guide

| Resolution | Clusters | Max Size | Use Case |
|------------|----------|----------|----------|
| 0.001 | ~474 | ~900 | Broad themes |
| 0.002 | ~845 | ~500 | Balanced (recommended) |
| 0.005 | ~1,600 | ~250 | Focused themes |
| 0.01+ | 3,000+ | <150 | Very granular |

## Results

With `--resolution 0.002 --remove-hubs --hub-percentile 95`:

- **845 coherent theme clusters**
- **Largest cluster: 506 topics** (down from 2,927 before hub removal)
- **Books appear in 5-50 relevant clusters** (down from 200+)

Example clusters:
- Cluster 0: Financial markets, investing, stock analysis
- Cluster 1: Ethics, morality, human nature
- Cluster 8: Space exploration, spacecraft
- Cluster 10: Cheese, dairy, food culture

## CLI Commands

```bash
# Run clustering with hub removal
libtrails cluster --remove-hubs --hub-percentile 95 --resolution 0.002

# Dry run to test parameters
libtrails cluster --remove-hubs --dry-run --resolution 0.003

# Skip co-occurrence recomputation (faster)
libtrails cluster --remove-hubs --skip-cooccur

# See which clusters a book belongs to
libtrails book-clusters --title "Anathem"

# Browse cluster hierarchy
libtrails tree
```

## Technical Details

### Graph Construction

The topic graph has two edge types:

1. **Co-occurrence edges**: Topics appearing together in the same chunk
   - Weighted by PMI (Pointwise Mutual Information)
   - ~142K edges

2. **k-NN embedding edges**: Topics with similar BGE embeddings
   - k=10 nearest neighbors per topic
   - ~797K edges

### Why Leiden over Louvain?

Leiden provides stronger guarantees:
- Clusters are always connected (no disconnected subcommunities)
- Better quality in fewer iterations
- CPM variant allows explicit resolution tuning

The hub removal technique is algorithm-agnostic - it's graph preprocessing that works with any community detection method.

### Memory Requirements

| Dataset Size | Graph Memory | Leiden Time |
|--------------|--------------|-------------|
| 108K topics | ~3.4 GB | ~45 seconds |

The graph building step is the memory bottleneck. Co-occurrence computation can spike to 20-40 GB for large libraries.
