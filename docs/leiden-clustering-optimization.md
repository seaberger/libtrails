# Leiden Clustering Optimization

## Summary

Successfully optimized topic clustering from **80+ minute timeouts** to **~3 minutes** with high-quality results producing **~340 coherent topic clusters** from 108,668 topics.

---

## Final Configuration

```bash
# Optimized defaults (now built-in)
uv run libtrails cluster --skip-cooccur

# Explicit settings
uv run libtrails cluster \
  --skip-cooccur \
  --mode knn \
  --knn-k 10 \
  --min-cooccur 2 \
  --partition-type cpm \
  --resolution 0.001
```

### Config Settings (`config.py`)

```python
# Clustering defaults (optimized for ~300-400 coherent clusters)
CLUSTER_MODE = "knn"            # "cooccurrence", "knn", or "full"
CLUSTER_KNN_K = 10              # k neighbors for knn mode
CLUSTER_PARTITION_TYPE = "cpm"  # "modularity", "surprise", or "cpm"
CLUSTER_RESOLUTION = 0.001      # Resolution for CPM (lower = fewer clusters)
COOCCURRENCE_MIN_COUNT = 2      # Lowered for better connectivity
```

---

## Problem Statement

### Original Issue
- **108,668 topics** with embeddings
- **Embedding similarity**: O(n²) = 5.9 billion pairs to check
- **Co-occurrence edges**: Only 34k with count ≥ 5
- **Result**: 80+ minute runs that never completed

### Root Cause
The `build_topic_graph()` function computed a full pairwise similarity matrix:
```python
sim_matrix = np.dot(embeddings, embeddings.T)  # 108k × 108k = 47 GB RAM
```

---

## Solution: Three Graph Construction Modes

### Mode 1: `cooccurrence` (Fastest)
- Uses only co-occurrence edges from chunks
- ~30k-150k edges depending on threshold
- Completes in seconds
- **Problem**: 99.6% of topics end up as singletons (isolated)

### Mode 2: `knn` (Recommended)
- Co-occurrence edges + k-nearest neighbor embedding edges
- ~940k edges with k=10
- Completes in ~3 minutes
- **Best balance** of speed and cluster quality

### Mode 3: `full` (Original - Slow)
- All pairwise embedding similarity above threshold
- Millions of edges
- Often times out or runs out of memory
- Not recommended for large topic sets

---

## Implementation Details

### New Functions Added

**`topic_graph.py`:**
- `build_topic_graph_cooccurrence_only()` - Fast co-occurrence only
- `build_topic_graph_knn()` - k-NN with sklearn NearestNeighbors

**`clustering.py`:**
- Updated `cluster_topics()` with mode, partition_type, resolution, knn_k params

**`cli.py`:**
- Added `--mode`, `--partition-type`, `--resolution`, `--knn-k`, `--skip-cooccur` flags

### Timing Breakdown (108k topics, knn mode)

| Step | Time |
|------|------|
| Load embeddings from SQLite | ~20s |
| Build k-NN index (sklearn) | ~90s |
| Construct igraph | ~20s |
| **Leiden clustering** | **~5s** |
| Save to database | ~10s |
| **Total** | **~2.5 min** |

---

## Partition Type Comparison

| Type | Clusters | Quality | Notes |
|------|----------|---------|-------|
| `modularity` | 33 | 0.66 | Very broad categories |
| `surprise` | varies | - | Original, no tuning |
| `cpm` (res=0.001) | ~340 | 1.1M | **Best granularity** |
| `cpm` (res=0.0005) | ~170 | 1.2M | Larger clusters |
| `cpm` (res=0.005) | ~1,200 | 1.0M | Too granular |

**Recommendation**: Use CPM with resolution 0.001 for ~300-400 coherent clusters.

---

## Co-occurrence Threshold Analysis

| Threshold | Edge Count | Effect |
|-----------|------------|--------|
| count ≥ 10 | 13,501 | Very sparse |
| count ≥ 5 | 34,108 | Original default |
| count ≥ 3 | 73,283 | Moderate |
| count ≥ 2 | 156,974 | **Recommended** |

Lower threshold = more connections = fewer isolated topics.

---

## Cluster Quality Results

### Final Distribution (341 clusters)
| Size Range | Count | % |
|------------|-------|---|
| Small (11-50) | 4 | 1.2% |
| Medium (51-200) | 100 | 29.3% |
| Large (201-1000) | 227 | 66.6% |
| Very Large (1000+) | 10 | 2.9% |

**Zero singletons** - all topics grouped meaningfully.

### Sample Clusters (High Coherence)
| Cluster | Size | Top Topics |
|---------|------|------------|
| relationships | 3,097 | conflict, travel, family, crime |
| cooking techniques | 2,895 | baking, italian cuisine, food prep |
| machine learning | 2,040 | neural nets, deep learning, pytorch |
| risk management | 1,975 | financial markets, investment |
| probability | 894 | Bayesian, distributions |
| plant physiology | 850 | photosynthesis, plant biology |
| linear algebra | 703 | geometry, vectors, matrices |
| data structures | 683 | algorithms, search |

---

## CLI Usage Examples

```bash
# Use optimized defaults
uv run libtrails cluster --skip-cooccur

# Fewer, larger clusters (~170)
uv run libtrails cluster --skip-cooccur --resolution 0.0005

# More, smaller clusters (~1000)
uv run libtrails cluster --skip-cooccur --resolution 0.005

# Fast sparse clustering (many singletons)
uv run libtrails cluster --skip-cooccur --mode cooccurrence

# View results
uv run libtrails tree
uv run libtrails status
```

### The `--skip-cooccur` Flag

Co-occurrence computation is expensive (~5 min for 178k chunks). Once computed, the data is saved to `topic_cooccurrences` table. Use `--skip-cooccur` to reuse existing data when experimenting with different clustering parameters.

---

## Key Learnings

1. **k-NN is the sweet spot**: Provides semantic connections without O(n²) complexity
2. **Co-occurrence alone is too sparse**: Most topics only appear in 1-2 books
3. **CPM partition with tunable resolution** gives best control over cluster count
4. **Resolution 0.001** produces ~300-400 coherent clusters for our 108k topics
5. **min-cooccur 2** provides better connectivity than the original 5
6. **Graph building dominates runtime**, not Leiden itself

---

## Future Improvements

### Not Implemented (Lower Priority)
- **FAISS/Annoy**: Approximate k-NN could speed up graph building
- **Two-stage clustering**: Refine large clusters with embeddings
- **Graph sparsification**: Reduce edges while preserving structure
- **Hierarchical clustering**: Recursive sub-clustering of large groups

### Potential Enhancements
- Cache k-NN graph to disk for faster restarts
- Parallel k-NN computation with joblib
- LLM-generated cluster labels (already partially implemented)

---

## References

- [leidenalg Documentation](https://leidenalg.readthedocs.io/en/stable/intro.html)
- [From Louvain to Leiden - Nature](https://www.nature.com/articles/s41598-019-41695-z)
- [sklearn NearestNeighbors](https://scikit-learn.org/stable/modules/neighbors.html)
- [igraph Python](https://python.igraph.org/en/stable/)
