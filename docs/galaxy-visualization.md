# Galaxy/Universe Visualization

A stunning 2D projection of all topic clusters, creating a visual "galaxy" of knowledge where semantically similar themes appear close together spatially.

## Concept

```
                    ★ Quantum Physics
              ★ Cosmology
         ★ Astrophysics          ★ Statistics
                                      ★ Machine Learning
    ★ Philosophy of Science              ★ Neural Networks

  ★ Ethics                    ★ Linear Algebra
       ★ Stoicism
                         ★ Economics
    ★ Eastern Philosophy              ★ Finance
```

Each star represents a Leiden cluster. Position is determined by UMAP projection of the cluster's centroid embedding. Color indicates the super-cluster (domain) it belongs to.

## Data Generation

### Step 1: Robust Centroids

For each of 845 Leiden clusters, compute a robust centroid:

```python
def compute_robust_centroid(cluster_id, top_n=15, min_label_length=4):
    """
    Robust centroid approach:
    - Filter topics with labels < 4 chars (weak embedding signal)
    - Take top N topics by occurrence count
    - Weight by log1p(occurrence_count) for stability
    """
    topics = get_cluster_topics(cluster_id)
    topics = [t for t in topics if len(t['label']) >= min_label_length]
    topics = sorted(topics, key=lambda t: t['occurrence_count'], reverse=True)[:top_n]

    if len(topics) < 3:
        return None

    embeddings = [t['embedding'] for t in topics]
    weights = np.array([np.log1p(t['occurrence_count']) for t in topics])
    weights = weights / weights.sum()

    return np.average(embeddings, axis=0, weights=weights)
```

### Step 2: UMAP Projection

Project 384-dimensional centroids to 2D:

```python
from umap import UMAP

umap = UMAP(
    n_components=2,
    n_neighbors=15,      # Balance local vs global structure
    min_dist=0.3,        # How tightly points cluster
    metric='cosine',     # Match embedding similarity metric
    random_state=42,     # Reproducibility
)

coords_2d = umap.fit_transform(centroid_matrix)

# Normalize to [0, 1] for frontend
coords_2d[:, 0] = (coords_2d[:, 0] - coords_2d[:, 0].min()) / (coords_2d[:, 0].max() - coords_2d[:, 0].min())
coords_2d[:, 1] = (coords_2d[:, 1] - coords_2d[:, 1].min()) / (coords_2d[:, 1].max() - coords_2d[:, 1].min())
```

### UMAP Parameter Tuning

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `n_neighbors` | Local structure, tight clusters | Global structure, spread out |
| `min_dist` | Points can overlap | Points spread apart |
| `spread` | Compact overall | Expanded overall |

Recommended experiments:
```python
# Tighter, more defined regions
UMAP(n_neighbors=10, min_dist=0.05, spread=1.0)

# More spread, see individual points
UMAP(n_neighbors=20, min_dist=0.2, spread=1.5)
```

## Data Files

### experiments/universe_coords.json

```json
{
  "clusters": [
    {
      "cluster_id": 0,
      "label": "financial markets",
      "size": 506,
      "book_count": 164,
      "domain_id": 1,
      "domain_label": "risk management / finance / risk assessment",
      "x": 0.348,
      "y": 0.975
    }
  ],
  "domains": [
    {"domain_id": 1, "label": "risk management / finance / risk assessment", "color": "hsl(0, 65%, 55%)"}
  ]
}
```

### Current Results (Feb 5, 2025)

- **845 clusters** projected to 2D
- **25 domains** (super-clusters) for color coding
- Coordinates normalized to [0, 1]

Sample spatial relationships:
- "financial markets" (0.35, 0.97) - top center
- "cheese" and "food preservation" (0.93-0.98, 0.64) - right side, near each other
- "linear algebra" (0.05, 0.77) - left side, ML domain
- "political philosophy" (0.24, 0.18) - lower left

## API Endpoint

```
GET /api/v1/universe
```

Returns the full `universe_coords.json` structure for the frontend.

## Frontend Implementation

### Technology Choice

**Recommended: D3.js** for balance of interactivity and simplicity.

Alternatives:
- **Canvas**: Better performance for 1000+ points
- **WebGL/Three.js**: 3D rotation, particle effects
- **SVG**: Simple, accessible, good for static views

### React Component Structure

```tsx
// components/GalaxyView.tsx

export function GalaxyView({ clusters, domains }) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredCluster, setHoveredCluster] = useState(null);
  const [zoom, setZoom] = useState(1);

  // D3 setup with:
  // - Color scale by domain
  // - Size scale by book_count (sqrt for area perception)
  // - Zoom/pan behavior
  // - Hover tooltips
  // - Click navigation to cluster detail
}
```

### Visual Design

**Dark mode "night sky" aesthetic:**
```css
.galaxy-container {
  background: radial-gradient(ellipse at center, #1a1a2e 0%, #0f0f1a 100%);
}

.cluster-point {
  filter: drop-shadow(0 0 3px currentColor);  /* Glow effect */
}
```

**Interaction states:**
- **Default:** Semi-transparent circles, sized by book count
- **Hover:** Brighten, show tooltip with theme details
- **Click:** Navigate to cluster detail page
- **Zoom:** Labels appear at 1.5x+, fine details at 3x+

### Features

1. **Core**
   - Pan and zoom
   - Hover tooltips with cluster name, book count, top topics
   - Click to navigate to cluster detail
   - Color legend for domains

2. **Enhanced**
   - Labels appear on zoom
   - Domain filter toggles
   - Search highlights matching clusters
   - Smooth animations

3. **Optional Polish**
   - Parallax floating animation
   - Connecting lines between related clusters
   - Star/glow particle effects

## Regenerating Data

```bash
# Generate super-cluster assignments (25 domains)
uv run python experiments/super_clusters_robust.py

# Generate UMAP coordinates
uv run python experiments/umap_universe.py
```

## Open Questions

1. **Star sizing**: By book count or topic count?
2. **Connections**: Show edges between related clusters?
3. **Label overlap**: How to handle crowded regions?
4. **Mobile**: Simplified view or full interactivity?
