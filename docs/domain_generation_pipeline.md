# Domain Generation Pipeline

How to generate and manage the three-tier topic hierarchy for the V2 library.

---

## Overview

The topic hierarchy groups 837K extracted topics into three browsable tiers using Leiden CPM at different resolutions on the same topic KNN graph.

```
Topics (837K)
    → Topic KNN graph (k=6, 6.4M edges, cached to disk)
    → Leiden CPM (γ=0.001)    → 6,575 clusters   (search neighborhoods)
    → Leiden CPM (γ=5.5e-5)   → ~250 communities  (Topics tab, Universe)
    → Cluster-level KNN graph  → 29 domains        (Themes tab)

Weighted membership pipeline (refresh-stats):
    cluster_books → book_communities → community_stats
                  → book_domains     → domain_stats
```

### Three-Tier Architecture

| Tier | Count | Resolution | User Sees | Purpose |
|------|-------|-----------|-----------|---------|
| **Domains** | 29 | γ=0.0008 (cluster graph) | Themes tab | Broad theme browsing |
| **Communities** | ~250 | γ=5.5e-5 (topic graph) | Topics tab, Universe | Specific topic exploration |
| **Clusters** | 6,575 | γ=0.001 (topic graph) | Search results only | Search neighborhoods |

See `docs/topic_hierarchy_architecture.md` for design rationale and threshold analysis.

---

## Step 1: Generate Super-Clusters

```bash
LIBTRAILS_DB=v2 uv run libtrails regenerate-domains \
    --method leiden \
    --cluster-knn-k 4 \
    --resolution 0.0008 \
    --output experiments/super_clusters_v2_k4_g0008.json
```

**Hyperparameters** (see `docs/v2_knn_leiden_sweeps.md` Phase 3 for full rationale):
- **k=4**: Cluster graph density. Swept k=2-10; k=4 chosen for tightest domain count stability (std=0.9) and high significance
- **gamma=0.0008**: Leiden CPM resolution. Swept 0.0001-0.005 across k=2,3,4; gives ~25 domains with Q=0.738
- **min_similarity=0.3**: Cluster graph edge threshold (default)
- **Outlier reassignment**: Enabled (participation coefficient > 0.7)

### Split the Mega-Domain

The largest domain typically contains 1,200+ clusters (mixed fiction catch-all). Split it using `split_catchall_superclusters()`:

```python
from libtrails.domains import generate_super_clusters_leiden, split_catchall_superclusters

result = generate_super_clusters_leiden(
    resolution=0.0008,
    k=4,
    min_similarity=0.3,
    remove_outliers=True,
)
scs = result['super_clusters']

# Split mega-domain (ID 0) into 5 sub-groups
scs = split_catchall_superclusters(scs, {0: 5})
# Result: ~29 domains
```

Or do it interactively to preview different split counts (4, 5, 6) before choosing.

---

## Step 2: Enrich with Book/Topic Counts

Before labeling, query the database for each domain's book count and topic count:

```python
for sc in scs:
    cluster_ids = [lc['cluster_id'] for lc in sc['leiden_clusters']]
    placeholders = ','.join('?' * len(cluster_ids))

    # Topic count
    sc['topic_count'] = conn.execute(f'''
        SELECT COUNT(*) FROM topics WHERE cluster_id IN ({placeholders})
    ''', cluster_ids).fetchone()[0]

    # Book count (distinct books via chunk_topic_links)
    sc['book_count'] = conn.execute(f'''
        SELECT COUNT(DISTINCT b.id)
        FROM books b
        JOIN chunks c ON c.book_id = b.id
        JOIN chunk_topic_links ctl ON ctl.chunk_id = c.id
        JOIN topics t ON t.id = ctl.topic_id
        WHERE t.cluster_id IN ({placeholders})
    ''', cluster_ids).fetchone()[0]
```

---

## Step 3: LLM Auto-Label

Use `experiments/label_domains.py` (or adapted version below) to generate domain names.

**LM Studio** (gemma-3-27b on localhost:1234):

```python
import httpx

API_URL = 'http://localhost:1234/v1/chat/completions'
MODEL = 'google/gemma-3-27b'

def generate_domain_name(topics: list[str]) -> str:
    topics_str = ', '.join(topics[:20])
    prompt = f'''You are naming categories for a book library. Given these related topics:

{topics_str}

Generate a single concise category name (2-4 words) that captures the overall theme.
Use title case. Be specific but not too narrow. Just output the category name.'''

    response = httpx.post(API_URL, json={
        'model': MODEL,
        'messages': [{'role': 'user', 'content': prompt}],
        'temperature': 0.3,
        'max_tokens': 30,
    }, timeout=30.0)
    name = response.json()['choices'][0]['message']['content'].strip()
    return name.split('\n')[0].strip('"\'`*')
```

**Output**: `experiments/domain_labels_v2_llm.json` — each entry has `domain_id`, `cluster_count`, `topic_count`, `book_count`, `llm_label`, `auto_label`, `top_topics`.

---

## Step 4: Manual Review & Curation

Review the LLM-generated labels alongside the top topics, book counts, and cluster counts. Things to look for:

1. **Vague labels** (e.g., "Mind & Potential") — rename to something more descriptive
2. **Overlapping domains** (e.g., three "Family & ..." domains) — consider merging
3. **Mismatched labels** (label doesn't match top topics) — rename
4. **Domains that are too small** (<50 clusters) — consider merging into a neighbor

Update the `REFINED_LABELS` dict in `experiments/domain_labels_final.py`:

```python
REFINED_LABELS = {
    1: "Human Psychology",
    25: "Science Fiction & Fantasy",
    2: "Science & Technology",
    3: "War & Military",
    # ... etc
}
```

Then generate the final JSON:

```bash
uv run python experiments/domain_labels_final.py
```

This produces `experiments/domain_labels_final.json` with the curated labels, merging any domains that share the same label.

---

## Step 5: Load into Database

```bash
LIBTRAILS_DB=v2 uv run libtrails load-domains -f experiments/domain_labels_final.json
```

This populates the `domains` table and updates `cluster_stats` with domain assignments.

---

## V2 Domain Summary (Feb 2026)

| Metric | Value |
|--------|-------|
| Leiden clusters | 6,575 |
| Cluster graph | k=4, 6,562 nodes, 21,550 edges |
| Leiden resolution | gamma=0.0008 (CPM) |
| Initial domains | 25 |
| After mega-split | 29 |
| Q (modularity) | 0.738 |
| Significance | 31,834 |
| Domain count stability | 25.3 ± 0.9 (30 seeds) |
| LLM labeling model | gemma-3-27b (LM Studio) |

---

## Phase 6: Weighted Domain Membership

### The Problem

After loading 29 domains, the themes page showed **zero differentiation**: 26 of 29 domains listed 90-100% of all 937 books. Even the most specific domain (Russian Literature, 32 clusters) included 426 books — but 87% of them had <1% of their topics there. "On Cooking" and "Mistborn" showed up as Russian Literature books.

**Root cause**: Binary "any topic link" membership. A typical book has ~1,000 topic mentions across hundreds of clusters. With binary membership, every book touches every domain.

### The Solution: Concentration Scoring

New `book_domains` bridge table materializes each book's **concentration per domain** — pre-computed during `refresh-stats`, not on every API request.

```sql
CREATE TABLE book_domains (
    book_id INTEGER NOT NULL,
    domain_id INTEGER NOT NULL,
    topic_count INTEGER NOT NULL,       -- book's topics in this domain
    book_total_topics INTEGER NOT NULL,  -- book's total topics across all domains
    concentration REAL NOT NULL,         -- topic_count / book_total_topics
    relevance_score REAL NOT NULL,       -- book_cluster_relevance() output
    is_primary INTEGER NOT NULL DEFAULT 0, -- 1 if this is the book's top domain
    PRIMARY KEY (book_id, domain_id)
);
```

**Scoring**: Uses the same `book_cluster_relevance()` function (BM25 saturation + PPMI) that scores books for individual clusters, applied at the domain level with `min_topics=1` (since domains aggregate many clusters).

**Thresholds**:
- `book_count` in `domain_stats` = books with ≥1% concentration (eliminates noise)
- `primary_book_count` = books where this is their highest-scoring domain
- `is_primary` = 1 for each book's single top domain (tie-breaks by first match)

### Implementation

**Files changed** (`feat/weighted-domain-membership` branch):

| File | Change |
|------|--------|
| `config.py` | `COMMUNITY_RESOLUTION = 5.5e-5` |
| `database.py` | `book_domains`, `communities`, `cluster_communities`, `book_communities`, `community_stats` tables; `topics.community_id` column; `domain_stats.primary_book_count` column |
| `clustering.py` | `--tier` parameter (cluster/community); `_populate_community_tables()` for auto-labels + majority vote mappings |
| `stats.py` | `refresh_book_communities()`, `refresh_community_stats()`, `refresh_book_domains()`, rewritten `refresh_domain_stats()` |
| `cli.py` | `--tier` option on `cluster` command |
| `api/schemas.py` | `DomainBook`, `CommunitySummary`, `CommunityBook`, `CommunityDetail` schemas |
| `api/routers/domains.py` | Queries from `book_domains` with concentration/is_primary |
| `api/routers/communities.py` | **New** — list + detail endpoints |
| `web/src/lib/types.ts` | `DomainBook`, `CommunitySummary`, `CommunityBook`, `CommunityDetail` |
| `web/src/lib/api.ts` | `getCommunities()`, `getCommunity()` |
| `web/src/pages/clusters/index.astro` | Renamed to Topics tab, fetches communities |
| `web/src/pages/clusters/[id].astro` | Shows community detail with clusters + books |
| `web/src/pages/themes/index.astro` | Shows "X primary books · Y related" |
| `web/src/layouts/*.astro` | Nav "Clusters" → "Topics" |

**Refresh pipeline order** (in `_refresh_all_stats_impl`):
1. `refresh_cluster_books()` — the 4-table join
2. `refresh_cluster_stats()` — per-cluster stats
3. `refresh_book_communities()` — aggregates cluster_books by community, scores, marks primary
4. `refresh_community_stats()` — per-community stats from book_communities
5. `refresh_book_domains()` — aggregates cluster_books by domain, scores, marks primary
6. `refresh_domain_stats()` — reads from `book_domains` instead of raw cluster_books

### Verification

```bash
# Refresh stats (creates book_domains table and populates it)
LIBTRAILS_DB=v2 uv run libtrails refresh-stats

# Check differentiation — should see Russian Lit with ~20-60 books, not 426
sqlite3 data/ipad_library_v2.db "
  SELECT d.label, ds.book_count, ds.primary_book_count
  FROM domain_stats ds
  JOIN domains d ON d.id = ds.domain_id
  ORDER BY ds.primary_book_count DESC
"

# Spot-check: Russian Lit primary books should actually be Russian literature
sqlite3 data/ipad_library_v2.db "
  SELECT b.title, b.author, bd.concentration, bd.is_primary
  FROM book_domains bd
  JOIN books b ON b.id = bd.book_id
  JOIN domains d ON d.id = bd.domain_id
  WHERE d.label LIKE '%Russian%'
  ORDER BY bd.relevance_score DESC
  LIMIT 10
"
```

---

## Phase 7: Community Generation

Communities are the middle tier (~250 groups) between domains (29) and fine-grained clusters (6,575). They run Leiden CPM at a coarser resolution on the **same topic KNN graph** used for clusters.

### Step 1: Run Community Clustering

```bash
LIBTRAILS_DB=v2 uv run libtrails cluster --tier community
```

This runs Leiden at `γ=5.5e-5` (configured in `config.py` as `COMMUNITY_RESOLUTION`) on the cached topic KNN graph. It:

1. Saves `community_id` to the `topics` table (batch update via `batch_update_topic_communities()`)
2. Creates rows in the `communities` table with auto-generated labels (top topic by occurrence count)
3. Derives `cluster_communities` mapping via **majority vote** — each cluster's community is whichever `community_id` most of its topics belong to
4. Assigns each community to a domain via **majority vote** from `cluster_domains`

**Runtime**: ~2 min on the V2 topic KNN graph (837K topics, 6.4M edges).

### Step 2: Refresh Stats

```bash
LIBTRAILS_DB=v2 uv run libtrails refresh-stats
```

This populates `book_communities` and `community_stats` (concentration scoring, primary marking — same pattern as domains). See Phase 6 refresh pipeline order above.

### Step 3: Verify

```bash
# Community count and stats
sqlite3 data/ipad_library_v2.db "
  SELECT COUNT(*) as communities FROM communities;
"

# Top communities by primary book count
sqlite3 data/ipad_library_v2.db "
  SELECT label, topic_count, book_count, primary_book_count, domain_label
  FROM community_stats
  ORDER BY primary_book_count DESC
  LIMIT 15
"

# API check
curl localhost:8000/api/v1/communities | python -m json.tool | head -30
```

### Community Labels

Currently auto-generated from the highest-occurrence topic in each community. LLM-generated labels (using gemma-3-27b, same process as domain labels) are planned for a future pass.

### Resolution Selection

`γ=5.5e-5` was chosen from a 25-point sweep (see `docs/topic_hierarchy_architecture.md`):
- Phase 2 of the CPM landscape (~175-230 communities at γ ≈ 4e-5 to 5.5e-5)
- Stable ascending region — monotonically increasing cluster count from γ=1e-5 onward
- Higher resolution = tighter, more semantically coherent communities than Phase 1 (γ ≈ 3e-6)

### New Tables

| Table | Purpose |
|-------|---------|
| `communities` | `id`, `label`, `domain_id`, `topic_count` |
| `cluster_communities` | Maps each cluster to its parent community |
| `book_communities` | Per-book community membership (concentration, relevance_score, is_primary) |
| `community_stats` | Materialized stats: topic_count, book_count, primary_book_count, sample_books, domain info |

### New API Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/v1/communities` | All communities (filterable by `?domain_id=N`) |
| `GET /api/v1/communities/{id}` | Community detail with constituent clusters and books |

---

## Quick Reference Commands

```bash
# === Full pipeline from scratch ===

# 1. Fine-grained clustering (clusters, γ=0.001)
LIBTRAILS_DB=v2 uv run libtrails cluster --tier cluster

# 2. Generate domains (super-clusters from cluster graph)
LIBTRAILS_DB=v2 uv run libtrails regenerate-domains --method leiden --cluster-knn-k 4 --resolution 0.0008
uv run python experiments/domain_labels_final.py
LIBTRAILS_DB=v2 uv run libtrails load-domains -f experiments/domain_labels_final.json

# 3. Coarse clustering (communities, γ=5.5e-5)
LIBTRAILS_DB=v2 uv run libtrails cluster --tier community

# 4. Refresh all stats (cluster_books → book_communities → community_stats → book_domains → domain_stats)
LIBTRAILS_DB=v2 uv run libtrails refresh-stats

# === Individual operations ===

# Sweep domain hyperparameters
LIBTRAILS_DB=v2 uv run libtrails regenerate-domains --method leiden --cluster-knn-k 4 --sweep --dry-run

# Sweep community resolution
LIBTRAILS_DB=v2 uv run libtrails cluster --sweep --sweep-range 0.000001 0.0005 --sweep-resolutions 25 --dry-run

# Check current stats
LIBTRAILS_DB=v2 uv run libtrails status
```
