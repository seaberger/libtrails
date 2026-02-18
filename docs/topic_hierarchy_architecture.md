  # Topic Hierarchy Architecture

How LibTrails organizes 837K extracted topics into browsable layers for discovery.

---

## The Problem: Binary Membership Produces No Differentiation

After generating 29 V2 domains (super-clusters from Leiden clustering), we discovered that **binary "any topic link" membership means every book appears in every domain**. A typical book has ~1,000 topic mentions across hundreds of Leiden clusters. With 29 domains grouping those clusters, every book touches every domain.

**The data** (Feb 2026, 937 books, 6,575 Leiden clusters, 29 domains):
- Average books per domain (binary): **859 of 937** (92%)
- Even the most specific domain (Russian Literature, 32 clusters): 426 books
- "On Cooking" and "Mistborn" both showed up as Russian Literature books

---

## Solution Phase 1: Weighted Domain Membership (Implemented)

**Branch**: `feat/weighted-domain-membership`

Instead of binary membership, compute each book's **concentration** per domain — what fraction of its topics land in that domain.

### New table: `book_domains`

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

**Scoring**: Uses `book_cluster_relevance()` (BM25 saturation + PPMI) to score each book-domain pair. Each book's highest-scoring domain gets `is_primary = 1`.

### Results: Before vs After

| Method | Avg books/domain | Min | Max |
|--------|-----------------|-----|-----|
| Binary (old) | 859 | 426 | 936 |
| Weighted ≥1% | 656 | 57 | 911 |
| **Primary only** | **32** | **1** | **147** |

### Threshold Sweep

We swept concentration thresholds to find the sweet spot between "everything" and "too restrictive":

| Threshold | Avg books/domain | Avg domains/book | Feel |
|-----------|-----------------|-----------------|------|
| Primary | 32 | 1.0 | "Books that ARE this theme" |
| ≥3% | 320 | 9.9 | Still noisy — 1/3 of domains |
| ≥4% | 225 | 7.0 | Getting better |
| **≥5%** | **168** | **5.2** | **Meaningful secondary connections** |
| ≥7% | 99 | 3.1 | Tight |
| ≥10% | 55 | 1.7 | Almost same as primary |

**Recommendation**: Primary is the default display; ≥5% for "show more / related" expansion. At 5%, each book appears in ~5 domains — enough to show cross-domain connections without noise.

### Spot-Check: Russian Literature Domain

| Book | Concentration | Primary? |
|------|--------------|----------|
| Reading Chekhov (Janet Malcolm) | 15.9% | Yes |
| Life and Fate (Vasily Grossman) | 10.7% | Yes |
| Complete Works of Anton Chekhov | 6.5% | Yes |
| Doctor Zhivago (Boris Pasternak) | 3.9% | Yes |
| Tinker Tailor Soldier Spy | 4.7% | No |

Previously this domain showed 426 books. Now: 7 primary, 57 at ≥1%, 7 at ≥5%.

### Files Changed

| File | Change |
|------|--------|
| `database.py` | `book_domains` table + `primary_book_count` on `domain_stats` |
| `stats.py` | `refresh_book_domains()` + rewritten `refresh_domain_stats()` |
| `api/schemas.py` | `DomainBook` schema with `concentration`, `is_primary` |
| `api/routers/domains.py` | Queries from `book_domains` |
| `web/src/lib/types.ts` | `DomainBook` type, `primary_book_count` |
| `web/src/pages/themes/index.astro` | Shows primary + related counts |

---

## The Cluster Granularity Problem

With weighted domain membership solved, we examined the next layer down: **are the 6,575 Leiden clusters useful as user-facing units?**

### Answer: No.

**Cluster-level concentration is almost nonexistent:**
- 6,306 of 6,575 clusters (96%) have **zero books** with ≥5% concentration
- Only 269 clusters have 1-3 books at 5%
- Only 6 clusters have 4-10 books

A typical book's ~1,000 topics are spread across hundreds of clusters. No single cluster captures a meaningful fraction of any book's content. Showing users 6,575 clusters is like showing them individual pages of an index instead of chapter headings.

### Cluster size distribution

| Size bucket | Clusters | Avg books (binary) |
|------------|----------|-------------------|
| 1-5 topics | 13 | 1 |
| 6-20 topics | 42 | 9 |
| 21-50 topics | 534 | 25 |
| 51-100 topics | 2,451 | 50 |
| 101-200 topics | 2,648 | 86 |
| 200+ topics | 887 | 148 |

Most clusters are 51-200 topics — substantial individually, but too granular for users to browse 6,575 of them.

### But clusters ARE valuable internally

The fine-grained Leiden clusters serve a critical role in **search context**. When a user searches for "Russian revolution," the search system finds matching topics and groups them by `cluster_id`. The cluster provides the semantic neighborhood: "these 80 related topics all cluster together — Russian revolution timeline, Bolshevik movement, Tsarist collapse, etc."

This requires fine granularity. A coarser grouping would produce neighborhoods of thousands of topics labeled "History & Conflict" — useless for specific exploration.

---

## Solution Phase 2: Three-Tier Topic Hierarchy (Planned)

### Architecture

```
User-facing                         Internal
──────────                          ────────
Themes tab  →  29 domains           domain_stats, book_domains
Topics tab  →  ~200 communities     community_stats, book_communities (NEW)
Search      →  6,575 clusters       cluster_stats, cluster_books (existing)
```

Each topic gets two assignments:
- `cluster_id` (6,575, fine) — powers search context, topic neighborhoods
- `community_id` (~200, coarse) — powers Topics tab, Universe 3D view

### Why Three Tiers

| Layer | Count | Purpose | User sees? |
|-------|-------|---------|-----------|
| Domains | 29 | Broad theme browsing | Themes tab |
| Communities | ~200 | Specific topic exploration + Universe | Topics tab, Universe |
| Clusters | 6,575 | Search neighborhoods | Search results only |

### Table structure

```
domains (29)
  └── community_domains → communities (~200)
       └── cluster_communities → clusters (6,575)
            └── topics (837K)
```

New tables:
- `communities` — id, label, domain_id, cluster_count
- `cluster_communities` — cluster_id → community_id
- `book_communities` — same pattern as book_domains (concentration, relevance_score, is_primary)
- `community_stats` — same pattern as domain_stats

### How to Create Communities

**Recommended: Coarser Leiden on the topic KNN graph.**

The topic KNN graph is already cached to disk (PR #47). We run Leiden at a higher CPM resolution (γ) to produce ~200 communities directly. This preserves both semantic similarity and co-occurrence signals — the same principled approach as the fine-grained clusters, just at a different scale.

```
Topics → KNN graph → Leiden (γ_fine ~ 0.001)  → 6,575 clusters  (search)
                   → Leiden (γ_coarse ~ ???)   → ~200 communities (browsing)
```

Each community maps to a domain by majority vote: if most of community X's constituent clusters belong to domain Y, then community X is in domain Y.

### Experiment: Leiden vs K-means for ~200 Communities

We tested both approaches on the full V2 dataset (837K topics, 6,575 Leiden clusters, topic KNN graph with 6.4M edges).

#### K-means on Cluster Centroids

`regenerate-domains --method kmeans --n-domains 200 --dry-run` — completed in 7 seconds.

**Result**: 200 groups, size range 6–140 clusters per group. Auto-labels are **incoherent**: concatenations of unrelated topic strings from different books (e.g., "briony's concern for barrick / doctor manette's in...", "anguilla anguilla life cycle / randy's physical di..."). K-means groups clusters by embedding proximity, but averaged centroids from semantically diverse clusters produce meaningless neighborhoods. These labels cannot be shown to users.

#### Leiden CPM Resolution Sweep on Topic KNN Graph

`cluster --sweep --sweep-range 0.000001 0.0005 --sweep-resolutions 25 --dry-run` — completed in 2,833s (47 min).

The resolution-to-cluster-count relationship is **non-monotonic**, revealing phase transitions in the CPM landscape:

| Resolution (γ) | Clusters | Note |
|---------------|----------|------|
| 1.00e-06 | 44 | |
| 1.30e-06 | 44 | |
| 1.68e-06 | 88 | |
| 2.17e-06 | 108 | |
| **2.82e-06** | **178** | **~200 range (Phase 1)** |
| **3.65e-06** | **229** | **~200 range (Phase 1)** |
| 4.73e-06 | 111 | Phase transition ↓ |
| 6.13e-06 | 99 | |
| 7.94e-06 | 74 | |
| 1.03e-05 | 51 | Local minimum |
| 1.33e-05 | 65 | Recovery begins |
| 1.73e-05 | 79 | |
| 2.24e-05 | 102 | |
| 2.90e-05 | 137 | |
| **3.75e-05** | **175** | **~200 range (Phase 2)** |
| **4.86e-05** | **230** | **~200 range (Phase 2)** |
| 6.30e-05 | 307 | |
| 8.16e-05 | 396 | |
| 1.06e-04 | 511 | |
| 1.37e-04 | 661 | |
| 1.77e-04 | 828 | |
| 2.30e-04 | 1,057 | |
| 2.98e-04 | 1,293 | |
| 3.86e-04 | 1,598 | |
| 5.00e-04 | 1,974 | Current fine clusters at γ=0.001 → 6,575 |

**Key finding**: ~200 communities appears at two distinct resolution scales, 13x apart:
- **Phase 1** (γ ≈ 3e-6): Few large communities. Ultra-low resolution, possibly too coarse — each community may span very different semantic regions.
- **Phase 2** (γ ≈ 4e-5): More balanced communities. Higher resolution produces tighter, more semantically coherent groups.

**Recommendation**: Use **Phase 2 at γ ≈ 4e-5** (175–230 communities). The higher resolution produces smaller, tighter communities that will produce more meaningful labels and better user browsing experience. The Phase 1 resolution likely produces communities that are internally heterogeneous (similar to the domain-level problem we already solved).

#### Why Leiden Wins

| Criterion | K-means | Leiden (Phase 2) |
|-----------|---------|-------------------|
| Auto-labels | Incoherent (concatenated fragments) | Expected: coherent (graph-connected topics) |
| Respects co-occurrence | No — only embedding distance | Yes — graph topology + co-occurrence |
| Deterministic | Yes | No — but CPM plateau analysis finds stable regions |
| Speed | 7s | ~2 min per resolution (47 min for 25-point sweep) |
| Semantic coherence | Low — averaged centroids → mushy boundaries | High — graph communities share real connections |

### Resolution Selection

For production, we'll use **γ ≈ 4.86e-5** (Phase 2, ~230 communities). This is the right neighborhood because:
1. It produces ~200–230 communities — right for the Topics tab
2. It's in the stable ascending region of the curve (monotonically increasing from γ=1e-5 onward)
3. Higher resolution means tighter communities with better label coherence
4. Fine-tuning ±10% around this value will let us dial in the exact count

---

## Universe 3D View Considerations

### Current state (6,575 points)
The Universe shows every Leiden cluster as a 3D point, positioned by UMAP projection of cluster centroids, colored by domain. It's visually dense but individual clusters are hard to explore.

### Proposed (~200 community points)
Each community becomes a 3D node. Open questions:

1. **Will ~200 points show meaningful spatial structure?** UMAP works better with more points for local structure, but each community represents a larger, more distinct semantic region. Domain-colored regions should still be visible. We'll know when we test.

2. **Node sizing**: Communities vary in size (topic count). Rendering as variable-size spheres (not uniform points) would convey this information visually.

3. **Fallback: two-level zoom**: If ~200 points feels too sparse, the Universe could show domains as large regions → click to zoom into that domain's communities → optionally zoom further into clusters. This provides the "explore at your own depth" experience.

4. **Could keep fine-grained view as option**: The 6,575-cluster Universe could remain as an advanced/debug view alongside the community Universe.

### Decision: Test empirically
Build the ~200 community data, run UMAP, render it, and evaluate. If spatial structure is poor, implement the two-level zoom. This is a visual design question best answered by looking at it.

---

## Navigation Model

### Current (flat)
```
Themes tab: 29 domains → domain detail (all books, flat list)
Clusters tab: 6,575 clusters → cluster detail (topics + books)
Universe: 6,575 3D points → click → cluster sidebar
Books tab: book list → book detail (topics, cluster links)
```

### Proposed (hierarchical, theme-first)
```
Themes tab: 29 domains → primary books + "show more" at ≥5%
Topics tab: ~200 communities → community detail (books by concentration)
             ↳ filterable by parent domain
Universe: ~200 community nodes → click → community sidebar with books
Books tab: book detail → concentration profile across communities + domains
             ↳ "this book is 24% Science Fiction, 15% Family & Conflict, ..."
```

### Key UX principle
**Primary membership is the default, concentration expansion is opt-in.** Users see tight, accurate lists by default. "Show more" reveals cross-domain connections — the "trails" that connect books across themes.

---

## Implementation Sequence

1. ~~**Sweep γ** for ~200 communities on the topic KNN graph~~ ✅ Done — γ ≈ 4.86e-5 → ~230 communities
2. ~~**Create communities table** and `cluster_communities` mapping~~ ✅ Done — `communities` table + `topics.community_id` column + `community_books`, `community_stats`, `book_communities` tables
3. ~~**Build `book_communities`** (same pattern as `book_domains`)~~ ✅ Done — `refresh_book_communities()` in `stats.py` with concentration/relevance scoring
4. ~~**Generate community labels** (auto-labels from top topics per community)~~ ✅ Done — auto-labels concatenated from top 3 topics; LLM labels deferred to future PR
5. ~~**Update Topics tab** (rename from Clusters, show communities)~~ ✅ Done — Topics tab fetches communities via API, displays with book covers and stats
6. **Update Universe view** (render communities instead of clusters)
7. **Test Universe spatial structure** — decide if two-level zoom is needed
8. **Update book detail page** — show community/domain concentration profile

---

## Appendix: What Uses Fine-Grained Clusters (Must Preserve)

| Usage | Purpose | Keep? |
|-------|---------|-------|
| `topics.cluster_id` | Topic's cluster assignment | Yes — search context |
| `cluster_books` | Bridge table for book-cluster lookups | Yes — feeds book_domains |
| `cluster_stats` | Per-cluster materialized stats | Yes — search results |
| Hybrid search cluster results | Groups matching topics by cluster | Yes — core search feature |
| Theme search (`/themes/search`) | Finds clusters matching query | Evolve — search communities |
| Theme detail (`/themes/{id}`) | Shows cluster topics + books | Evolve — becomes community detail |
| Clusters page (`/clusters/`) | Displays cluster cards | Replace — becomes Topics tab |
| Universe (`GalaxyView.tsx`) | 3D cluster visualization | Replace — community nodes |
| Domain "top clusters" pills | Shows top clusters in domain | Replace — top communities |
| Book detail topic links | Links topics to clusters | Evolve — link to communities |
| `cluster_domains` | Maps clusters to domains | Keep — now feeds cluster_communities |
| `cluster_labels` | LLM labels per cluster | Keep — internal reference |
| `topic_cluster_memberships` | Hub multi-cluster membership | Keep — internal |
