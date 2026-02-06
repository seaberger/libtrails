# LibTrails Web Interface - Project Plan

## Overview

Build a modern, visually appealing web interface for browsing the indexed book library by themes, discovering related books through semantic search, and exploring connections across the collection.

**Inspiration**: [Pieter Maes' Trails](https://trails.pieterma.es) - clean design with 3D book covers, smooth transitions, theme clustering.

---

## Core Features

### 1. Theme Browser (Home Page)
Visual clusters of books grouped by topic/theme.

**Design**:
- Grid of theme "cards" or clusters
- Each cluster shows 3-5 overlapping book covers (stacked with slight offset)
- Theme label below/beside the cluster
- Hover effect reveals more covers or expands slightly
- Click theme → navigates to Theme Detail page

**Data source**: Leiden clusters from `topics` table + `chunk_topic_links`

### 2. Theme Detail Page
All books belonging to a selected theme.

**Design**:
- Theme title as header
- Vertical stack of book "cards" (not a grid - linear scroll)
- Each card shows: cover image, title, author, maybe top 3 sub-topics
- Smooth scroll, cards slightly overlap or have subtle shadow/depth
- Click book → navigates to Book Detail page

**Data source**: Books linked to the theme cluster via topic relationships

### 3. Book Detail Page
Single book view with related books discovery.

**Design**:
- Large cover image + title + author
- List of extracted topics (as tags/chips)
- "Related Books" section showing semantically similar titles
- Two similarity modes:
  - **Semantic**: Vector similarity via sqlite-vec embeddings
  - **Topic overlap**: Books sharing the most topics

**Data source**:
- Book metadata from `books` table
- Topics from `chunk_topics` / `topics`
- Similarity from `topic_vectors` (sqlite-vec)

### 4. Dynamic Search/Filter Bar
Real-time filtering as user types.

**Design**:
- Sticky search bar at top of page
- Placeholder: "Search by title, author, or topic..."
- As user types:
  - Filter visible books/themes in real-time
  - Show autocomplete suggestions (titles, authors, topics)
- Works on Theme Browser (filters which themes show)
- Works on a dedicated "All Books" view

**Implementation**: Client-side filtering with pre-loaded JSON index, or lightweight API endpoint

### 5. All Books View (Optional)
Flat list/grid of all books for browsing without theme grouping.

**Design**:
- Sortable by title, author, date added
- Filterable via search bar
- Toggle between grid (covers) and list (title/author rows) view

---

## Technology Evaluation

### Key Constraint: Hybrid Fusion Search

The responsive search bar requires **both keyword and semantic search** combined. This has architectural implications:

| Approach | Keyword Search | Semantic Search | Tradeoffs |
|----------|---------------|-----------------|-----------|
| **Client-only** | Fuse.js (fast) | Pre-computed similarities only | Can't search arbitrary queries semantically |
| **API backend** | SQLite FTS5 | sqlite-vec in real-time | Needs server, but full flexibility |
| **Hybrid** | Client for keywords | API for semantic | Best UX, moderate complexity |

**Recommendation**: We need a **lightweight API backend** for semantic search. This means the "purely static site" approach won't fully work for our needs.

### Search Performance Strategy

To avoid bogging down typing:
1. **Debounce input**: Wait 150-200ms after user stops typing before searching
2. **Show instant results for keywords**: Client-side filter runs immediately
3. **Async semantic results**: API call returns and merges results after ~200-500ms
4. **Optimistic UI**: Show "searching..." indicator, stream results as they arrive

### Framework Options Evaluated

#### Option A: **Next.js** (React-based)

| Pros | Cons |
|------|------|
| Full-featured from day one | Heavier than needed for static pages |
| Built-in API routes (search endpoint) | React learning curve |
| Huge ecosystem, lots of examples | More JavaScript complexity |
| Easy deployment (Vercel, or self-host) | Larger bundle size |
| Scales to full web app if needed | |

**Upgrade path**: Already a full framework. Can add authentication, user accounts, database, etc.

#### Option B: **Astro** + **FastAPI backend**

| Pros | Cons |
|------|------|
| Static pages are blazing fast | Two separate projects to maintain |
| Python backend matches libtrails codebase | More moving parts |
| Clean separation of concerns | Need to deploy two services |
| View transitions built-in | |

**Upgrade path**: Can swap Astro for anything else; FastAPI backend is independent and reusable.

#### Option C: **SvelteKit**

| Pros | Cons |
|------|------|
| Lighter than React/Next.js | Smaller ecosystem |
| Built-in API routes | Less familiar to most developers |
| Excellent performance | Fewer tutorials/examples |
| Can do static + server hybrid | |

**Upgrade path**: Similar to Next.js - full framework that can grow.

#### Option D: **Plain HTML/CSS/JS** + **FastAPI backend**

| Pros | Cons |
|------|------|
| Simplest mental model | More manual work for interactivity |
| No framework lock-in | Harder to maintain as it grows |
| Python does all the work | No component reusability |
| Easiest to understand | View transitions require manual setup |

**Upgrade path**: Can migrate to any framework later, but may require rewrite.

### Architecture Decision Matrix

| Requirement | Next.js | Astro+FastAPI | SvelteKit | Plain+FastAPI |
|-------------|---------|---------------|-----------|---------------|
| Hybrid search | ✅ API routes | ✅ FastAPI | ✅ API routes | ✅ FastAPI |
| Fast static pages | ⚠️ Good | ✅ Excellent | ✅ Excellent | ✅ Excellent |
| Smooth transitions | ✅ Built-in | ✅ Built-in | ✅ Built-in | ⚠️ Manual |
| Single codebase | ✅ Yes | ❌ Two projects | ✅ Yes | ❌ Two projects |
| Python integration | ⚠️ Separate API | ✅ Same language | ⚠️ Separate API | ✅ Same language |
| Learning curve | Medium | Medium | Medium | Low |
| Future scalability | ✅ High | ✅ High | ✅ High | ⚠️ Medium |

### Decision: Astro + FastAPI

**Confirmed approach** based on project requirements:

1. **Python consistency**: libtrails is Python, FastAPI keeps search logic in same language. Matches existing projects.
2. **Clean separation**: Static site can be hosted on CDN, API on Lightsail
3. **Flexibility**: Can swap frontend framework without touching search backend
4. **Performance**: Astro's static output is ideal for theme/book browsing pages

**Architecture**:
```
┌─────────────────────────────────────────────────────────┐
│                      Browser                            │
├─────────────────────────────────────────────────────────┤
│  Astro Static Pages          │  Interactive Components  │
│  - Theme Browser             │  - SearchBar (Svelte)    │
│  - Theme Detail              │  - Filters               │
│  - Book Detail               │                          │
│  - All Books                 │                          │
└──────────────┬───────────────┴──────────┬───────────────┘
               │ Static files             │ API calls
               ▼                          ▼
┌──────────────────────────┐  ┌───────────────────────────┐
│  CDN / Static Hosting    │  │  FastAPI Backend          │
│  (Lightsail or S3)       │  │  - /api/search            │
│                          │  │  - /api/similar/{book_id} │
└──────────────────────────┘  │  - sqlite-vec queries     │
                              │  - SQLite FTS5            │
                              └───────────────────────────┘
```

---

## Data Pipeline

### Export from SQLite to JSON

Before building the site, export the indexed data:

```
libtrails export --format json --output ./web/src/data/
```

**Exported files**:
```
data/
├── books.json          # All books with metadata
├── themes.json         # Leiden clusters with labels
├── book-themes.json    # Book → theme mappings
├── book-topics.json    # Book → topic list
├── topic-vectors.json  # For client-side similarity (optional)
└── search-index.json   # Pre-built search index for Fuse.js
```

### Data Schema

**books.json**:
```json
[
  {
    "id": 123,
    "title": "Siddhartha",
    "author": "Hermann Hesse",
    "cover": "/covers/123.jpg",
    "topics": ["spirituality", "self-discovery", "buddhism"],
    "themeIds": [5, 12],
    "wordCount": 45000
  }
]
```

**themes.json**:
```json
[
  {
    "id": 5,
    "label": "Eastern Philosophy",
    "bookCount": 23,
    "topBooks": [123, 456, 789],
    "subTopics": ["buddhism", "taoism", "meditation"]
  }
]
```

---

## Page Structure

```
/                       → Theme Browser (home)
/theme/[id]             → Theme Detail (books in theme)
/book/[id]              → Book Detail (single book + related)
/books                  → All Books (flat list/grid)
/search?q=...           → Search Results (optional dedicated page)
```

---

## Design Specifications

### Visual Style
- **Clean, minimal** - lots of whitespace
- **Book covers as primary visual** - high quality images
- **Muted color palette** - let covers provide color
- **Typography**: Modern sans-serif (Inter, System UI)
- **Depth**: Subtle shadows, overlapping elements for 3D feel

### Book Cover Display

**Theme cluster (overlapping)**:
```
    ┌─────┐
   ┌┼─────┼┐
  ┌┼┼─────┼┼┐
  │││ ... │││  ← 3-5 covers, offset by ~20px
  └┴┴─────┴┴┘
   Theme Name
```

**Theme detail (vertical stack)**:
```
┌──────────────────────────┐
│ ┌─────┐                  │
│ │cover│ Title            │
│ │     │ Author           │
│ └─────┘ topic, topic     │
├──────────────────────────┤
│ ┌─────┐                  │
│ │cover│ Title            │
│ │     │ Author           │
│ └─────┘ topic, topic     │
└──────────────────────────┘
```

### Responsive Breakpoints
- **Desktop**: 3-4 theme clusters per row
- **Tablet**: 2 clusters per row
- **Mobile**: 1 cluster, full-width cards

---

## Implementation Phases

### Phase 1: Data Export & Project Setup
- [x] ~~Add `libtrails export` CLI command~~ (Using direct API instead)
- [x] ~~Export books, themes, topics to JSON~~ (Direct queries)
- [x] ~~Extract/copy book covers from Calibre library~~ (Serving directly)
- [x] Initialize Astro project with Tailwind

### Phase 2: Theme Browser (Home) ✅
- [x] Create theme cluster component
- [x] Layout themes in responsive grid
- [x] Add hover effects and transitions
- [x] Link to theme detail pages

### Phase 3: Theme Detail Page ✅
- [x] Create book card component
- [x] Vertical scrolling layout
- [x] Display book metadata and topics
- [x] Link to book detail pages

### Phase 4: Book Detail Page ✅
- [x] Large cover + metadata display
- [x] Topic tags/chips
- [x] "Related Books" section
- [x] Implement similarity lookup (topic overlap)

### Phase 5: Search & Filter (TODO)
- [ ] Add search bar component (React)
- [ ] Debounced API search
- [ ] Real-time filtering on Theme Browser
- [ ] Autocomplete suggestions

### Phase 6: Polish & Deploy (TODO)
- [ ] View transitions between pages
- [ ] Loading states and skeleton screens
- [ ] SEO metadata
- [ ] Deploy to AWS Lightsail
- [ ] Custom domain setup

---

## Book Cover Strategy

Calibre stores covers as `cover.jpg` in each book's folder.

**Options**:

1. **Copy covers to static folder** during export
   - Pro: Simple, works offline
   - Con: Duplicates data, larger repo

2. **Generate optimized thumbnails**
   - Use Sharp/ImageMagick to create 200px and 400px versions
   - Pro: Faster loading, responsive images
   - Con: Build step required

3. **Reference Calibre library directly** (local dev only)
   - Pro: No duplication
   - Con: Won't work in production

**Recommendation**: Option 2 - generate optimized thumbnails during export, store in `web/public/covers/`.

---

## Open Questions

1. **Theme labeling**: Use LLM to generate human-readable cluster names, or show top topics?

2. **Similarity threshold**: How many "related books" to show? Top 10? Configurable?

3. **Cover fallbacks**: What to show for books without covers? Generate placeholder with title/author?

4. **Mobile navigation**: Bottom tab bar or hamburger menu?

5. **Dark mode**: Support system preference?

---

## Future Enhancements

- **Reading lists**: Save books to custom collections
- **Trails view**: Connect excerpts across books (like Pieter's trails)
- **Import highlights**: Pull annotations from MapleRead
- **Social features**: Share themes/trails publicly
- **API endpoint**: Enable search from other tools (Alfred, Raycast)

---

## File Structure (Proposed)

```
calibre_lib/
├── src/libtrails/          # Existing CLI
│   └── export.py           # NEW: JSON export for web
├── web/                    # NEW: Astro project
│   ├── astro.config.mjs
│   ├── package.json
│   ├── src/
│   │   ├── layouts/
│   │   │   └── BaseLayout.astro
│   │   ├── pages/
│   │   │   ├── index.astro         # Theme Browser
│   │   │   ├── theme/[id].astro    # Theme Detail
│   │   │   ├── book/[id].astro     # Book Detail
│   │   │   └── books.astro         # All Books
│   │   ├── components/
│   │   │   ├── ThemeCluster.astro
│   │   │   ├── BookCard.astro
│   │   │   ├── SearchBar.svelte    # Interactive
│   │   │   └── RelatedBooks.astro
│   │   └── data/                   # Exported JSON
│   │       ├── books.json
│   │       ├── themes.json
│   │       └── ...
│   └── public/
│       └── covers/                 # Book cover images
└── docs/
    └── web-interface-plan.md       # This file
```

---

---

## Library Update Workflow

### Goal

A simple, semi-automated routine to add new books from iPad without remembering multiple CLI commands.

### User Experience

```bash
# 1. Open MapleRead server on iPad
# 2. Run single command:

libtrails sync --ipad http://192.168.1.124:8082

# That's it. Everything else is automated.
```

### What `libtrails sync` Does

```
┌─────────────────────────────────────────────────────────────────┐
│                     libtrails sync                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. SCRAPE iPad Library                                         │
│     └─→ Fetch book list from MapleRead HTTP server              │
│     └─→ Compare to existing books in local DB                   │
│     └─→ Identify NEW books (not yet indexed)                    │
│                                                                 │
│  2. MATCH to Calibre                                            │
│     └─→ Find each new book in Calibre library                   │
│     └─→ Pull metadata (author, tags, description, cover)        │
│     └─→ Locate EPUB/PDF file path                               │
│                                                                 │
│  3. INDEX new books                                             │
│     └─→ Extract text from EPUB/PDF                              │
│     └─→ Chunk into ~500 word segments                           │
│     └─→ Extract topics via Ollama (gemma3:4b)                   │
│     └─→ Generate embeddings (sentence-transformers)             │
│                                                                 │
│  4. UPDATE search indexes                                       │
│     └─→ Add to sqlite-vec for semantic search                   │
│     └─→ Update FTS5 full-text index                             │
│     └─→ Re-run topic deduplication (optional)                   │
│     └─→ Re-cluster if significant new content (optional)        │
│                                                                 │
│  5. REPORT summary                                              │
│     └─→ "Added 5 new books, indexed 3, skipped 2 (no EPUB)"     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Sync Options

```bash
# Basic sync - index new books
libtrails sync --ipad http://192.168.1.124:8082

# Sync with re-clustering (after adding many books)
libtrails sync --ipad http://... --recluster

# Dry run - show what would be added without doing it
libtrails sync --ipad http://... --dry-run

# Skip indexing (just update book list, index later)
libtrails sync --ipad http://... --skip-index
```

### Deployment to Lightsail

Processing happens on MacBook (needs Ollama/GPU), then data is pushed to server.

**Option A: Database + Assets Sync**

```bash
# After sync completes, push to production
libtrails deploy --target lightsail

# This would:
# 1. Export SQLite to production format (or copy directly)
# 2. Rsync book covers to server
# 3. Regenerate static Astro pages
# 4. Upload to Lightsail
# 5. Restart FastAPI service
```

**Option B: Export Bundle**

```bash
# Create a deployment bundle
libtrails export --bundle ./deploy-2025-01-31/

# Bundle contains:
# - database.db (SQLite with all data)
# - covers/ (optimized book cover images)
# - static/ (pre-built Astro pages)
# - config.json (API configuration)

# Then manually upload or use separate deploy script
scp -r ./deploy-2025-01-31/ lightsail:/app/
```

**Option C: Git-based Deploy**

```bash
# Commit data changes to repo
libtrails export --to-repo

# Push triggers CI/CD pipeline on Lightsail
git push origin main
# → GitHub Action / Lightsail deploy hook rebuilds site
```

### Recommended: Option A (Direct Deploy)

For your use case (personal site, few books at a time), direct deploy is simplest:

```bash
# Full workflow after adding books to iPad:
libtrails sync --ipad http://192.168.1.124:8082
libtrails deploy --target lightsail
```

Two commands. Could even combine into one:

```bash
libtrails sync --ipad http://... --deploy
```

### Configuration File

Store defaults so you don't have to remember the IP:

**~/.libtrails/config.yaml**
```yaml
ipad:
  default_url: http://192.168.1.124:8082

deploy:
  target: lightsail
  host: your-lightsail-ip
  user: ubuntu
  app_path: /home/ubuntu/libtrails

indexing:
  model: gemma3:4b
  max_words: 300000
  min_battery: 15
```

Then just:
```bash
libtrails sync   # Uses saved iPad URL
libtrails deploy # Uses saved Lightsail config
```

### Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────────────┐
│   iPad      │     │  MacBook    │     │  Lightsail Server       │
│  MapleRead  │────▶│  Pro        │────▶│                         │
│  (source)   │http │  (process)  │rsync│  FastAPI + SQLite       │
└─────────────┘     │             │     │  Astro static pages     │
                    │  - Ollama   │     │                         │
┌─────────────┐     │  - sqlite   │     └─────────────────────────┘
│  Calibre    │────▶│  - covers   │                │
│  Library    │file │             │                │
│  (EPUBs)    │     └─────────────┘                ▼
└─────────────┘                            ┌───────────────┐
                                           │   Browser     │
                                           │   (users)     │
                                           └───────────────┘
```

### Implementation Tasks

1. [ ] Refactor scraping into `libtrails sync` command
2. [ ] Add incremental indexing (only new books)
3. [ ] Add `--dry-run` mode
4. [ ] Create config file support (~/.libtrails/config.yaml)
5. [ ] Implement `libtrails deploy` command
6. [ ] Set up Lightsail server with FastAPI
7. [ ] Create rsync/scp deployment script

---

## Implementation Status (Feb 3, 2025)

### Completed ✅

**Backend (FastAPI)**:
- `src/libtrails/api/` - Full API implementation
- Endpoints: `/api/v1/themes`, `/api/v1/books`, `/api/v1/search`, `/api/v1/covers`
- Pydantic schemas for all responses
- CLI command: `uv run libtrails serve`

**Frontend (Astro + Tailwind + React)**:
- `web/` - Complete Astro project
- Theme Browser home page with overlapping book cover cards
- Theme Detail page (`/themes/[id]`) with all books in cluster
- Book Detail page (`/books/[id]`) with topics and related books
- All Books page (`/books`) with pagination
- Components: ThemeCard, BookCard, BookCover, TopicChip, RelatedBooks
- Clean, minimal design with warm color palette

**Data**:
- 925 books indexed
- 108,668 topics with embeddings
- ~340 Leiden clusters
- Story of Civilization: 4.9M words, 1,637 chunks, 3,708 topics

### Key Implementation Decisions

1. **Direct API vs JSON Export**: Went with direct SQLite queries via FastAPI instead of pre-exporting JSON. More efficient for 100k+ topics.

2. **Cover Strategy**: Serve covers directly from Calibre library path (`/api/v1/covers/{calibre_id}`) instead of copying. Simpler, no duplication.

3. **SSR for Dynamic Routes**: Theme and book detail pages use `prerender = false` for server-side rendering to avoid generating thousands of static pages.

4. **React over Svelte**: Used React for interactive islands (search bar to come) since it's more familiar and Astro supports both.

### Remaining Work

**Phase 5: Search & Filter** (Not Started):
- [ ] Add SearchBar.tsx React component
- [ ] Debounced search with API calls
- [ ] Real-time filtering on Theme Browser
- [ ] Autocomplete suggestions

**Phase 6: Polish & Deploy** (Not Started):
- [ ] Astro view transitions between pages
- [ ] Loading states and skeleton screens
- [ ] SEO metadata
- [ ] Deploy to AWS Lightsail
- [ ] Custom domain setup

**Additional Enhancements**:
- [ ] LLM-generated cluster labels (instead of "topic1 / topic2 / topic3")
- [ ] Optimize theme cover stacking animation
- [ ] Mobile responsive testing
- [ ] Dark mode support

### Branch

All code committed to: `feature/web-interface` (pushed to GitHub)

### Running Locally

```bash
# Terminal 1: Start API server
uv run libtrails serve

# Terminal 2: Start frontend
cd web && npm run dev

# Open http://localhost:4321
```

---

## Original Next Steps (Archived)

1. ~~Wait for indexing to complete~~ ✅
2. ~~Run Leiden clustering~~ ✅
3. ~~Implement `libtrails export`~ Skipped (using direct API)
4. ~~Initialize Astro project~~ ✅
5. ~~Build Phase 2 (Theme Browser)~~ ✅

---

## Phase 7: Domain Hierarchy (Feb 2025)

### Problem

Browsing 845 Leiden clusters directly is overwhelming. Users need a higher-level entry point.

### Solution: Two-Level Navigation

```
┌─────────────────────────────────────────────────────────────┐
│  LibTrails    [Themes]  [Clusters]  [Books]  [Search]       │
└─────────────────────────────────────────────────────────────┘
```

| Tab | Content | Status |
|-----|---------|--------|
| **Themes** | ~20 broad domains (super-clusters) | NEW |
| **Clusters** | ~845 Leiden clusters (current themes page) | RENAME |
| Books | Browse all indexed books | Existing |
| Search | Semantic search across topics/books | Existing |

### Themes Tab: Two-Panel Domain Browser

```
┌────────────────────────┬──────────────────────────────────────────┐
│  DOMAINS               │                                          │
│  ──────────────────    │  [Selected domain detail]                │
│                        │                                          │
│  ○ Conflict & Warfare  │  🔬 Artificial Intelligence & Math       │
│  ○ Technology          │  ────────────────────────────────────    │
│  ● AI & Math      ←    │  68 clusters · 8,750 topics              │
│  ○ Space & Earth       │                                          │
│  ○ Food & Drink        │  Featured Clusters:                      │
│  ○ Financial Markets   │  ┌─────────┐ ┌─────────┐ ┌─────────┐    │
│  ○ Ancient Cultures    │  │ Machine │ │ Neural  │ │  Stats  │    │
│  ○ Spirituality        │  │Learning │ │Networks │ │  & Prob │    │
│  ○ Human Condition     │  │ 📚 142  │ │ 📚 98   │ │ 📚 87   │    │
│  ○ Entertainment       │  └─────────┘ └─────────┘ └─────────┘    │
│  ○ Biology & Health    │                                          │
│  ○ ...                 │  [View all 68 clusters →]                │
│                        │                                          │
│  ────────────────      │  Sample Books:                           │
│  845 clusters total    │  📖 Deep Learning (Goodfellow)           │
│  108,668 topics        │  📖 Pattern Recognition (Bishop)         │
│                        │  📖 The Master Algorithm                 │
│                        │                                          │
└────────────────────────┴──────────────────────────────────────────┘
```

### Super-Cluster Generation (Experiment Results)

**Method**: K-means on Leiden cluster centroid embeddings

```python
# 1. Compute centroid for each of 845 Leiden clusters
centroids[cluster_id] = mean(topic_embeddings_in_cluster)

# 2. K-means clustering on centroids → 20 super-clusters
kmeans = KMeans(n_clusters=20)
super_assignments = kmeans.fit_predict(centroid_matrix)

# 3. LLM generates label for each super-cluster
```

**Results** (Feb 5, 2025):

| Domain | Label | Clusters | Topics |
|--------|-------|----------|--------|
| 0 | Conflict and Warfare | 31 | 3,779 |
| 1 | Technology & Systems | 66 | 7,657 |
| 2 | Chess & Strategy | 97 | 8,460 |
| 3 | Social & Ethical Issues | 42 | 5,328 |
| 4 | International Relations & Law | 38 | 5,538 |
| 5 | Energy & Industry | 34 | 3,856 |
| 6 | Business & Education | 34 | 3,715 |
| 7 | Food & Drink | 28 | 4,663 |
| 8 | Transportation & Rescue | 26 | 3,278 |
| 9 | Biology & Health | 41 | 5,448 |
| 10 | Creative Writing & Mind | 53 | 8,081 |
| 11 | Artificial Intelligence & Math | 68 | 8,750 |
| 12 | Ancient Civilizations & Cultures | 36 | 5,076 |
| 13 | Human Condition | 39 | 5,960 |
| 14 | Space Exploration & Earth | 42 | 5,733 |
| 15 | Intelligence & Deception | 23 | 3,719 |
| 16 | Financial Markets & Assets | 38 | 5,600 |
| 17 | Entertainment & Culture | 43 | 6,043 |
| 18 | Spirituality & Belief | 26 | 3,733 |
| 19 | 18th Century European History | 40 | 4,251 |

**Quality Issues**:
- Some domains are coherent (Transportation & Rescue, Financial Markets)
- Some are junk drawers (Chess & Strategy contains random short-word topics)
- Short topic labels (≤3 chars) have weak embedding signal

**Data**: `experiments/super_clusters.json`

### Implementation Tasks

#### Phase 7a: Data Infrastructure
- [ ] Fix super-cluster quality (filter short topics before centroid calculation)
- [ ] Re-run k-means with improved centroids
- [ ] Create `domains` table in database
- [ ] Store domain assignments for each Leiden cluster
- [ ] Add API endpoints: `GET /domains`, `GET /domains/{id}`

#### Phase 7b: Themes Tab (New)
- [ ] Create `/themes` page with two-panel layout
- [ ] Left panel: Domain list component (selectable)
- [ ] Right panel: Domain detail with featured clusters
- [ ] Client-side panel switching (no page reload)
- [ ] "View all clusters" link to filtered Clusters tab

#### Phase 7c: Clusters Tab (Rename)
- [ ] Rename current "Themes" to "Clusters" in navigation
- [ ] Update routes: `/themes/*` → `/clusters/*`
- [ ] Add domain filter dropdown
- [ ] Add domain badge to cluster cards

### API Design

**GET /api/v1/domains**
```json
[
  {
    "domain_id": 11,
    "label": "Artificial Intelligence & Math",
    "cluster_count": 68,
    "topic_count": 8750,
    "book_count": 342,
    "featured_clusters": [
      {"cluster_id": 5, "label": "machine learning", "book_count": 142},
      {"cluster_id": 57, "label": "neural networks", "book_count": 98}
    ],
    "sample_books": [
      {"id": 123, "title": "Deep Learning", "calibre_id": 456}
    ]
  }
]
```

**GET /api/v1/domains/{domain_id}**
```json
{
  "domain_id": 11,
  "label": "Artificial Intelligence & Math",
  "clusters": [
    {"cluster_id": 5, "label": "machine learning", "size": 357, "book_count": 142},
    {"cluster_id": 57, "label": "neural networks", "size": 245, "book_count": 98}
    // ... all 68 clusters
  ]
}
```

### URL Structure (Updated)

```
/                       → Redirect to /themes
/themes                 → Domain browser (two-panel)
/themes/{domain_id}     → Same page, domain pre-selected
/clusters               → All clusters grid (renamed from /themes)
/clusters?domain={id}   → Filtered by domain
/clusters/{cluster_id}  → Cluster detail (existing theme detail)
/books                  → All books
/books/{book_id}        → Book detail
```

### Open Questions

1. **Super-cluster quality**: How to handle junk drawer domains?
   - Filter Leiden clusters with mostly short topics
   - Manual curation of the 20 domains
   - Use hierarchical Leiden instead of k-means

2. **Number of domains**: Is 20 the right number?
   - Could try 15 (broader) or 25 (more specific)
   - Could make it dynamic based on cluster coherence

3. **Domain naming**:
   - LLM-generated (current) vs manual curation
   - Could add icons for visual distinction
