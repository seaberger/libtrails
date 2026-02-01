# Topic Processing Pipeline Validation

This document tracks the deep validation of each step in the topic processing pipeline, ensuring data integrity and correctness at every stage.

## Pipeline Overview

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   1. PARSE   │───▶│   2. CHUNK   │───▶│  3. EXTRACT  │───▶│   4. STORE   │
│    EPUB      │    │   (~500w)    │    │   TOPICS     │    │  RAW TOPICS  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
        ┌──────────────────────────────────────────────────────────┘
        ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 5. NORMALIZE │───▶│  6. EMBED    │───▶│  7. DEDUPE   │───▶│ 8. CO-OCCUR  │
│   lowercase  │    │  BGE-small   │    │  cosine>0.85 │    │   pairs      │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
        ┌──────────────────────────────────────────────────────────┘
        ▼
┌──────────────┐    ┌──────────────┐
│  9. GRAPH    │───▶│ 10. CLUSTER  │
│   igraph     │    │   Leiden     │
└──────────────┘    └──────────────┘
```

---

## Validation Status

| Step | Component | Status | Date | Notes |
|------|-----------|--------|------|-------|
| 1 | EPUB Parsing | Not verified | - | - |
| 2 | Chunking (~500 words) | Not verified | - | - |
| 3 | Topic Extraction (LLM) | Not verified | - | - |
| 4 | Raw Topic Storage | **VERIFIED** | 2025-01-31 | 80k raw topics stored correctly |
| 5 | Normalization | **VERIFIED** | 2025-01-31 | Deep dive completed, bugs fixed |
| 6 | Embeddings | Not verified | - | Counts checked only |
| 7 | Deduplication | **VERIFIED** | 2025-01-31 | Algorithm bug fixed |
| 8 | Co-occurrence | Not verified | - | - |
| 9 | Graph Building | Not verified | - | - |
| 10 | Leiden Clustering | Not verified | - | - |

---

## Step 4: Raw Topic Storage

**Status**: VERIFIED

### What We Checked
- Total raw topics in `chunk_topics` table
- Unique topic counts per book
- Data integrity between chunks and topics

### Findings
- 80,788 unique raw topics stored across all indexed books
- Siddhartha: 223 unique raw topics from 415 entries (83 chunks)
- Dubliners: 355 unique raw topics from 695 entries (139 chunks)
- Topics correctly associated with chunk IDs

### Issues Found
None - raw topic storage is working correctly.

---

## Step 5: Normalization

**Status**: VERIFIED

### What We Checked
1. `normalize_topic()` function behavior
2. Case variation merging
3. `chunk_topic_links` accuracy (every raw topic → normalized topic → correct chunks)
4. `occurrence_count` accuracy

### Test Cases

**Siddhartha (223 raw topics)**
- All 223 raw topics traced to normalized form
- Verified all chunk links are correct
- Found and cleaned 13 development artifacts (incorrect links from testing)

**Dubliners (355 raw topics)**
- 355 raw → 349 normalized (6 case variations merged)
- 139/139 chunks have correct link counts
- 355/355 raw topics verified with correct links
- Full chain trace on 'Childhood' topic confirmed accuracy

### Case Variations Merged (Dubliners example)
```
'Childhood' + 'childhood' → 'childhood' (6 chunks)
'Appearance' + 'appearance' → 'appearance'
'Relationships' + 'relationships' → 'relationships'
'Rural Life' + 'rural life' → 'rural life'
'Shopping' + 'shopping' → 'shopping'
'Weather' + 'weather' → 'weather'
```

### Bugs Found & Fixed

#### Bug 1: Migration Double-Counting
- **Issue**: `migrate_raw_topics_to_normalized()` added to `occurrence_count` each run, causing counts to double
- **Impact**: Running migration twice doubled all counts (95% of topics affected)
- **Fix**: Made migration idempotent - recalculates counts from actual links at the end
- **Commit**: `dafa888` - "fix: make topic migration idempotent to prevent double-counting"

#### Bug 2: Development Artifacts in Siddhartha
- **Issue**: 13 incorrect chunk_topic_links existed (substring matches like 'brahmans' linked to 'brahman')
- **Root Cause**: Development/testing artifacts, not a code bug
- **Evidence**: 10 other books verified with zero incorrect links
- **Fix**: Manually cleaned 13 incorrect links from Siddhartha

---

## Step 7: Deduplication

**Status**: VERIFIED

### What We Checked
1. `find_duplicate_groups()` algorithm
2. Similarity scores between merged topics
3. Canonical topic selection (highest occurrence count)

### Bug Found & Fixed

#### Bug: Transitive Chain Merging
- **Issue**: Union-Find algorithm created chains where A~B and B~C grouped A,B,C even when A~C similarity was low
- **Example**: "identity" (0.67 sim) was wrongly grouped with "transformation" via chain:
  ```
  identity (0.85)→ identity and transformation (0.86)→ transformation
  ```
- **Impact**: 4 of 28 groups had incorrect merges
- **Fix**: Added validation that all group members must be directly similar to the canonical topic
- **Commit**: `0ec8fc0` - "fix: prevent transitive chains in topic deduplication"

### After Fix
- 28 duplicate groups for Siddhartha
- 0 problematic groups (all members directly similar to canonical)
- Examples of correct merges:
  - "brahman" ← "brahmans" (0.957 similarity)
  - "river symbolism" ← "river as symbol" (0.955 similarity)
  - "samanas" ← "samana" (0.939 similarity)

---

## Validation Methodology

### Deep Dive Process
For each step, we:
1. Check aggregate counts (quick sanity check)
2. Trace individual records through the transformation
3. Verify input→output mapping for specific examples
4. Check for edge cases and anomalies
5. Compare multiple books to distinguish code bugs from data artifacts

### Books Used for Validation
| Book | Author | Chunks | Raw Topics | Purpose |
|------|--------|--------|------------|---------|
| Siddhartha | Hermann Hesse | 83 | 223 | Primary test case |
| Dubliners | James Joyce | 139 | 355 | Secondary verification |
| 10 other books | Various | Various | Various | Spot checks |

---

## Remaining Validation Work

### Steps 1-3: EPUB → Chunks → Topics
- Need to verify text extraction quality
- Need to verify chunk boundaries (~500 words)
- Need to verify LLM topic extraction quality

### Step 6: Embeddings
- Currently only verified counts match
- Should verify embedding dimensions (384)
- Should verify vector normalization

### Steps 8-10: Co-occurrence → Graph → Clustering
- Not yet examined
- Should verify PMI calculations
- Should verify graph edge weights
- Should verify Leiden partition quality

---

## Related Issues & PRs

- **PR #5**: iPad library sync with full indexing pipeline
- **Issue #6**: Normalization strategy with ongoing indexing
- **Issue #4**: External author lookup for improved matching

---

*Last updated: 2025-01-31*
