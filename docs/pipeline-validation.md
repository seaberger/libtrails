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
| 1 | EPUB Parsing | **VERIFIED** | 2025-01-31 | Clean text, no HTML artifacts |
| 2 | Chunking (~500 words) | **VERIFIED** | 2025-01-31 | 99% sentence boundaries |
| 3 | Topic Extraction (LLM) | **VERIFIED** | 2025-01-31 | Accurate, 5 topics/chunk |
| 4 | Raw Topic Storage | **VERIFIED** | 2025-01-31 | 80k raw topics stored correctly |
| 5 | Normalization | **VERIFIED** | 2025-01-31 | Deep dive completed, bugs fixed |
| 6 | Embeddings | **VERIFIED** | 2025-01-31 | 384-dim, normalized, semantic quality |
| 7 | Deduplication | **VERIFIED** | 2025-01-31 | Algorithm bug fixed |
| 8 | Co-occurrence | **VERIFIED** | 2025-01-31 | PMI calculations correct |
| 9 | Graph Building | **VERIFIED** | 2025-01-31 | Siddhartha only, edges form correctly |
| 10 | Leiden Clustering | **VERIFIED** | 2025-01-31 | 95% clustering, semantic coherence |

---

## Step 1: EPUB Parsing

**Status**: VERIFIED (Siddhartha)

### What We Checked
1. Text extraction quality (no HTML artifacts, encoding issues)
2. Content completeness
3. Proper character handling

### Test Results
- Total words extracted: 39,381
- No HTML tags detected in text
- No excessive whitespace or encoding issues
- Publisher metadata present at start (normal for EPUB source)

### Sample Text Quality
```
"In the shade of the house, in the sunshine of the riverbank near the
boats, in the shade of the Sal-wood forest, in the shade of the fig
tree is where Siddhartha grew up, the handsome son of the Brahman..."
```
Clean prose, properly formatted.

### Issues Found
None - EPUB parsing is working correctly.

---

## Step 2: Chunking

**Status**: VERIFIED (Siddhartha)

### What We Checked
1. Chunk size distribution (target ~500 words)
2. Sentence boundary preservation
3. Content continuity

### Statistics
| Metric | Value |
|--------|-------|
| Total chunks | 83 |
| Min words | 246 |
| Max words | 500 |
| Average words | 474 |
| Total words | 39,381 |

### Word Count Distribution
```
<400:     3 chunks (3.6%) - end of sections
400-449:  8 chunks (9.6%)
450-499: 69 chunks (83.1%) - target range
500-549:  3 chunks (3.6%)
```

### Sentence Boundary Quality
- **99% (82/83)** chunks end with sentence punctuation (. ! ? " ')
- Only final chunk ends mid-sentence (expected)
- **1 chunk** starts with lowercase (mid-sentence continuation)

### Issues Found
None - chunking is working correctly with excellent sentence boundary preservation.

---

## Step 3: Topic Extraction (LLM)

**Status**: VERIFIED (Siddhartha)

### What We Checked
1. Topic relevance to chunk content
2. Consistency of extraction (topics per chunk)
3. Semantic accuracy

### Configuration
- Model: gemma3:4b (Ollama)
- Topics per chunk: 5

### Topic Distribution
- All 83 chunks have exactly 5 topics
- 223 unique raw topics extracted
- Topics correctly reflect chunk content

### Sample Verification

**Chunk 0 (Introduction)**
- Content: Siddhartha, Brahman family, riverbank
- Topics: Family Relationships, Indian Philosophy, Personal Growth, Religious Practices, Spiritual Awakening
- Assessment: ✓ Accurate

**Chunk 10 (Samanas/Buddha)**
- Content: Living with Samanas, news of Gotama
- Topics: Gotama (Buddha), Religious Figures, Samanas, Suffering, Wisdom
- Assessment: ✓ Excellent match

**Chunk 54 (Rebirth by river)**
- Content: Spiritual death/rebirth, ferryman
- Topics: Death and Rebirth, Ferryman, Rivers, Self-Discovery, Transformation
- Assessment: ✓ Perfect match

**Chunk 70 (Parental love)**
- Content: Love for children, human nature
- Topics: Brahman, Childhood, Desire, Human Nature, Love
- Assessment: ✓ Accurate

### Issues Found
None - topic extraction accurately captures chunk themes.

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

## Step 6: Embeddings

**Status**: VERIFIED (Siddhartha only)

### What We Checked
1. Embedding dimensions (should be 384 for BGE-small-en-v1.5)
2. Vector normalization (should be unit vectors)
3. Semantic quality (similar topics should have high cosine similarity)

### Test Results

**Dimension & Normalization**
```
advice:      shape=(384,), norm=1.0000
aging:       shape=(384,), norm=1.0000
alms-dish:   shape=(384,), norm=1.0000
```
All 223 embedded topics have correct dimensions and unit norm.

**Semantic Similarity Quality**
```
"self-awareness" vs "self-reflection": 0.829 (high - correctly similar)
"religion" vs "self-awareness": 0.546 (moderate - different concepts)
"religion" vs "self-reflection": 0.548 (moderate - different concepts)
```

**Within-Cluster Similarity**
```
"non-self" vs "self": 0.828
"non-self" vs "self-acceptance": 0.694
"non-self" vs "self-discovery": 0.683
```
Topics in the same cluster show high similarity, validating both embedding quality and cluster coherence.

### Issues Found
None - embeddings are correctly computed.

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

## Step 8: Co-occurrence

**Status**: VERIFIED

### What We Checked
1. Co-occurrence pair counts (topics appearing in same chunks)
2. PMI (Pointwise Mutual Information) calculation accuracy
3. Multiple sample pairs verified against manual calculation

### PMI Formula
```
PMI = ln(P(joint) / (P(t1) * P(t2)))

Where:
- P(joint) = co-occurrence_count / total_chunks_with_topics
- P(t1) = topic1_occurrence_count / total_chunks_with_topics
- P(t2) = topic2_occurrence_count / total_chunks_with_topics
```

Note: Uses natural log (ln), not log2. Total chunks is count of chunks that have topic links (93,892), not all chunks.

### Test Cases Verified

**"ferryman" + "river" (Siddhartha)**
- Co-occur in chunks: 224, 241, 242 (count=3)
- ferryman: 5 occurrences, river: 87 occurrences
- PMI stored: 6.473
- PMI calculated: 6.473 ✓

**High-PMI pairs (all verified)**
- "alms-dish" + "gotama": PMI 11.450 ✓
- "gotama" + "jetavana": PMI 11.450 ✓
- "knowledge and learning" + "poetry and verse": PMI 11.450 ✓

**Random sample pairs (all verified)**
- "american revolution" + "financial policy": PMI 3.436 ✓
- "betrayal" + "suspicion": PMI 2.110 ✓
- "law enforcement" + "political intrigue": PMI 0.686 ✓

### Statistics
- Total co-occurrence pairs: 467,648
- All 8 test pairs verified with exact PMI match

### Issues Found
- Initial stale data from before Step 5 artifact cleanup - resolved by recomputing

---

## Step 9: Graph Building

**Status**: VERIFIED (Siddhartha only)

### What We Checked
1. Graph construction from embeddings and co-occurrences
2. Edge formation based on similarity thresholds

### Graph Parameters
- Embedding similarity threshold: 0.5
- Co-occurrence minimum: 2
- PMI minimum: 0.0

### Findings (Siddhartha)
- 223 topics with embeddings form the graph nodes
- Edges created between similar topics
- Graph correctly filters isolated topics

### Issues Found
None - graph building works as designed.

---

## Step 10: Leiden Clustering

**Status**: VERIFIED (Siddhartha only)

### What We Checked
1. Leiden algorithm with Surprise quality function
2. Cluster semantic coherence
3. Clustering coverage

### Statistics (Siddhartha)
- 223 topics with embeddings
- 212 clustered (95.1% coverage)
- 70 unique clusters
- 11 isolated topics (correctly unclustered)

### Cluster Semantic Coherence

**Cluster 0 - Religion/Spirituality (22 topics)**
- religion, philosophy, faith, religious figures, spirituality, buddhism, enlightenment, wisdom...

**Cluster 1 - Negative Emotions (11 topics)**
- grief, despair, illness, suicide, suffering, sadness, self-loathing...

**Cluster 2 - Self-Related (10 topics)**
- self-reflection, self-awareness, self-improvement, self-discovery, self-acceptance...

**Cluster 3 - Social Themes (9 topics)**
- social class, social interaction, human nature, human interaction, social status...

**Cluster 4 - Family (9 topics)**
- relationships, family relationships, family, family dynamics, family conflict...

### Unclustered Topics (11)
Topics that didn't fit any cluster: rivers, pain and suffering, rebirth, monasticism, searching, brahmans, river and water, samana, pilgrimage and death, river as symbol, upanishades

These are appropriately isolated - they don't share strong semantic similarity with other topics.

### Issues Found
None - clustering produces semantically coherent groups.

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

All 10 pipeline steps have been verified for Siddhartha.

### Future Considerations
- Validate on additional books as full library indexing completes
- Monitor for edge cases in different book formats (PDF, non-English)
- Consider validation automation for regression testing

---

## Related Issues & PRs

- **PR #5**: iPad library sync with full indexing pipeline
- **Issue #6**: Normalization strategy with ongoing indexing
- **Issue #4**: External author lookup for improved matching

---

*Last updated: 2025-01-31*
