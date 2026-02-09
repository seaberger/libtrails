"""
Compare batched vs single-chunk topic extraction on Siddhartha.

Loads Siddhartha's 90 chunks from the v2 database, then runs:
  A) 5-chunk batch extraction (current pipeline prompt)
  B) Single-chunk extraction with DSPy-optimized instruction + 4 few-shot demos

Both use the same Gemini model. Compares topic quality, diversity, and timing.

Usage:
    LIBTRAILS_DB=v2 uv run python experiments/compare_batch_vs_single.py \
        --model gemini/gemini-2.5-flash-lite

    # Or with a local model:
    LIBTRAILS_DB=v2 uv run python experiments/compare_batch_vs_single.py \
        --model gemma3:4b
"""

import argparse
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from libtrails.config import IPAD_DB_PATH, TOPICS_PER_CHUNK
from libtrails.embeddings import embed_texts
from libtrails.topic_extractor import (
    _extract_batch,
    _extract_single_contextualized,
    _filter_topics,
    extract_book_themes,
    extract_topics_single_optimized,
    extract_topics_single_optimized_parallel,
)

SIDDHARTHA_BOOK_ID = 517


def load_chunks(book_id: int) -> list[dict]:
    """Load chunks for a book from the v2 database."""
    conn = sqlite3.connect(IPAD_DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, chunk_index, text, word_count FROM chunks "
        "WHERE book_id = ? ORDER BY chunk_index",
        (book_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_book_meta(book_id: int) -> dict:
    """Load book metadata from the v2 database."""
    conn = sqlite3.connect(IPAD_DB_PATH)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, title, author, book_themes FROM books WHERE id = ?",
        (book_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise ValueError(f"Book {book_id} not found in {IPAD_DB_PATH}")
    meta = dict(row)
    if meta.get("book_themes"):
        meta["book_themes"] = json.loads(meta["book_themes"])
    else:
        meta["book_themes"] = []
    return meta


def run_batched(
    chunks: list[str], context: str, model: str, batch_size: int = 5
) -> tuple[list[list[str]], float]:
    """Run batched extraction on all chunks. Returns (topics_per_chunk, total_seconds)."""
    results: list[list[str]] = [[] for _ in range(len(chunks))]
    t0 = time.time()

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]

        batch_results = _extract_batch(
            batch_chunks, context, model, TOPICS_PER_CHUNK, batch_start
        )

        if batch_results is not None:
            for i, topics in enumerate(batch_results):
                results[batch_start + i] = _filter_topics(topics)
        else:
            # Fallback: extract individually
            print(f"  [!] Batch {batch_start}-{batch_end} failed, falling back to individual")
            for i, chunk in enumerate(batch_chunks):
                topics = _extract_single_contextualized(
                    chunk, context, model, TOPICS_PER_CHUNK
                )
                results[batch_start + i] = _filter_topics(topics)

        done = min(batch_end, len(chunks))
        print(f"  Batched: {done}/{len(chunks)} chunks", flush=True)

    elapsed = time.time() - t0
    return results, elapsed


def run_single_optimized(
    chunks: list[str], context: str, model: str
) -> tuple[list[list[str]], float]:
    """Run single-chunk DSPy-optimized extraction on all chunks."""
    results: list[list[str]] = []
    t0 = time.time()

    for i, chunk in enumerate(chunks):
        raw = extract_topics_single_optimized(chunk, context, model)
        results.append(_filter_topics(raw))

        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  Single: {i + 1}/{len(chunks)} chunks", flush=True)

    elapsed = time.time() - t0
    return results, elapsed


def compute_topic_stats(topics_per_chunk: list[list[str]]) -> dict:
    """Compute summary statistics for a set of per-chunk topics."""
    all_topics = [t for chunk_topics in topics_per_chunk for t in chunk_topics]
    unique = set(all_topics)
    counts = Counter(all_topics)

    # Word count distribution
    word_counts = [len(t.split()) for t in all_topics]
    single_word = sum(1 for wc in word_counts if wc == 1)
    multi_word = sum(1 for wc in word_counts if wc >= 2)

    # Empty chunks
    empty = sum(1 for ct in topics_per_chunk if not ct)

    return {
        "total_topics": len(all_topics),
        "unique_topics": len(unique),
        "empty_chunks": empty,
        "avg_per_chunk": len(all_topics) / max(len(topics_per_chunk), 1),
        "single_word_pct": single_word / max(len(all_topics), 1) * 100,
        "multi_word_pct": multi_word / max(len(all_topics), 1) * 100,
        "top_10": counts.most_common(10),
        "all_topics": all_topics,
        "unique_set": unique,
    }


def compute_semantic_overlap(topics_a: set[str], topics_b: set[str]) -> dict:
    """Compute semantic overlap between two topic sets using BGE embeddings."""
    if not topics_a or not topics_b:
        return {"exact_overlap": 0, "avg_similarity": 0.0}

    list_a = sorted(topics_a)
    list_b = sorted(topics_b)

    # Exact overlap
    exact = topics_a & topics_b

    # Embed both sets
    vecs_a = np.array(embed_texts(list_a))
    vecs_b = np.array(embed_texts(list_b))

    # Pairwise cosine similarity (BGE embeddings are normalized)
    sim_matrix = vecs_a @ vecs_b.T

    # For each topic in A, find best match in B
    a_to_b = sim_matrix.max(axis=1)
    # For each topic in B, find best match in A
    b_to_a = sim_matrix.max(axis=0)

    return {
        "exact_overlap": len(exact),
        "a_unique": len(topics_a),
        "b_unique": len(topics_b),
        "avg_a_to_b": float(a_to_b.mean()),
        "avg_b_to_a": float(b_to_a.mean()),
        "high_match_a_to_b": int((a_to_b > 0.8).sum()),
        "high_match_b_to_a": int((b_to_a > 0.8).sum()),
    }


def print_side_by_side(
    chunk_idx: int,
    chunk_text: str,
    batch_topics: list[str],
    single_topics: list[str],
):
    """Print a side-by-side comparison for one chunk."""
    print(f"\n{'='*80}")
    print(f"CHUNK {chunk_idx} ({len(chunk_text.split())} words)")
    print(f"{'='*80}")
    print(chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text)
    print()
    print(f"  {'BATCHED':<40} {'SINGLE (DSPy)':<40}")
    print(f"  {'-'*38}   {'-'*38}")
    max_len = max(len(batch_topics), len(single_topics))
    for i in range(max_len):
        bt = batch_topics[i] if i < len(batch_topics) else ""
        st = single_topics[i] if i < len(single_topics) else ""
        print(f"  {bt:<40} {st:<40}")


def main():
    parser = argparse.ArgumentParser(description="Compare batch vs single-chunk extraction")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-lite",
        help="Model to use for chunk extraction (default: gemini/gemini-2.5-flash-lite)",
    )
    parser.add_argument(
        "--theme-model",
        default="gemini/gemini-3-flash-preview",
        help="Model for book theme extraction (default: gemini/gemini-3-flash-preview)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Batch size for batched extraction (default: 5)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Limit to first N chunks for quick testing (0 = all)",
    )
    parser.add_argument(
        "--book-id",
        type=int,
        default=SIDDHARTHA_BOOK_ID,
        help=f"Book ID to compare on (default: {SIDDHARTHA_BOOK_ID} = Siddhartha)",
    )
    parser.add_argument(
        "--side-by-side",
        type=int,
        default=5,
        help="Number of chunks to show side-by-side (default: 5)",
    )
    parser.add_argument(
        "--parallel-only",
        action="store_true",
        help="Only run parallel single-chunk extraction (skip batch and sequential)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=30,
        help="Max concurrent workers for parallel extraction (default: 30)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  Batch vs Single-Chunk Topic Extraction Comparison")
    print("=" * 60)
    print(f"  DB:          {IPAD_DB_PATH}")
    print(f"  Chunk model: {args.model}")
    print(f"  Theme model: {args.theme_model}")
    print(f"  Batch size:  {args.batch_size}")
    print()

    # Load book metadata
    meta = load_book_meta(args.book_id)
    print(f"  Book: {meta['title']} by {meta['author']}")

    # Load chunks
    chunks_data = load_chunks(args.book_id)
    if not chunks_data:
        print(f"  ERROR: No chunks found for book {args.book_id}. Run `index --id {args.book_id} --reindex` first.")
        return

    if args.max_chunks > 0:
        chunks_data = chunks_data[: args.max_chunks]
        print(f"  (Limited to first {args.max_chunks} chunks)")

    chunk_texts = [c["text"] for c in chunks_data]
    print(f"  Chunks: {len(chunk_texts)}")
    print()

    # Get or extract book themes
    book_themes = meta.get("book_themes", [])
    if not book_themes:
        print(f"  Extracting book themes with {args.theme_model}...")
        from libtrails.database import get_calibre_book_metadata

        calibre_meta = get_calibre_book_metadata(meta.get("calibre_id"))
        book_themes = extract_book_themes(
            title=meta["title"],
            author=meta.get("author", "Unknown"),
            tags=calibre_meta.get("tags") if calibre_meta else None,
            description=calibre_meta.get("description") if calibre_meta else None,
            sample_text=chunk_texts[0] if chunk_texts else None,
            model=args.theme_model,
        )

    print(f"  Book themes: {', '.join(book_themes)}")
    print()

    # Build context string (shared by both approaches)
    context_parts = [f"Book: {meta['title']} by {meta['author']}"]
    if book_themes:
        context_parts.append(f"Book themes: {', '.join(book_themes)}")
    context = "\n".join(context_parts)

    if args.parallel_only:
        # --- Parallel-only mode ---
        print("-" * 60)
        print(f"  [C] PARALLEL SINGLE-CHUNK ({args.workers} workers)")
        print("-" * 60)

        t0 = time.time()
        parallel_topics = extract_topics_single_optimized_parallel(
            chunk_texts,
            book_title=meta["title"],
            author=meta.get("author", "Unknown"),
            book_themes=book_themes,
            model=args.model,
            max_workers=args.workers,
            progress_callback=lambda done, total: (
                print(f"  Parallel: {done}/{total} chunks", flush=True)
                if done % 10 == 0 or done == total
                else None
            ),
        )
        parallel_time = time.time() - t0

        par_stats = compute_topic_stats(parallel_topics)

        print(f"\n  {'Metric':<30} {'Parallel':>12}")
        print(f"  {'-'*42}")
        print(f"  {'Time (seconds)':<30} {parallel_time:>12.1f}")
        print(f"  {'Time per chunk (s)':<30} {parallel_time/len(chunk_texts):>12.2f}")
        print(f"  {'Effective RPM':<30} {len(chunk_texts)/parallel_time*60:>12.0f}")
        print(f"  {'Total topics':<30} {par_stats['total_topics']:>12}")
        print(f"  {'Unique topics':<30} {par_stats['unique_topics']:>12}")
        print(f"  {'Empty chunks':<30} {par_stats['empty_chunks']:>12}")
        print(f"  {'Avg topics/chunk':<30} {par_stats['avg_per_chunk']:>12.1f}")
        print(f"  {'Multi-word topics %':<30} {par_stats['multi_word_pct']:>11.1f}%")

        print(f"\n  Top 10 topics:")
        for topic, count in par_stats["top_10"]:
            print(f"    {count:3d}x  {topic}")

        # Show sample chunks
        n_compare = min(args.side_by_side, len(chunk_texts))
        if n_compare > 0:
            indices = np.linspace(0, len(chunk_texts) - 1, n_compare, dtype=int)
            for idx in indices:
                print(f"\n{'='*80}")
                print(f"CHUNK {idx} ({len(chunk_texts[idx].split())} words)")
                print(f"{'='*80}")
                text = chunk_texts[idx]
                print(text[:300] + "..." if len(text) > 300 else text)
                print(f"\n  Topics:")
                for t in parallel_topics[idx]:
                    print(f"    - {t}")

        print()
        return

    # --- Run A: Batched extraction ---
    print("-" * 60)
    print("  [A] BATCHED EXTRACTION (5-chunk batches)")
    print("-" * 60)
    batch_topics, batch_time = run_batched(
        chunk_texts, context, args.model, args.batch_size
    )

    # --- Run B: Single-chunk DSPy-optimized (sequential) ---
    print()
    print("-" * 60)
    print("  [B] SINGLE-CHUNK EXTRACTION (DSPy-optimized, sequential)")
    print("-" * 60)
    single_topics, single_time = run_single_optimized(
        chunk_texts, context, args.model
    )

    # --- Run C: Single-chunk DSPy-optimized (parallel) ---
    print()
    print("-" * 60)
    print(f"  [C] SINGLE-CHUNK EXTRACTION (DSPy-optimized, {args.workers} workers)")
    print("-" * 60)

    t0 = time.time()
    parallel_topics = extract_topics_single_optimized_parallel(
        chunk_texts,
        book_title=meta["title"],
        author=meta.get("author", "Unknown"),
        book_themes=book_themes,
        model=args.model,
        max_workers=args.workers,
        progress_callback=lambda done, total: (
            print(f"  Parallel: {done}/{total} chunks", flush=True)
            if done % 10 == 0 or done == total
            else None
        ),
    )
    parallel_time = time.time() - t0

    # --- Analysis ---
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)

    batch_stats = compute_topic_stats(batch_topics)
    single_stats = compute_topic_stats(single_topics)
    par_stats = compute_topic_stats(parallel_topics)

    print(f"\n  {'Metric':<30} {'Batched':>12} {'Sequential':>12} {'Parallel':>12}")
    print(f"  {'-'*66}")
    print(f"  {'Time (seconds)':<30} {batch_time:>12.1f} {single_time:>12.1f} {parallel_time:>12.1f}")
    print(f"  {'Time per chunk (s)':<30} {batch_time/len(chunk_texts):>12.2f} {single_time/len(chunk_texts):>12.2f} {parallel_time/len(chunk_texts):>12.2f}")
    print(f"  {'Effective RPM':<30} {'':>12} {'':>12} {len(chunk_texts)/parallel_time*60:>12.0f}")
    print(f"  {'Total topics':<30} {batch_stats['total_topics']:>12} {single_stats['total_topics']:>12} {par_stats['total_topics']:>12}")
    print(f"  {'Unique topics':<30} {batch_stats['unique_topics']:>12} {single_stats['unique_topics']:>12} {par_stats['unique_topics']:>12}")
    print(f"  {'Empty chunks':<30} {batch_stats['empty_chunks']:>12} {single_stats['empty_chunks']:>12} {par_stats['empty_chunks']:>12}")
    print(f"  {'Avg topics/chunk':<30} {batch_stats['avg_per_chunk']:>12.1f} {single_stats['avg_per_chunk']:>12.1f} {par_stats['avg_per_chunk']:>12.1f}")
    print(f"  {'Multi-word topics %':<30} {batch_stats['multi_word_pct']:>11.1f}% {single_stats['multi_word_pct']:>11.1f}% {par_stats['multi_word_pct']:>11.1f}%")

    # Top topics
    print(f"\n  Top 10 topics (BATCHED):")
    for topic, count in batch_stats["top_10"]:
        print(f"    {count:3d}x  {topic}")

    print(f"\n  Top 10 topics (SEQUENTIAL SINGLE):")
    for topic, count in single_stats["top_10"]:
        print(f"    {count:3d}x  {topic}")

    print(f"\n  Top 10 topics (PARALLEL SINGLE):")
    for topic, count in par_stats["top_10"]:
        print(f"    {count:3d}x  {topic}")

    # Side-by-side comparison (batch vs parallel)
    n_compare = min(args.side_by_side, len(chunk_texts))
    if n_compare > 0:
        indices = np.linspace(0, len(chunk_texts) - 1, n_compare, dtype=int)
        for idx in indices:
            print_side_by_side(
                idx,
                chunk_texts[idx],
                batch_topics[idx],
                parallel_topics[idx],
            )

    # Summary verdict
    print(f"\n{'='*60}")
    print("  VERDICT")
    print(f"{'='*60}")
    seq_ratio = single_time / max(batch_time, 0.01)
    par_ratio = batch_time / max(parallel_time, 0.01)
    print(f"  Sequential single: {seq_ratio:.1f}x slower than batched")
    print(f"  Parallel single:   {par_ratio:.1f}x {'faster' if par_ratio > 1 else 'slower'} than batched")
    print(f"  Speedup from parallelism: {single_time / max(parallel_time, 0.01):.1f}x")
    print(f"  Diversity: {batch_stats['unique_topics']} (batch) vs {par_stats['unique_topics']} (parallel)")

    print()


if __name__ == "__main__":
    main()
