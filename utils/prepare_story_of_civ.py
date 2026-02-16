#!/usr/bin/env python3
"""Prepare Story of Civilization volumes for V2 extraction.

Splits the single EPUB into 11 volumes, creates book entries,
chunks at 1000 words, extracts themes per volume with 27b,
and optionally extracts topics with Gemini.

Usage:
    # Phase 1: Split, chunk, and extract themes (requires local 27b)
    LIBTRAILS_DB=v2 uv run python utils/prepare_story_of_civ.py

    # Phase 2: Extract topics with Gemini (after Phase 1)
    LIBTRAILS_DB=v2 uv run python utils/prepare_story_of_civ.py --extract-topics

    # Extract topics for a single volume (for testing)
    LIBTRAILS_DB=v2 uv run python utils/prepare_story_of_civ.py --extract-topics --volume "Our Oriental Heritage"
"""

import json
import re
import sqlite3
import sys
import zipfile
from pathlib import Path

from selectolax.parser import HTMLParser

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import shutil

from libtrails.chunker import chunk_text
from libtrails.config import CALIBRE_LIBRARY_PATH, DATA_DIR, IPAD_DB_PATH
from libtrails.database import get_db, save_chunk_topics
from libtrails.topic_extractor import (
    extract_book_themes,
    extract_topics_single_optimized_parallel,
)

# Volume definitions: prefix -> (title, era description for theme context)
VOLUMES = {
    "a": ("Our Oriental Heritage", "Ancient civilizations of Egypt, Mesopotamia, India, China, Japan"),
    "b": ("The Life of Greece", "Ancient Greek civilization, philosophy, art, and democracy"),
    "c": ("Caesar and Christ", "Roman Republic and Empire, rise of Christianity"),
    "d": ("The Age of Faith", "Medieval Europe, Byzantine Empire, Islamic golden age, 325-1300"),
    "e": ("The Renaissance", "Italian and Northern Renaissance, 1300-1500"),
    "f": ("The Reformation", "Protestant Reformation, religious wars, 1300-1564"),
    "g": ("The Age of Reason Begins", "Early modern philosophy and science, 1558-1648"),
    "h": ("The Age of Louis XIV", "French absolutism, Baroque culture, 1648-1715"),
    "i": ("The Age of Voltaire", "Enlightenment philosophy, science, culture, 1715-1756"),
    "j": ("Rousseau and Revolution", "Pre-revolutionary Europe, 1756-1789"),
    "k": ("The Age of Napoleon", "French Revolution aftermath, Napoleonic era, 1789-1815"),
}

EPUB_PATH = (
    CALIBRE_LIBRARY_PATH
    / "Will Durant"
    / "The Complete Story of Civilization (42928)"
    / "The Complete Story of Civilization - Will Durant.epub"
)

CHUNK_SIZE = 1000  # words per chunk
AUTHOR = "Will Durant"
SERIES = "The Story of Civilization"
CALIBRE_ID = 42928  # shared calibre_id for cover image


def _html_to_text(html_content: str) -> str:
    """Convert HTML to plain text, preserving paragraph structure."""
    # Strip XML namespaces
    content = re.sub(r'\s+xmlns(?::[a-zA-Z]+)?="[^"]*"', "", html_content)
    # Convert XHTML self-closing tags
    void_elements = {
        "area", "base", "br", "col", "embed", "hr", "img",
        "input", "link", "meta", "param", "source", "track", "wbr",
    }

    def fix_self_closing(match):
        tag = match.group(1).lower()
        if tag in void_elements:
            return match.group(0)
        return f"<{match.group(1)}{match.group(2)}></{match.group(1)}>"

    content = re.sub(r"<([a-zA-Z][a-zA-Z0-9]*)([^>]*)/>", fix_self_closing, content)
    tree = HTMLParser(content)

    for tag in tree.css("script, style, nav, header, footer"):
        tag.decompose()

    # Extract text from block elements with paragraph breaks
    parts = []
    body = tree.css_first("body") or tree.css_first("html")
    if body:
        for el in body.css("p, h1, h2, h3, h4, h5, h6, li, blockquote, div"):
            text = el.text(strip=True)
            if text and len(text.split()) > 2:
                parts.append(text)

    return "\n\n".join(parts)


def extract_volume_text(epub_path: Path, prefix: str) -> str:
    """Extract text from a single volume of the EPUB by file prefix."""
    text_parts = []

    with zipfile.ZipFile(epub_path, "r") as zf:
        html_files = sorted(
            f for f in zf.namelist()
            if f.endswith((".xhtml", ".html", ".htm"))
            and f.split("/")[-1].startswith(f"{prefix}_")
        )

        # Skip cover, copyright, TOC files
        skip_suffixes = ("_cover.html", "_copy.html", "_toc.html", "_bib.html")
        html_files = [f for f in html_files if not any(f.endswith(s) for s in skip_suffixes)]

        for name in html_files:
            try:
                content = zf.read(name).decode("utf-8", errors="ignore")
                text = _html_to_text(content)
                if len(text.split()) > 10:
                    text_parts.append(text)
            except Exception:
                continue

    return "\n\n".join(text_parts)


def create_volume_book(conn: sqlite3.Connection, title: str, author: str, volume_index: int) -> int:
    """Create a book entry for a volume. Returns the new book ID.

    calibre_id is left NULL because the UNIQUE partial index prevents sharing
    one calibre_id across 11 volumes. Covers are handled separately via
    data/covers/book_{book_id}.jpg symlinks.
    """
    cursor = conn.cursor()
    ipad_id = f"story-of-civ-vol-{volume_index}"
    cursor.execute(
        "INSERT INTO books (ipad_id, title, author, series, series_index) VALUES (?, ?, ?, ?, ?)",
        (ipad_id, f"Story of Civilization: {title}", author, SERIES, volume_index),
    )
    conn.commit()
    return cursor.lastrowid


def save_volume_chunks(conn: sqlite3.Connection, book_id: int, chunks: list[str]):
    """Save chunks for a volume book."""
    cursor = conn.cursor()
    # Clear any existing chunks
    cursor.execute(
        "DELETE FROM chunk_topics WHERE chunk_id IN (SELECT id FROM chunks WHERE book_id = ?)",
        (book_id,),
    )
    cursor.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))

    for i, text in enumerate(chunks):
        cursor.execute(
            "INSERT INTO chunks (book_id, chunk_index, text, word_count) VALUES (?, ?, ?, ?)",
            (book_id, i, text, len(text.split())),
        )
    conn.commit()


def save_book_themes(conn: sqlite3.Connection, book_id: int, themes: list[str]):
    """Save themes to the book_themes column."""
    conn.execute(
        "UPDATE books SET book_themes = ? WHERE id = ?",
        (json.dumps(themes), book_id),
    )
    conn.commit()


def setup_volume_covers(conn: sqlite3.Connection):
    """Copy the Story of Civilization cover for each volume as book_{id}.jpg."""
    # Find the source cover from Calibre
    source_cover = (
        CALIBRE_LIBRARY_PATH
        / "Will Durant"
        / "The Complete Story of Civilization (42928)"
        / "cover.jpg"
    )
    if not source_cover.exists():
        print("  WARNING: Source cover not found, skipping cover setup")
        return

    covers_dir = DATA_DIR / "covers"
    covers_dir.mkdir(exist_ok=True)

    existing = get_existing_volumes(conn)
    for title, book_id in existing.items():
        dest = covers_dir / f"book_{book_id}.jpg"
        if not dest.exists():
            shutil.copy2(source_cover, dest)
            print(f"  Cover: book_{book_id}.jpg ({title})")


def get_existing_volumes(conn: sqlite3.Connection) -> dict[str, int]:
    """Find existing Story of Civilization volume entries. Returns title -> book_id."""
    rows = conn.execute(
        "SELECT id, title FROM books WHERE title LIKE 'Story of Civilization:%'"
    ).fetchall()
    return {row[1].replace("Story of Civilization: ", ""): row[0] for row in rows}


def phase1_split_and_chunk():
    """Split EPUB into volumes, create book entries, chunk at 1000 words."""
    print("=" * 60)
    print("Phase 1: Split EPUB into volumes and chunk")
    print("=" * 60)

    if not EPUB_PATH.exists():
        print(f"ERROR: EPUB not found at {EPUB_PATH}")
        return False

    conn = sqlite3.connect(str(IPAD_DB_PATH))
    existing = get_existing_volumes(conn)

    for vol_idx, (prefix, (title, era)) in enumerate(VOLUMES.items(), start=1):
        if title in existing:
            book_id = existing[title]
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE book_id = ?", (book_id,)
            ).fetchone()[0]
            print(f"\n[{prefix}] {title}: already exists (book_id={book_id}, {chunk_count} chunks)")
            continue

        print(f"\n[{prefix}] Extracting: {title}...")
        text = extract_volume_text(EPUB_PATH, prefix)
        word_count = len(text.split())
        print(f"  Words: {word_count:,}")

        if word_count < 100:
            print(f"  WARNING: Very little text extracted, skipping")
            continue

        # Create book entry
        book_id = create_volume_book(conn, title, AUTHOR, vol_idx)
        print(f"  Created book_id={book_id}")

        # Chunk
        chunks = chunk_text(text, CHUNK_SIZE)
        print(f"  Chunks: {len(chunks)} (~{CHUNK_SIZE} words/chunk)")

        # Save chunks
        save_volume_chunks(conn, book_id, chunks)
        print(f"  Saved {len(chunks)} chunks")

    # Set up cover images for all volumes
    print("\nSetting up covers...")
    setup_volume_covers(conn)

    conn.close()
    return True


def phase2_extract_themes():
    """Extract themes for each volume using 27b local model."""
    print("\n" + "=" * 60)
    print("Phase 2: Extract themes per volume (gemma3:27b)")
    print("=" * 60)

    conn = sqlite3.connect(str(IPAD_DB_PATH))
    existing = get_existing_volumes(conn)

    for prefix, (title, era) in VOLUMES.items():
        if title not in existing:
            print(f"\n[{prefix}] {title}: no book entry, run Phase 1 first")
            continue

        book_id = existing[title]

        # Check if themes already exist
        row = conn.execute("SELECT book_themes FROM books WHERE id = ?", (book_id,)).fetchone()
        if row[0]:
            themes = json.loads(row[0])
            print(f"\n[{prefix}] {title}: already has themes: {', '.join(themes[:3])}...")
            continue

        # Get sample text (first 1000 words)
        sample_row = conn.execute(
            "SELECT text FROM chunks WHERE book_id = ? ORDER BY chunk_index LIMIT 3",
            (book_id,),
        ).fetchall()
        sample_text = " ".join(r[0] for r in sample_row)

        full_title = f"The Story of Civilization: {title}"
        print(f"\n[{prefix}] Extracting themes for: {title}...")
        print(f"  Era context: {era}")

        themes = extract_book_themes(
            title=full_title,
            author=AUTHOR,
            series=SERIES,
            description=f"Volume covering {era}. Part of Will Durant's 11-volume history of civilization.",
            sample_text=sample_text,
        )

        print(f"  Themes: {', '.join(themes)}")
        save_book_themes(conn, book_id, themes)

    conn.close()
    return True


def phase3_extract_topics(volume_filter: str | None = None, workers: int = 20):
    """Extract topics for each volume's chunks using Gemini 2.0 Flash."""
    print("\n" + "=" * 60)
    print("Phase 3: Extract topics per chunk (gemini/gemini-2.0-flash)")
    print("=" * 60)

    conn = sqlite3.connect(str(IPAD_DB_PATH))
    existing = get_existing_volumes(conn)

    total_extracted = 0
    total_chunks = 0

    for prefix, (title, era) in VOLUMES.items():
        if volume_filter and title != volume_filter:
            continue

        if title not in existing:
            print(f"\n[{prefix}] {title}: no book entry, run Phase 1 first")
            continue

        book_id = existing[title]

        # Get themes
        row = conn.execute("SELECT book_themes FROM books WHERE id = ?", (book_id,)).fetchone()
        if not row[0]:
            print(f"\n[{prefix}] {title}: no themes, run Phase 2 first")
            continue
        themes = json.loads(row[0])

        # Get chunks needing topics
        all_chunk_ids = [
            r[0] for r in conn.execute(
                "SELECT id FROM chunks WHERE book_id = ? ORDER BY chunk_index", (book_id,)
            ).fetchall()
        ]
        done_ids = {
            r[0] for r in conn.execute(
                "SELECT DISTINCT chunk_id FROM chunk_topics WHERE chunk_id IN ({})".format(
                    ",".join("?" * len(all_chunk_ids))
                ),
                all_chunk_ids,
            ).fetchall()
        } if all_chunk_ids else set()

        pending_ids = [cid for cid in all_chunk_ids if cid not in done_ids]

        if not pending_ids:
            print(f"\n[{prefix}] {title}: all {len(all_chunk_ids)} chunks done")
            continue

        # Get text for pending chunks
        pending_texts = []
        for cid in pending_ids:
            row = conn.execute("SELECT text FROM chunks WHERE id = ?", (cid,)).fetchone()
            pending_texts.append(row[0])

        print(f"\n[{prefix}] {title}: {len(pending_ids)}/{len(all_chunk_ids)} chunks to process")
        print(f"  Themes: {', '.join(themes[:3])}...")

        # Define save callback
        def on_chunk_done(idx, topics, _pending=pending_ids):
            if topics:
                save_chunk_topics(_pending[idx], topics)

        def on_progress(done, total):
            if done % 50 == 0 or done == total:
                print(f"  Processing: {done}/{total} ({100 * done // total}%)")

        # Extract topics
        topics_per_chunk = extract_topics_single_optimized_parallel(
            pending_texts,
            book_title=f"Story of Civilization: {title}",
            author=AUTHOR,
            book_themes=themes,
            model="gemini/gemini-2.0-flash",
            max_workers=workers,
            progress_callback=on_progress,
            save_callback=on_chunk_done,
            use_extended_prompt=True,
        )

        extracted = sum(1 for t in topics_per_chunk if t)
        total_extracted += extracted
        total_chunks += len(pending_ids)
        print(f"  Done: {extracted}/{len(pending_ids)} chunks extracted")

    conn.close()
    print(f"\nTotal: {total_extracted}/{total_chunks} chunks extracted")
    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Story of Civilization for V2 extraction")
    parser.add_argument("--extract-topics", action="store_true", help="Run Phase 3 (topic extraction with Gemini)")
    parser.add_argument("--volume", type=str, default=None, help="Process only this volume (by title)")
    parser.add_argument("--workers", type=int, default=20, help="Parallel workers for topic extraction")
    parser.add_argument("--themes-only", action="store_true", help="Run only Phase 2 (theme extraction)")
    args = parser.parse_args()

    if args.extract_topics:
        phase3_extract_topics(volume_filter=args.volume, workers=args.workers)
    elif args.themes_only:
        phase2_extract_themes()
    else:
        # Default: run Phase 1 + Phase 2
        if phase1_split_and_chunk():
            phase2_extract_themes()

    print("\nDone!")


if __name__ == "__main__":
    main()
