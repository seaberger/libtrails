#!/usr/bin/env python3
"""
backfill_descriptions.py

Pulls book descriptions from the Calibre demo library (metadata.db) into the
LibTrails demo database (demo_library.db). Descriptions are stored as HTML in
Calibre's `comments` table — this script strips HTML tags before writing.

For books missing Calibre descriptions (typically ~4 books), fetches the
description from Project Gutenberg's website as a fallback.

Usage:
  uv run python scripts/backfill_descriptions.py
"""

from __future__ import annotations

import re
import sqlite3
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).parent.parent
DEMO_DB = PROJECT_ROOT / "data" / "demo_library.db"
CALIBRE_DB = Path.home() / "Calibre_Demo_Library" / "metadata.db"


def clean_html(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub("<[^<]+?>", "", text).strip()


def fetch_gutenberg_description(gutenberg_id: int) -> str | None:
    """Fetch the description paragraph from a Gutenberg ebook page."""
    if httpx is None:
        return None
    url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}"
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=15.0)
        resp.raise_for_status()
    except httpx.HTTPError as e:
        print(f"  HTTP error fetching {url}: {e}")
        return None

    html = resp.text
    # Gutenberg uses <td property="dcterms:description"> for the description
    match = re.search(
        r'<td property="dcterms:description"[^>]*>(.*?)</td>',
        html,
        re.DOTALL,
    )
    if not match:
        # Try the subject field as fallback
        match = re.search(
            r'<td property="dcterms:subject"[^>]*>(.*?)</td>',
            html,
            re.DOTALL,
        )
    if match:
        return clean_html(match.group(1))
    return None


def rebuild_fts(demo: sqlite3.Connection) -> None:
    """Drop and recreate the FTS index with current book data."""
    demo.executescript("""
        DROP TRIGGER IF EXISTS books_ai;
        DROP TRIGGER IF EXISTS books_au;
        DROP TRIGGER IF EXISTS books_ad;
        DROP TABLE IF EXISTS books_fts;

        CREATE VIRTUAL TABLE books_fts USING fts5(
            title, author, description, series,
            content='books',
            content_rowid='id'
        );

        INSERT INTO books_fts(rowid, title, author, description, series)
        SELECT id, title, author, description, series FROM books;

        CREATE TRIGGER books_ai AFTER INSERT ON books BEGIN
            INSERT INTO books_fts(rowid, title, author, description, series)
            VALUES (new.id, new.title, new.author, new.description, new.series);
        END;

        CREATE TRIGGER books_au AFTER UPDATE ON books BEGIN
            INSERT INTO books_fts(books_fts, rowid, title, author, description, series)
            VALUES ('delete', old.id, old.title, old.author, old.description, old.series);
            INSERT INTO books_fts(rowid, title, author, description, series)
            VALUES (new.id, new.title, new.author, new.description, new.series);
        END;

        CREATE TRIGGER books_ad AFTER DELETE ON books BEGIN
            INSERT INTO books_fts(books_fts, rowid, title, author, description, series)
            VALUES ('delete', old.id, old.title, old.author, old.description, old.series);
        END;
    """)


def main() -> int:
    if not CALIBRE_DB.exists():
        print(f"Calibre DB not found: {CALIBRE_DB}")
        return 1
    if not DEMO_DB.exists():
        print(f"Demo DB not found: {DEMO_DB}")
        return 1

    demo = sqlite3.connect(DEMO_DB)
    calibre = sqlite3.connect(f"file:{CALIBRE_DB}?mode=ro", uri=True)

    # Build mapping: calibre_id -> cleaned description
    rows = calibre.execute("SELECT c.book, c.text FROM comments c").fetchall()
    descriptions = {cal_id: clean_html(text) for cal_id, text in rows}
    calibre.close()

    # Update demo books that have a calibre_id
    books = demo.execute(
        "SELECT id, calibre_id, ipad_id, title FROM books WHERE calibre_id IS NOT NULL"
    ).fetchall()

    from_calibre = 0
    from_gutenberg = 0
    still_missing = []

    for book_id, calibre_id, ipad_id, title in books:
        desc = descriptions.get(calibre_id)
        if desc:
            demo.execute("UPDATE books SET description = ? WHERE id = ?", (desc, book_id))
            from_calibre += 1
        elif ipad_id and ipad_id.startswith("gutenberg:"):
            gid = int(ipad_id.split(":")[1])
            print(f"  Fetching Gutenberg description for '{title}' (id={gid})...")
            desc = fetch_gutenberg_description(gid)
            if desc:
                demo.execute("UPDATE books SET description = ? WHERE id = ?", (desc, book_id))
                from_gutenberg += 1
            else:
                still_missing.append((book_id, title))
            time.sleep(2)  # polite rate limiting
        else:
            still_missing.append((book_id, title))

    demo.commit()

    # Rebuild FTS index to reflect updated descriptions
    rebuild_fts(demo)
    demo.commit()
    demo.close()

    total = len(books)
    print(f"\nUpdated {from_calibre + from_gutenberg}/{total} books with descriptions:")
    print(f"  From Calibre: {from_calibre}")
    print(f"  From Gutenberg: {from_gutenberg}")
    if still_missing:
        print(f"  Still missing: {len(still_missing)}")
        for bid, title in still_missing:
            print(f"    - [{bid}] {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
