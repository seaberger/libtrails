#!/usr/bin/env python3
"""
create_demo_db.py

Creates the LibTrails demo database (data/demo_library.db) from a Calibre
demo library that was set up by setup_calibre_library.py.

This populates the books table by reading metadata from the Calibre library's
metadata.db, then initializes the chunks/topics schema.

Usage:
  python scripts/create_demo_db.py --library-dir ~/Calibre_Demo_Library

The resulting DB is used with: LIBTRAILS_DB=demo uv run libtrails ...
"""

from __future__ import annotations

import argparse
import logging
import re
import sqlite3
from pathlib import Path

LOG = logging.getLogger("create_demo_db")

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def clean_html(text: str | None) -> str | None:
    if not text:
        return None
    return re.sub("<[^<]+?>", "", text).strip()


def create_demo_db(library_dir: Path, db_path: Path) -> None:
    calibre_db = library_dir / "metadata.db"
    if not calibre_db.exists():
        raise SystemExit(f"Calibre metadata.db not found at {calibre_db}")

    if db_path.exists():
        LOG.info("Removing existing demo DB: %s", db_path)
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the books table (same schema as ipad_library.db)
    cursor.executescript("""
        CREATE TABLE books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ipad_id TEXT UNIQUE,
            calibre_id INTEGER,
            title TEXT NOT NULL,
            author TEXT,
            format TEXT,
            series TEXT,
            series_index REAL,
            publisher TEXT,
            pubdate TEXT,
            description TEXT,
            has_calibre_match BOOLEAN DEFAULT 1
        );

        CREATE TABLE authors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE book_authors (
            book_id INTEGER REFERENCES books(id),
            author_id INTEGER REFERENCES authors(id),
            PRIMARY KEY (book_id, author_id)
        );

        CREATE TABLE calibre_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL
        );

        CREATE TABLE book_calibre_tags (
            book_id INTEGER REFERENCES books(id),
            tag_id INTEGER REFERENCES calibre_tags(id),
            PRIMARY KEY (book_id, tag_id)
        );

        CREATE TABLE identifiers (
            book_id INTEGER REFERENCES books(id),
            type TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (book_id, type)
        );

        CREATE INDEX idx_books_title ON books(title);
        CREATE INDEX idx_books_calibre_id ON books(calibre_id);
        CREATE INDEX idx_authors_name ON authors(name);

        CREATE VIRTUAL TABLE books_fts USING fts5(
            title, author, description, series,
            content='books',
            content_rowid='id'
        );

        CREATE TRIGGER books_ai AFTER INSERT ON books BEGIN
            INSERT INTO books_fts(rowid, title, author, description, series)
            VALUES (new.id, new.title, new.author, new.description, new.series);
        END;
    """)

    # Read all books from the Calibre demo library
    cal_conn = sqlite3.connect(f"file:{calibre_db}?mode=ro", uri=True)
    cal_conn.row_factory = sqlite3.Row
    cal_cursor = cal_conn.cursor()

    cal_cursor.execute("""
        SELECT b.id, b.title, b.author_sort, b.pubdate,
               s.name as series, b.series_index,
               c.text as description
        FROM books b
        LEFT JOIN books_series_link bsl ON b.id = bsl.book
        LEFT JOIN series s ON bsl.series = s.id
        LEFT JOIN comments c ON c.book = b.id
        ORDER BY b.id
    """)

    book_count = 0
    for cal_book in cal_cursor.fetchall():
        cal_id = cal_book["id"]

        # Get authors
        cal_cursor.execute(
            """
            SELECT a.name FROM authors a
            JOIN books_authors_link bal ON bal.author = a.id
            WHERE bal.book = ?
            """,
            (cal_id,),
        )
        authors = [row[0] for row in cal_cursor.fetchall()]
        author_str = ", ".join(authors) if authors else cal_book["author_sort"]

        # Check for EPUB format
        cal_cursor.execute(
            "SELECT format FROM data WHERE book = ? AND LOWER(format) = 'epub'",
            (cal_id,),
        )
        fmt_row = cal_cursor.fetchone()
        book_format = fmt_row[0].lower() if fmt_row else None

        # Use gutenberg ID as ipad_id for uniqueness
        cal_cursor.execute(
            """
            SELECT val FROM identifiers WHERE book = ? AND type = 'gutenberg'
            """,
            (cal_id,),
        )
        id_row = cal_cursor.fetchone()
        gutenberg_id = id_row[0] if id_row else None
        ipad_id = f"gutenberg:{gutenberg_id}" if gutenberg_id else f"calibre:{cal_id}"

        cursor.execute(
            """
            INSERT INTO books (ipad_id, calibre_id, title, author, format,
                              series, series_index, pubdate, description, has_calibre_match)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                ipad_id,
                cal_id,
                cal_book["title"],
                author_str,
                book_format or "epub",
                cal_book["series"],
                cal_book["series_index"],
                cal_book["pubdate"],
                clean_html(cal_book["description"]),
            ),
        )
        book_id = cursor.lastrowid

        # Add authors to authors table
        for author in authors:
            cursor.execute("SELECT id FROM authors WHERE name = ?", (author,))
            row = cursor.fetchone()
            if row:
                author_id = row[0]
            else:
                cursor.execute("INSERT INTO authors (name) VALUES (?)", (author,))
                author_id = cursor.lastrowid
            cursor.execute(
                "INSERT OR IGNORE INTO book_authors VALUES (?, ?)", (book_id, author_id)
            )

        # Add tags
        cal_cursor.execute(
            """
            SELECT t.name FROM tags t
            JOIN books_tags_link btl ON btl.tag = t.id
            WHERE btl.book = ?
            """,
            (cal_id,),
        )
        for (tag_name,) in cal_cursor.fetchall():
            cursor.execute("SELECT id FROM calibre_tags WHERE name = ?", (tag_name,))
            row = cursor.fetchone()
            if row:
                tag_id = row[0]
            else:
                cursor.execute("INSERT INTO calibre_tags (name) VALUES (?)", (tag_name,))
                tag_id = cursor.lastrowid
            cursor.execute(
                "INSERT OR IGNORE INTO book_calibre_tags VALUES (?, ?)", (book_id, tag_id)
            )

        # Add identifiers
        cal_cursor.execute("SELECT type, val FROM identifiers WHERE book = ?", (cal_id,))
        for id_type, id_val in cal_cursor.fetchall():
            cursor.execute(
                "INSERT OR IGNORE INTO identifiers VALUES (?, ?, ?)",
                (book_id, id_type, id_val),
            )

        book_count += 1

    conn.commit()
    cal_conn.close()

    LOG.info("Imported %s books from Calibre library", book_count)

    # Initialize chunks/topics tables using libtrails
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    # Temporarily override config before importing database module
    import os

    os.environ["LIBTRAILS_DB"] = "demo"
    os.environ["CALIBRE_LIBRARY_PATH"] = str(library_dir)

    from libtrails.database import init_chunks_table

    init_chunks_table()
    LOG.info("Initialized chunks/topics schema")

    # Print stats
    cursor.execute("SELECT COUNT(*) FROM books")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM authors")
    authors_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM calibre_tags")
    tags_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM books WHERE description IS NOT NULL")
    with_desc = cursor.fetchone()[0]

    conn.close()

    LOG.info("Demo DB created: %s", db_path)
    LOG.info("  Books: %s", total)
    LOG.info("  Authors: %s", authors_count)
    LOG.info("  Tags: %s", tags_count)
    LOG.info("  With descriptions: %s", with_desc)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create LibTrails demo database from a Calibre demo library."
    )
    ap.add_argument(
        "--library-dir",
        type=Path,
        default=Path.home() / "Calibre_Demo_Library",
        help="Calibre demo library directory (default: ~/Calibre_Demo_Library)",
    )
    ap.add_argument(
        "--db-path",
        type=Path,
        default=DATA_DIR / "demo_library.db",
        help="Output database path (default: data/demo_library.db)",
    )
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    create_demo_db(args.library_dir, args.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
