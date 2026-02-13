#!/usr/bin/env python3
"""Create SQLite database for iPad library with Calibre metadata."""
import json
import sqlite3
import re
from pathlib import Path

# Load enriched data
PROJECT_ROOT = Path(__file__).parent.parent

with open(str(PROJECT_ROOT / "data" / "ipad_library_enriched.json")) as f:
    enriched = json.load(f)

with open(str(PROJECT_ROOT / "data" / "ipad_unmatched.json")) as f:
    unmatched = json.load(f)

db_path = str(PROJECT_ROOT / "data" / "ipad_library.db")
Path(db_path).unlink(missing_ok=True)  # Remove if exists

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Create tables
cursor.executescript("""
-- Main books table
CREATE TABLE books (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ipad_id TEXT UNIQUE NOT NULL,
    calibre_id INTEGER,
    title TEXT NOT NULL,
    author TEXT,
    format TEXT,
    series TEXT,
    series_index REAL,
    publisher TEXT,
    pubdate TEXT,
    description TEXT,
    has_calibre_match BOOLEAN DEFAULT 0
);

-- Authors (many-to-many)
CREATE TABLE authors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE book_authors (
    book_id INTEGER REFERENCES books(id),
    author_id INTEGER REFERENCES authors(id),
    PRIMARY KEY (book_id, author_id)
);

-- Tags from iPad (MapleRead)
CREATE TABLE ipad_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE book_ipad_tags (
    book_id INTEGER REFERENCES books(id),
    tag_id INTEGER REFERENCES ipad_tags(id),
    PRIMARY KEY (book_id, tag_id)
);

-- Tags from Calibre
CREATE TABLE calibre_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL
);

CREATE TABLE book_calibre_tags (
    book_id INTEGER REFERENCES books(id),
    tag_id INTEGER REFERENCES calibre_tags(id),
    PRIMARY KEY (book_id, tag_id)
);

-- Identifiers (ISBN, Amazon, etc)
CREATE TABLE identifiers (
    book_id INTEGER REFERENCES books(id),
    type TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (book_id, type)
);

-- Create indexes
CREATE INDEX idx_books_title ON books(title);
CREATE INDEX idx_books_calibre_id ON books(calibre_id);
CREATE INDEX idx_authors_name ON authors(name);

-- Full-text search on books
CREATE VIRTUAL TABLE books_fts USING fts5(
    title, author, description, series,
    content='books',
    content_rowid='id'
);

-- Trigger to keep FTS in sync
CREATE TRIGGER books_ai AFTER INSERT ON books BEGIN
    INSERT INTO books_fts(rowid, title, author, description, series)
    VALUES (new.id, new.title, new.author, new.description, new.series);
END;
""")

def get_or_create(cursor, table, name):
    """Get or create an entity, return ID."""
    cursor.execute(f"SELECT id FROM {table} WHERE name = ?", (name,))
    row = cursor.fetchone()
    if row:
        return row[0]
    cursor.execute(f"INSERT INTO {table} (name) VALUES (?)", (name,))
    return cursor.lastrowid

def clean_html(text):
    """Remove HTML tags from description."""
    if not text:
        return None
    return re.sub('<[^<]+?>', '', text).strip()

# Insert enriched books
for item in enriched:
    ipad = item['ipad']
    cal = item['calibre']
    
    cursor.execute("""
        INSERT INTO books (ipad_id, calibre_id, title, author, format, series, 
                          series_index, publisher, pubdate, description, has_calibre_match)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
    """, (
        ipad['id'],
        item['calibre_id'],
        cal['title'],
        ', '.join(cal['authors']),
        ipad['format'],
        cal['series'],
        cal.get('series_index'),
        cal['publisher'],
        cal['pubdate'],
        clean_html(cal['description'])
    ))
    book_id = cursor.lastrowid
    
    # Add authors
    for author in cal['authors']:
        author_id = get_or_create(cursor, 'authors', author)
        cursor.execute("INSERT OR IGNORE INTO book_authors VALUES (?, ?)", (book_id, author_id))
    
    # Add iPad tags
    for tag in ipad.get('tags', []):
        tag_id = get_or_create(cursor, 'ipad_tags', tag)
        cursor.execute("INSERT OR IGNORE INTO book_ipad_tags VALUES (?, ?)", (book_id, tag_id))
    
    # Add Calibre tags
    for tag in cal.get('tags', []):
        tag_id = get_or_create(cursor, 'calibre_tags', tag)
        cursor.execute("INSERT OR IGNORE INTO book_calibre_tags VALUES (?, ?)", (book_id, tag_id))
    
    # Add identifiers
    for id_type, id_val in cal.get('identifiers', {}).items():
        cursor.execute("INSERT OR IGNORE INTO identifiers VALUES (?, ?, ?)", 
                      (book_id, id_type, id_val))

# Insert unmatched books (no Calibre data)
for ipad in unmatched:
    cursor.execute("""
        INSERT INTO books (ipad_id, title, author, format, has_calibre_match)
        VALUES (?, ?, ?, ?, 0)
    """, (ipad['id'], ipad['title'], ipad['author'], ipad['format']))
    book_id = cursor.lastrowid
    
    # Add iPad tags
    for tag in ipad.get('tags', []):
        tag_id = get_or_create(cursor, 'ipad_tags', tag)
        cursor.execute("INSERT OR IGNORE INTO book_ipad_tags VALUES (?, ?)", (book_id, tag_id))

conn.commit()

# Print stats
cursor.execute("SELECT COUNT(*) FROM books")
print(f"Total books: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM books WHERE has_calibre_match = 1")
print(f"With Calibre match: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM authors")
print(f"Unique authors: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM ipad_tags")
print(f"iPad tags: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM calibre_tags")
print(f"Calibre tags: {cursor.fetchone()[0]}")

cursor.execute("SELECT COUNT(*) FROM books WHERE description IS NOT NULL")
print(f"With descriptions: {cursor.fetchone()[0]}")

conn.close()
print(f"\nDatabase saved to: {db_path}")
