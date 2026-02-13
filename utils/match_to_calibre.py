#!/usr/bin/env python3
"""Match iPad library books to Calibre metadata."""
import json
import sqlite3
import re
from pathlib import Path

# Load iPad library
PROJECT_ROOT = Path(__file__).parent.parent

with open(str(PROJECT_ROOT / "data" / "ipad_library.json")) as f:
    ipad_books = json.load(f)

print(f"Loaded {len(ipad_books)} iPad books")

# Connect to Calibre database (read-only)
calibre_db = "/Users/seanbergman/Calibre_Main_Library/metadata.db"
conn = sqlite3.connect(f"file:{calibre_db}?mode=ro", uri=True)
conn.row_factory = sqlite3.Row

def normalize(s):
    """Normalize string for matching."""
    if not s:
        return ""
    # Remove punctuation, lowercase, collapse spaces
    s = re.sub(r'[^\w\s]', '', s.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def get_calibre_metadata(cursor, book_id):
    """Get full metadata for a Calibre book."""
    # Get basic info
    cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
    book = dict(cursor.fetchone())
    
    # Get authors
    cursor.execute("""
        SELECT a.name FROM authors a
        JOIN books_authors_link bal ON a.id = bal.author
        WHERE bal.book = ?
    """, (book_id,))
    book['authors'] = [r['name'] for r in cursor.fetchall()]
    
    # Get tags
    cursor.execute("""
        SELECT t.name FROM tags t
        JOIN books_tags_link btl ON t.id = btl.tag
        WHERE btl.book = ?
    """, (book_id,))
    book['tags'] = [r['name'] for r in cursor.fetchall()]
    
    # Get series
    cursor.execute("""
        SELECT s.name FROM series s
        JOIN books_series_link bsl ON s.id = bsl.series
        WHERE bsl.book = ?
    """, (book_id,))
    series = cursor.fetchone()
    book['series'] = series['name'] if series else None
    
    # Get description
    cursor.execute("SELECT text FROM comments WHERE book = ?", (book_id,))
    comment = cursor.fetchone()
    book['description'] = comment['text'] if comment else None
    
    # Get identifiers (ISBN, etc)
    cursor.execute("SELECT type, val FROM identifiers WHERE book = ?", (book_id,))
    book['identifiers'] = {r['type']: r['val'] for r in cursor.fetchall()}
    
    # Get publisher
    cursor.execute("""
        SELECT p.name FROM publishers p
        JOIN books_publishers_link bpl ON p.id = bpl.publisher
        WHERE bpl.book = ?
    """, (book_id,))
    pub = cursor.fetchone()
    book['publisher'] = pub['name'] if pub else None
    
    # Get formats
    cursor.execute("SELECT format FROM data WHERE book = ?", (book_id,))
    book['formats'] = [r['format'] for r in cursor.fetchall()]
    
    return book

# Build lookup index from Calibre
print("Building Calibre lookup index...")
cursor = conn.cursor()

# Index by normalized title
cursor.execute("SELECT id, title, author_sort FROM books")
calibre_index = {}
for row in cursor.fetchall():
    key = normalize(row['title'])
    if key not in calibre_index:
        calibre_index[key] = []
    calibre_index[key].append({
        'id': row['id'],
        'title': row['title'],
        'author_sort': row['author_sort']
    })

print(f"Indexed {len(calibre_index)} unique titles from Calibre")

# Match iPad books to Calibre
matched = []
unmatched = []

for ipad_book in ipad_books:
    ipad_title_norm = normalize(ipad_book['title'])
    ipad_author_norm = normalize(ipad_book['author'])
    
    candidates = calibre_index.get(ipad_title_norm, [])
    
    match = None
    if len(candidates) == 1:
        # Single match - use it
        match = candidates[0]
    elif len(candidates) > 1:
        # Multiple matches - try to match by author too
        for c in candidates:
            if normalize(c['author_sort']) == ipad_author_norm or \
               ipad_author_norm in normalize(c['author_sort']) or \
               normalize(c['author_sort']) in ipad_author_norm:
                match = c
                break
        if not match:
            match = candidates[0]  # Take first if no author match
    
    if match:
        # Get full metadata
        calibre_meta = get_calibre_metadata(cursor, match['id'])
        
        matched.append({
            'ipad': ipad_book,
            'calibre_id': match['id'],
            'calibre': {
                'title': calibre_meta['title'],
                'authors': calibre_meta['authors'],
                'tags': calibre_meta['tags'],
                'series': calibre_meta['series'],
                'series_index': calibre_meta.get('series_index'),
                'description': calibre_meta['description'],
                'identifiers': calibre_meta['identifiers'],
                'publisher': calibre_meta['publisher'],
                'pubdate': str(calibre_meta.get('pubdate', '')),
                'formats': calibre_meta['formats'],
                'has_cover': bool(calibre_meta.get('has_cover'))
            }
        })
    else:
        unmatched.append(ipad_book)

conn.close()

print(f"\nMatched: {len(matched)} books")
print(f"Unmatched: {len(unmatched)} books")

# Save results
output_path = str(PROJECT_ROOT / "data" / "ipad_library_enriched.json")
with open(output_path, 'w') as f:
    json.dump(matched, f, indent=2)
print(f"\nSaved enriched data to {output_path}")

# Save unmatched for review
if unmatched:
    unmatched_path = str(PROJECT_ROOT / "data" / "ipad_unmatched.json")
    with open(unmatched_path, 'w') as f:
        json.dump(unmatched, f, indent=2)
    print(f"Saved unmatched books to {unmatched_path}")

# Stats on enriched data
has_description = sum(1 for m in matched if m['calibre']['description'])
has_isbn = sum(1 for m in matched if 'isbn' in m['calibre']['identifiers'])
has_series = sum(1 for m in matched if m['calibre']['series'])
has_tags = sum(1 for m in matched if m['calibre']['tags'])

print(f"\nEnrichment stats:")
print(f"  With description: {has_description}")
print(f"  With ISBN: {has_isbn}")
print(f"  With series: {has_series}")
print(f"  With Calibre tags: {has_tags}")
