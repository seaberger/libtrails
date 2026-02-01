"""iPad library sync functionality."""

import re
import time
import urllib.request
from html import unescape
from typing import Optional

from .config import DEFAULT_MODEL
from .database import get_calibre_db, get_db

# Regex pattern for parsing book entries from MapleRead HTML
BOOK_PATTERN = re.compile(
    r"<a class='title' href='/book\?id=([^']+)'>([^<]+)</a><br /><span class='author'>([^<]+)</span>"
)


def _parse_book_id(book_id: str) -> tuple[str, str]:
    """Parse book ID into (file_id, format)."""
    if '.' in book_id:
        return book_id.rsplit('.', 1)
    return book_id, 'unknown'


def _fetch_url(url: str, timeout: int = 10) -> Optional[str]:
    """Fetch URL and return HTML content, or None on error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode('utf-8')
    except Exception:
        return None


def _extract_books_from_html(html: str) -> list[tuple[str, str, str, str]]:
    """
    Extract book tuples from HTML.

    Returns list of (file_id, title, author, format) tuples.
    """
    matches = BOOK_PATTERN.findall(html)
    results = []
    for book_id, title, author in matches:
        file_id, fmt = _parse_book_id(book_id)
        results.append((file_id, unescape(title), unescape(author), fmt))
    return results


def scrape_ipad_library(base_url: str, progress_callback: Optional[callable] = None) -> list[dict]:
    """
    Scrape all books from MapleRead HTTP server.

    Returns a list of book dicts with: id, title, author, format, tags
    """
    books_data = {}  # book_id -> {title, author, format}
    book_tags = {}   # book_id -> list of tags

    # First fetch all tags
    if progress_callback:
        progress_callback("Fetching tags from iPad...")

    tags_url = f"{base_url}/tags?set=0"
    try:
        with urllib.request.urlopen(tags_url, timeout=30) as response:
            html = response.read().decode('utf-8')
    except Exception as e:
        raise ConnectionError(f"Could not connect to iPad at {base_url}: {e}")

    # Extract tag names
    tag_pattern = r"<a class='title' href='/tags\?set=0&amp;tag=([^']+)'>([^<]+)</a><br /><span class='content'>Total (\d+) books</span>"
    tag_matches = re.findall(tag_pattern, html)

    if progress_callback:
        progress_callback(f"Found {len(tag_matches)} tags, fetching books...")

    # Fetch books by tag to build tag associations
    for i, (encoded_tag, tag_name, book_count) in enumerate(tag_matches):
        if progress_callback and i % 50 == 0:
            progress_callback(f"Processing tag {i+1}/{len(tag_matches)}...")

        tag_url = f"{base_url}/tags?set=0&tag={encoded_tag}"
        tag_html = _fetch_url(tag_url)
        if not tag_html:
            continue

        tag_clean = unescape(tag_name)
        for file_id, title, author, fmt in _extract_books_from_html(tag_html):
            if file_id not in books_data:
                books_data[file_id] = {'title': title, 'author': author, 'format': fmt}

            if file_id not in book_tags:
                book_tags[file_id] = []
            if tag_clean not in book_tags[file_id]:
                book_tags[file_id].append(tag_clean)

        time.sleep(0.05)  # Be nice to the server

    # Also fetch by title to catch books without tags
    if progress_callback:
        progress_callback("Fetching remaining books by title...")

    for sec in range(28):
        url = f"{base_url}/?set=0&sort=title&sec={sec}"
        html = _fetch_url(url)
        if not html:
            continue

        for file_id, title, author, fmt in _extract_books_from_html(html):
            if file_id not in books_data:
                books_data[file_id] = {'title': title, 'author': author, 'format': fmt}
            if file_id not in book_tags:
                book_tags[file_id] = []

    # Combine into final structure
    books = []
    for book_id, data in books_data.items():
        books.append({
            'ipad_id': book_id,
            'title': data['title'],
            'author': data['author'],
            'format': data['format'],
            'ipad_tags': book_tags.get(book_id, [])
        })

    books.sort(key=lambda x: x['title'].lower())
    return books


def get_existing_ipad_ids() -> set[str]:
    """Get set of iPad book IDs already in our database."""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT ipad_id FROM books WHERE ipad_id IS NOT NULL")
        return {row[0] for row in cursor.fetchall()}


def find_new_books(scraped_books: list[dict]) -> list[dict]:
    """Find books from scrape that aren't in our database yet."""
    existing_ids = get_existing_ipad_ids()
    return [b for b in scraped_books if b['ipad_id'] not in existing_ids]


def normalize_for_matching(s: str) -> str:
    """Normalize string for title/author matching."""
    if not s:
        return ""
    s = re.sub(r'[^\w\s]', '', s.lower())
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def match_to_calibre(book: dict) -> Optional[dict]:
    """
    Match a book to Calibre library.

    Returns Calibre metadata dict if found, None otherwise.
    """
    title_norm = normalize_for_matching(book['title'])
    author_norm = normalize_for_matching(book['author'])

    with get_calibre_db() as conn:
        cursor = conn.cursor()

        # Search by normalized title
        cursor.execute("""
            SELECT id, title, author_sort FROM books
            WHERE LOWER(REPLACE(REPLACE(title, '.', ''), ',', '')) LIKE ?
        """, (f"%{title_norm}%",))

        candidates = cursor.fetchall()

        if not candidates:
            return None

        # Find best match by author
        match = None
        for c in candidates:
            c_author_norm = normalize_for_matching(c['author_sort'])
            if c_author_norm == author_norm or \
               author_norm in c_author_norm or \
               c_author_norm in author_norm:
                match = c
                break

        if not match:
            match = candidates[0]

        # Get full metadata
        return get_calibre_metadata(conn, match['id'])


def get_calibre_metadata(conn, book_id: int) -> dict:
    """Get full metadata for a Calibre book."""
    cursor = conn.cursor()

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
    book['calibre_tags'] = [r['name'] for r in cursor.fetchall()]

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

    # Get identifiers
    cursor.execute("SELECT type, val FROM identifiers WHERE book = ?", (book_id,))
    book['identifiers'] = {r['type']: r['val'] for r in cursor.fetchall()}

    # Get formats
    cursor.execute("SELECT format FROM data WHERE book = ?", (book_id,))
    book['formats'] = [r['format'] for r in cursor.fetchall()]

    return book


def add_book_to_database(ipad_book: dict, calibre_meta: Optional[dict]) -> int:
    """
    Add a new book to our database.

    Returns the new book ID.
    """
    with get_db() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO books (
                ipad_id, title, author, format,
                calibre_id, series, series_index,
                publisher, pubdate, description, has_calibre_match
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ipad_book['ipad_id'],
            ipad_book['title'],
            ipad_book['author'],
            ipad_book.get('format'),
            calibre_meta['id'] if calibre_meta else None,
            calibre_meta.get('series') if calibre_meta else None,
            calibre_meta.get('series_index') if calibre_meta else None,
            calibre_meta.get('publisher') if calibre_meta else None,
            str(calibre_meta.get('pubdate', '')) if calibre_meta else None,
            calibre_meta.get('description') if calibre_meta else None,
            1 if calibre_meta else 0,
        ))

        conn.commit()
        return cursor.lastrowid


def sync_ipad_library(
    ipad_url: str,
    dry_run: bool = False,
    skip_index: bool = False,
    model: str = DEFAULT_MODEL,
    progress_callback: Optional[callable] = None
) -> dict:
    """
    Main sync function - scrapes iPad, finds new books, adds to database.

    Returns a summary dict with counts.
    """
    # Scrape iPad
    if progress_callback:
        progress_callback("Connecting to iPad...")

    scraped_books = scrape_ipad_library(ipad_url, progress_callback)

    if progress_callback:
        progress_callback(f"Found {len(scraped_books)} books on iPad")

    # Find new books
    new_books = find_new_books(scraped_books)

    if progress_callback:
        progress_callback(f"Found {len(new_books)} new books to add")

    if dry_run:
        return {
            "total_on_ipad": len(scraped_books),
            "new_books": len(new_books),
            "matched_to_calibre": 0,
            "added_to_db": 0,
            "indexed": 0,
            "dry_run": True,
            "new_book_titles": [b['title'] for b in new_books]
        }

    # Match and add new books
    matched = 0
    added = 0
    books_to_index = []

    for i, book in enumerate(new_books):
        if progress_callback:
            progress_callback(f"Processing {i+1}/{len(new_books)}: {book['title'][:50]}...")

        # Try to match to Calibre
        calibre_meta = match_to_calibre(book)
        if calibre_meta:
            matched += 1

        # Add to database
        book_id = add_book_to_database(book, calibre_meta)
        added += 1

        if calibre_meta and not skip_index:
            books_to_index.append({
                'id': book_id,
                'calibre_id': calibre_meta['id'],
                'title': book['title']
            })

    # Index new books if requested
    indexed = 0
    index_failed = []

    if books_to_index and not skip_index:
        if progress_callback:
            progress_callback(f"Indexing {len(books_to_index)} new books...")

        from .indexer import index_books_batch

        def book_progress(current, total, result):
            if progress_callback:
                status = "✓" if result.success else ("⊘" if result.skipped else "✗")
                progress_callback(f"  [{current}/{total}] {status} {result.book_id}")

        book_ids = [b['id'] for b in books_to_index]
        batch_result = index_books_batch(
            book_ids,
            model=model,
            progress_callback=progress_callback,
            book_callback=book_progress,
        )

        indexed = batch_result['successful']
        index_failed = batch_result.get('failed_details', [])

    result = {
        "total_on_ipad": len(scraped_books),
        "new_books": len(new_books),
        "matched_to_calibre": matched,
        "added_to_db": added,
        "indexed": indexed,
        "index_failed": len(index_failed),
        "dry_run": False,
        "books_to_index": books_to_index
    }

    return result
