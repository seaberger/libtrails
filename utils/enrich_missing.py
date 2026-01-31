#!/usr/bin/env python3
"""Fetch missing descriptions from Open Library and Google Books."""
import json
import sqlite3
import urllib.request
import urllib.parse
import time
import re

db_path = "/Users/seanbergman/Repositories/calibre_lib/data/ipad_library.db"
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def clean_html(text):
    if not text:
        return None
    return re.sub('<[^<]+?>', '', text).strip()

def fetch_open_library(isbn=None, title=None, author=None):
    """Try Open Library API."""
    try:
        if isbn:
            # Clean ISBN
            isbn_clean = re.sub(r'[^0-9X]', '', isbn.upper())
            url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn_clean}&jscmd=data&format=json"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data:
                    key = list(data.keys())[0]
                    book = data[key]
                    # Get description (might need work lookup)
                    desc = None
                    if 'description' in book:
                        desc = book['description'] if isinstance(book['description'], str) else book['description'].get('value')
                    
                    # Get subjects as tags
                    subjects = [s['name'] for s in book.get('subjects', [])][:10]
                    
                    return {'description': desc, 'subjects': subjects, 'source': 'openlibrary'}
        
        # Search by title/author if no ISBN match
        if title:
            query = urllib.parse.quote(f"{title} {author or ''}")
            url = f"https://openlibrary.org/search.json?q={query}&limit=1"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode('utf-8'))
                if data.get('docs'):
                    doc = data['docs'][0]
                    subjects = doc.get('subject', [])[:10]
                    # Open Library search doesn't return descriptions, need work key
                    work_key = doc.get('key')
                    desc = None
                    if work_key:
                        work_url = f"https://openlibrary.org{work_key}.json"
                        with urllib.request.urlopen(work_url, timeout=10) as w_response:
                            work = json.loads(w_response.read().decode('utf-8'))
                            if 'description' in work:
                                desc = work['description'] if isinstance(work['description'], str) else work['description'].get('value')
                    
                    return {'description': desc, 'subjects': subjects, 'source': 'openlibrary'}
    except Exception as e:
        pass
    return None

def fetch_google_books(isbn=None, title=None, author=None):
    """Try Google Books API."""
    try:
        if isbn:
            isbn_clean = re.sub(r'[^0-9X]', '', isbn.upper())
            query = f"isbn:{isbn_clean}"
        else:
            query = f"intitle:{title}"
            if author:
                query += f"+inauthor:{author.split(',')[0]}"
        
        url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(query)}&maxResults=1"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get('items'):
                vol = data['items'][0]['volumeInfo']
                desc = vol.get('description')
                categories = vol.get('categories', [])
                return {'description': desc, 'subjects': categories, 'source': 'google'}
    except Exception as e:
        pass
    return None

# Get books missing descriptions
cursor.execute("""
    SELECT b.id, b.title, b.author,
           (SELECT value FROM identifiers WHERE book_id = b.id AND type = 'isbn') as isbn
    FROM books b
    WHERE b.description IS NULL OR b.description = ''
""")
missing = cursor.fetchall()

print(f"Attempting to enrich {len(missing)} books missing descriptions...")

enriched = 0
for i, book in enumerate(missing):
    book_id = book['id']
    title = book['title']
    author = book['author']
    isbn = book['isbn']
    
    # Try Open Library first
    result = fetch_open_library(isbn=isbn, title=title, author=author)
    
    # Fall back to Google Books
    if not result or not result.get('description'):
        result = fetch_google_books(isbn=isbn, title=title, author=author)
    
    if result:
        desc = result.get('description')
        subjects = result.get('subjects', [])
        source = result.get('source', 'unknown')
        
        if desc:
            cursor.execute("UPDATE books SET description = ? WHERE id = ?", (clean_html(desc), book_id))
            enriched += 1
            print(f"  [{enriched}] {title[:50]} - got description from {source}")
        
        # Add subjects as calibre tags if we got any
        for subject in subjects[:5]:
            cursor.execute("INSERT OR IGNORE INTO calibre_tags (name) VALUES (?)", (subject,))
            cursor.execute("SELECT id FROM calibre_tags WHERE name = ?", (subject,))
            tag_id = cursor.fetchone()[0]
            cursor.execute("INSERT OR IGNORE INTO book_calibre_tags (book_id, tag_id) VALUES (?, ?)", 
                          (book_id, tag_id))
    
    # Be nice to APIs
    time.sleep(0.3)
    
    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{len(missing)}...")
        conn.commit()

conn.commit()
conn.close()

print(f"\nEnriched {enriched} out of {len(missing)} books with descriptions")
