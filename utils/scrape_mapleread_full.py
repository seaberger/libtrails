#!/usr/bin/env python3
"""Scrape all books with tags from MapleRead server."""
import re
import json
import urllib.request
from html import unescape
from pathlib import Path
from urllib.parse import unquote
import time

BASE_URL = "http://192.168.1.124:8082"

# First get all tags
print("Fetching all tags...")
tags_url = f"{BASE_URL}/tags?set=0"
with urllib.request.urlopen(tags_url, timeout=30) as response:
    html = response.read().decode('utf-8')

# Extract tag names and counts
tag_pattern = r"<a class='title' href='/tags\?set=0&amp;tag=([^']+)'>([^<]+)</a><br /><span class='content'>Total (\d+) books</span>"
tag_matches = re.findall(tag_pattern, html)
print(f"Found {len(tag_matches)} unique tags")

# Build tag -> books mapping
tag_to_books = {}
for encoded_tag, tag_name, count in tag_matches:
    tag_to_books[unescape(tag_name)] = []

# Now fetch books for each tag (this gives us tag associations)
# For efficiency, let's just get the books with their tags from the tag pages
print("\nFetching books by tag (to build tag associations)...")

book_tags = {}  # book_id -> list of tags
books_data = {}  # book_id -> {title, author, format}

count = 0
total_tags = len(tag_matches)
for encoded_tag, tag_name, book_count in tag_matches:
    count += 1
    if count % 50 == 0:
        print(f"  Processing tag {count}/{total_tags}...")
    
    try:
        tag_url = f"{BASE_URL}/tags?set=0&tag={encoded_tag}"
        with urllib.request.urlopen(tag_url, timeout=10) as response:
            tag_html = response.read().decode('utf-8')
        
        # Extract books from this tag page
        book_pattern = r"<a class='title' href='/book\?id=([^']+)'>([^<]+)</a><br /><span class='author'>([^<]+)</span>"
        book_matches = re.findall(book_pattern, tag_html)
        
        tag_clean = unescape(tag_name)
        for book_id, title, author in book_matches:
            if '.' in book_id:
                file_id, fmt = book_id.rsplit('.', 1)
            else:
                file_id, fmt = book_id, 'unknown'
            
            # Add book data
            if file_id not in books_data:
                books_data[file_id] = {
                    'title': unescape(title),
                    'author': unescape(author),
                    'format': fmt
                }
            
            # Add tag association
            if file_id not in book_tags:
                book_tags[file_id] = []
            if tag_clean not in book_tags[file_id]:
                book_tags[file_id].append(tag_clean)
                
    except Exception as e:
        print(f"  Error on tag '{tag_name}': {e}")
    
    time.sleep(0.05)  # Be nice to the server

# Also fetch by title to catch any books without tags
print("\nFetching remaining books by title sections...")
SECTIONS = list(range(0, 28))
for sec in SECTIONS:
    url = f"{BASE_URL}/?set=0&sort=title&sec={sec}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            html = response.read().decode('utf-8')
        
        book_pattern = r"<a class='title' href='/book\?id=([^']+)'>([^<]+)</a><br /><span class='author'>([^<]+)</span>"
        matches = re.findall(book_pattern, html)
        
        for book_id, title, author in matches:
            if '.' in book_id:
                file_id, fmt = book_id.rsplit('.', 1)
            else:
                file_id, fmt = book_id, 'unknown'
            
            if file_id not in books_data:
                books_data[file_id] = {
                    'title': unescape(title),
                    'author': unescape(author),
                    'format': fmt
                }
            if file_id not in book_tags:
                book_tags[file_id] = []
    except Exception as e:
        print(f"  Error on section {sec}: {e}")

# Combine into final structure
books = []
for book_id, data in books_data.items():
    books.append({
        'id': book_id,
        'title': data['title'],
        'author': data['author'],
        'format': data['format'],
        'tags': book_tags.get(book_id, [])
    })

# Sort by title
books.sort(key=lambda x: x['title'].lower())

print(f"\nTotal books: {len(books)}")
books_with_tags = sum(1 for b in books if b['tags'])
print(f"Books with tags: {books_with_tags}")
all_tags = set()
for b in books:
    all_tags.update(b['tags'])
print(f"Unique tags: {len(all_tags)}")

# Save to JSON
PROJECT_ROOT = Path(__file__).parent.parent
output_path = str(PROJECT_ROOT / "data" / "ipad_library.json")
with open(output_path, 'w') as f:
    json.dump(books, f, indent=2)
print(f"\nSaved to {output_path}")

# Also save unique tags
tags_path = str(PROJECT_ROOT / "data" / "ipad_tags.json")
with open(tags_path, 'w') as f:
    json.dump(sorted(list(all_tags)), f, indent=2)
print(f"Saved tags to {tags_path}")
