#!/usr/bin/env python3
"""Scrape all books from MapleRead server and save as JSON."""
import re
import json
import urllib.request
from html import unescape

BASE_URL = "http://192.168.1.124:8082"
SECTIONS = list(range(0, 28))  # 0-27 covers #, A-Z, ~

books = []

for sec in SECTIONS:
    url = f"{BASE_URL}/?set=0&sort=title&sec={sec}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            html = response.read().decode('utf-8')
            
        # Extract book entries using regex
        pattern = r"<a class='title' href='/book\?id=([^']+)'>([^<]+)</a><br /><span class='author'>([^<]+)</span>"
        matches = re.findall(pattern, html)
        
        for book_id, title, author in matches:
            # Parse format from ID (e.g., "abc123.epub")
            if '.' in book_id:
                file_id, fmt = book_id.rsplit('.', 1)
            else:
                file_id, fmt = book_id, 'unknown'
                
            books.append({
                'id': file_id,
                'title': unescape(title),
                'author': unescape(author),
                'format': fmt
            })
        print(f"Section {sec}: found {len(matches)} books")
    except Exception as e:
        print(f"Section {sec}: error - {e}")

print(f"\nTotal books: {len(books)}")

# Save to JSON
output_path = "/Users/seanbergman/Repositories/calibre_lib/data/ipad_library.json"
import os
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, 'w') as f:
    json.dump(books, f, indent=2)

print(f"Saved to {output_path}")
