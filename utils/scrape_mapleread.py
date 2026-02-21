#!/usr/bin/env python3
"""Scrape all books from MapleRead server and save as JSON."""
import json
import os
import re
import sys
import urllib.request
from html import unescape
from pathlib import Path

# Add project root to path so we can import config
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from libtrails.config import get_ipad_url

# iPad URL: CLI arg > env var > ~/.libtrails/config.yaml > error
BASE_URL = os.environ.get("MAPLEREAD_URL") or get_ipad_url()
if not BASE_URL:
    print("Error: No iPad URL configured.")
    print("Set it in ~/.libtrails/config.yaml under ipad.default_url,")
    print("or pass via MAPLEREAD_URL env var.")
    print("See config.example.yaml for details.")
    sys.exit(1)
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
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
output_path = str(PROJECT_ROOT / "data" / "ipad_library.json")

with open(output_path, 'w') as f:
    json.dump(books, f, indent=2)

print(f"Saved to {output_path}")
