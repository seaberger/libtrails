#!/usr/bin/env python3
"""Targeted enrichment for well-known books with clean titles."""
import json
import sqlite3
import urllib.request
import urllib.parse
import re

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
db_path = str(PROJECT_ROOT / "data" / "ipad_library.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Manual fixes for well-known books with messy titles
known_books = [
    ("A Single Man", "Christopher Isherwood"),
    ("Destination Void", "Frank Herbert"),
    ("The Count of Monte Cristo", "Alexandre Dumas"),
    ("Let's Explore Diabetes with Owls", "David Sedaris"),
    ("The Wind-Up Bird Chronicle", "Haruki Murakami"),
    ("Tenth of December", "George Saunders"),
    ("Jorge Luis Borges Selected Poems", "Jorge Luis Borges"),
    ("Deep Learning with PyTorch Step by Step", "Daniel Voigt Godoy"),
    ("Philosophy The Basics", "Nigel Warburton"),
    ("Hell Yeah or No", "Derek Sivers"),
    ("How to Live", "Derek Sivers"),
    ("The Portable MFA in Creative Writing", "New York Writers Workshop"),
    ("James Joyce", "Richard Ellmann"),
    ("Show Your Work", "Austin Kleon"),
    ("The Lean Startup", "Eric Ries"),
    ("Writing to Learn", "William Zinsser"),
    ("Introduction to Algorithms", "Thomas Cormen"),
    ("An Introduction to Statistical Learning", "Gareth James"),
    ("Algorithms", "Robert Sedgewick"),
]

def fetch_google(title, author):
    try:
        query = f"intitle:{title}+inauthor:{author.split()[0]}"
        url = f"https://www.googleapis.com/books/v1/volumes?q={urllib.parse.quote(query)}&maxResults=1"
        with urllib.request.urlopen(url, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            if data.get('items'):
                vol = data['items'][0]['volumeInfo']
                return vol.get('description'), vol.get('categories', [])
    except Exception as e:
        print(f"  Error: {e}")
    return None, []

updated = 0
for clean_title, clean_author in known_books:
    # Find matching book in DB
    cursor.execute("""
        SELECT id, title FROM books 
        WHERE (description IS NULL OR description = '')
          AND (title LIKE ? OR title LIKE ?)
    """, (f"%{clean_title[:20]}%", f"%{clean_title.split()[0]}%{clean_title.split()[-1]}%"))
    
    matches = cursor.fetchall()
    for book_id, db_title in matches:
        desc, categories = fetch_google(clean_title, clean_author)
        if desc:
            cursor.execute("UPDATE books SET description = ? WHERE id = ?", (desc, book_id))
            print(f"  Updated: {db_title[:50]} -> {len(desc)} chars")
            updated += 1
            
            # Add categories as tags
            for cat in categories[:3]:
                cursor.execute("INSERT OR IGNORE INTO calibre_tags (name) VALUES (?)", (cat,))
                cursor.execute("SELECT id FROM calibre_tags WHERE name = ?", (cat,))
                tag_id = cursor.fetchone()[0]
                cursor.execute("INSERT OR IGNORE INTO book_calibre_tags VALUES (?, ?)", (book_id, tag_id))

conn.commit()
conn.close()
print(f"\nUpdated {updated} more books")
