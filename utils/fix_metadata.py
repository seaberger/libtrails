#!/usr/bin/env python3
"""Fix malformed titles and authors in the database."""
import sqlite3
import re

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
db_path = str(PROJECT_ROOT / "data" / "ipad_library.db")
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Author fixes: "Last| First" -> "First Last", remove suffixes like _42834
author_fixes = {
    "Isherwood| Christopher": "Christopher Isherwood",
    "Haruki Marakami": "Haruki Murakami",  # typo
    "Siversd Dereke": "Derek Sivers",
    "murakami| haruki": "Haruki Murakami",
    "Warburton| Nigel(Author)": "Nigel Warburton",
    "CODING| MARK": "Mark Coding",
    " [Unknown Author] ": None,  # Keep as unknown
}

# Title cleanups
def clean_title(title):
    # Remove Z-Library suffixes
    title = re.sub(r'\s*\([^)]*Z-Library[^)]*\)', '', title)
    # Remove author names in parens at end
    title = re.sub(r'\s*\([A-Z][^)]+\)\s*$', '', title)
    # Remove trailing underscores and numbers
    title = re.sub(r'_+\d*$', '', title)
    # Remove leading underscores
    title = re.sub(r'^_+', '', title)
    return title.strip()

def clean_author(author):
    if not author or author.strip() == '[Unknown Author]':
        return None
    
    # Check manual fixes first
    if author in author_fixes:
        return author_fixes[author]
    
    # Fix "Last| First" format
    if '|' in author:
        parts = [p.strip() for p in author.split('|')]
        if len(parts) == 2:
            return f"{parts[1]} {parts[0]}"
    
    # Remove trailing _NNNNN
    author = re.sub(r'_\d+$', '', author)
    
    # Extract from Z-Library format: "Title (Author Name) (Z-Library)"
    match = re.search(r'\(([A-Z][^)]+)\)\s*\(Z-Library\)', author)
    if match:
        return match.group(1)
    
    return author.strip()

def infer_from_zlib_title(title):
    """Try to extract author from Z-Library filename format."""
    # Pattern: "Title (Author Name) (Z-Library)"
    match = re.search(r'^(.+?)\s*\(([A-Z][^)]+(?:,\s*[A-Z][^)]+)*)\)\s*(?:\(Z-Library\))?$', title)
    if match:
        return clean_title(match.group(1)), match.group(2)
    
    # Pattern: "Title (Author Name, Other Author etc.)"
    match = re.search(r'^(.+?)\s*\(([A-Z][a-z]+\s+[A-Z][^)]+)\)$', title)
    if match:
        return match.group(1).strip(), match.group(2)
    
    return None, None

# Process all books
cursor.execute("SELECT id, title, author FROM books")
books = cursor.fetchall()

fixed_titles = 0
fixed_authors = 0
inferred = 0

for book_id, title, author in books:
    new_title = clean_title(title)
    new_author = clean_author(author)
    
    # Try to infer from Z-Library title format
    if (not new_author or new_author == '[Unknown Author]') and '(' in title:
        inferred_title, inferred_author = infer_from_zlib_title(title)
        if inferred_author:
            new_title = inferred_title or new_title
            new_author = inferred_author
            inferred += 1
    
    updates = []
    if new_title != title:
        updates.append(f"title = '{new_title}'")
        fixed_titles += 1
    if new_author != author and new_author:
        updates.append(f"author = '{new_author}'")
        fixed_authors += 1
    
    if updates:
        sql = f"UPDATE books SET {', '.join(updates)} WHERE id = ?"
        try:
            cursor.execute(sql, (book_id,))
        except:
            # Handle quotes in titles/authors
            if new_title != title:
                cursor.execute("UPDATE books SET title = ? WHERE id = ?", (new_title, book_id))
            if new_author != author and new_author:
                cursor.execute("UPDATE books SET author = ? WHERE id = ?", (new_author, book_id))

conn.commit()
print(f"Fixed {fixed_titles} titles")
print(f"Fixed {fixed_authors} authors")
print(f"Inferred {inferred} authors from filenames")

# Show some examples
print("\nSample cleaned entries:")
cursor.execute("""
    SELECT title, author FROM books 
    WHERE has_calibre_match = 0
    LIMIT 10
""")
for title, author in cursor.fetchall():
    print(f"  {title[:50]:<52} | {author or 'Unknown'}")

conn.close()
