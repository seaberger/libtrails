#!/usr/bin/env python3
"""Use Calibre's fetch-ebook-metadata to fill in missing tags."""
import sqlite3
import subprocess
import re
import time

CALIBRE_FETCH = "/Applications/calibre.app/Contents/MacOS/fetch-ebook-metadata"
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
db_path = str(PROJECT_ROOT / "data" / "ipad_library.db")

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get books without any tags (neither iPad nor Calibre)
cursor.execute("""
    SELECT b.id, b.title, b.author FROM books b
    WHERE NOT EXISTS (SELECT 1 FROM book_calibre_tags WHERE book_id = b.id)
      AND NOT EXISTS (SELECT 1 FROM book_ipad_tags WHERE book_id = b.id)
""")
missing = cursor.fetchall()
print(f"Fetching tags for {len(missing)} books without tags...")

def parse_output(output):
    result = {}
    for line in output.split('\n'):
        if ':' in line and not line.startswith(' '):
            parts = line.split(':', 1)
            result[parts[0].strip().lower()] = parts[1].strip() if len(parts) > 1 else ''
    return result

updated = 0
for book_id, title, author in missing:
    clean_title = re.sub(r'\s*\([^)]*\)\s*$', '', title)
    clean_title = re.sub(r'[_]+', ' ', clean_title)
    clean_author = author if author and author != '[Unknown Author]' else ''
    clean_author = re.sub(r'[_\d]+$', '', clean_author)
    
    try:
        cmd = [CALIBRE_FETCH, '--title', clean_title]
        if clean_author:
            cmd.extend(['--author', clean_author.split(',')[0]])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout:
            data = parse_output(result.stdout)
            tags = data.get('tags', '')
            
            if tags:
                added = 0
                for tag in tags.split(','):
                    tag = tag.strip()
                    if tag:
                        cursor.execute("INSERT OR IGNORE INTO calibre_tags (name) VALUES (?)", (tag,))
                        cursor.execute("SELECT id FROM calibre_tags WHERE name = ?", (tag,))
                        tag_id = cursor.fetchone()[0]
                        cursor.execute("INSERT OR IGNORE INTO book_calibre_tags VALUES (?, ?)", 
                                     (book_id, tag_id))
                        added += 1
                
                if added:
                    updated += 1
                    print(f"  [{updated}] {title[:45]} - added {added} tags")
                    conn.commit()
        
        time.sleep(0.3)
        
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        pass

conn.close()
print(f"\nAdded tags to {updated} out of {len(missing)} books")
