#!/usr/bin/env python3
"""Use Calibre's fetch-ebook-metadata to fill in missing descriptions."""
import sqlite3
import subprocess
import re
import time

CALIBRE_FETCH = "/Applications/calibre.app/Contents/MacOS/fetch-ebook-metadata"
db_path = "/Users/seanbergman/Repositories/calibre_lib/data/ipad_library.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get books missing descriptions
cursor.execute("""
    SELECT id, title, author FROM books 
    WHERE description IS NULL OR description = ''
""")
missing = cursor.fetchall()
print(f"Fetching metadata for {len(missing)} books...")

def clean_html(text):
    if not text:
        return None
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def parse_output(output):
    """Parse calibre fetch-ebook-metadata output."""
    result = {}
    lines = output.split('\n')
    current_field = None
    current_value = []
    
    for line in lines:
        if ':' in line and not line.startswith(' '):
            # Save previous field
            if current_field and current_value:
                result[current_field] = '\n'.join(current_value).strip()
            
            # Parse new field
            parts = line.split(':', 1)
            current_field = parts[0].strip().lower()
            current_value = [parts[1].strip()] if len(parts) > 1 else []
        elif current_field:
            current_value.append(line)
    
    # Save last field
    if current_field and current_value:
        result[current_field] = '\n'.join(current_value).strip()
    
    return result

updated = 0
for book_id, title, author in missing:
    # Clean up title and author for search
    clean_title = re.sub(r'\s*\([^)]*\)\s*$', '', title)  # Remove trailing parens
    clean_title = re.sub(r'[_]+', ' ', clean_title)  # Replace underscores
    clean_author = author if author and author != '[Unknown Author]' else ''
    clean_author = re.sub(r'[_\d]+$', '', clean_author)  # Remove trailing numbers
    
    try:
        cmd = [CALIBRE_FETCH, '--title', clean_title]
        if clean_author:
            cmd.extend(['--author', clean_author.split(',')[0]])  # First author only
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0 and result.stdout:
            data = parse_output(result.stdout)
            
            desc = data.get('comments', '')
            if desc and len(desc) > 50:
                clean_desc = clean_html(desc)
                cursor.execute("UPDATE books SET description = ? WHERE id = ?", 
                             (clean_desc, book_id))
                updated += 1
                print(f"  [{updated}] {title[:50]} - got {len(clean_desc)} chars")
                
                # Also add tags if available
                tags = data.get('tags', '')
                if tags:
                    for tag in tags.split(','):
                        tag = tag.strip()
                        if tag:
                            cursor.execute("INSERT OR IGNORE INTO calibre_tags (name) VALUES (?)", (tag,))
                            cursor.execute("SELECT id FROM calibre_tags WHERE name = ?", (tag,))
                            tag_id = cursor.fetchone()[0]
                            cursor.execute("INSERT OR IGNORE INTO book_calibre_tags VALUES (?, ?)", 
                                         (book_id, tag_id))
                
                conn.commit()
        
        time.sleep(0.5)  # Rate limit
        
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {title[:50]}")
    except Exception as e:
        print(f"  Error on {title[:50]}: {e}")

conn.close()
print(f"\nUpdated {updated} out of {len(missing)} books")
