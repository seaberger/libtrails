#!/usr/bin/env python3
"""Try fetching from Goodreads via search (light scraping for personal use)."""
import urllib.request
import urllib.parse
import re
import time
import sqlite3

def search_goodreads(title, author):
    """Search Goodreads and extract description from search results."""
    try:
        query = f"{title} {author}".replace(" ", "+")
        url = f"https://www.goodreads.com/search?q={urllib.parse.quote(query)}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }
        req = urllib.request.Request(url, headers=headers)
        
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8', errors='ignore')
            
            # Extract first book link
            match = re.search(r'href="(/book/show/[^"]+)"', html)
            if match:
                book_url = "https://www.goodreads.com" + match.group(1)
                
                # Fetch book page
                time.sleep(1)  # Be nice
                req2 = urllib.request.Request(book_url, headers=headers)
                with urllib.request.urlopen(req2, timeout=15) as book_response:
                    book_html = book_response.read().decode('utf-8', errors='ignore')
                    
                    # Try to extract description
                    # Look for description in JSON-LD or meta tags
                    desc_match = re.search(r'"description"\s*:\s*"([^"]{50,})"', book_html)
                    if desc_match:
                        desc = desc_match.group(1)
                        desc = desc.replace('\\n', ' ').replace('\\u003c', '<').replace('\\u003e', '>')
                        desc = re.sub(r'<[^>]+>', '', desc)  # Strip HTML
                        return desc[:2000]
                    
                    # Try og:description
                    og_match = re.search(r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"', book_html)
                    if og_match:
                        return og_match.group(1)[:2000]
                        
    except Exception as e:
        print(f"  Error: {e}")
    return None

# Test on a few well-known books
test_books = [
    ("A Single Man", "Christopher Isherwood"),
    ("The Count of Monte Cristo", "Alexandre Dumas"),
    ("Let's Explore Diabetes with Owls", "David Sedaris"),
    ("The Wind-Up Bird Chronicle", "Haruki Murakami"),
]

print("Testing Goodreads search...")
for title, author in test_books:
    print(f"\nSearching: {title}")
    desc = search_goodreads(title, author)
    if desc:
        print(f"  Found: {desc[:100]}...")
    else:
        print("  Not found")
    time.sleep(2)  # Rate limit ourselves
