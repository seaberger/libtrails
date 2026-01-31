# Calibre Plugin Development Guide
## Essential Information for RAG Search Plugin

### Plugin Structure

```
calibre-rag-search/
├── __init__.py           # Plugin metadata and entry point
├── main.py              # Main plugin logic
├── config.py            # Configuration handling
├── search_engine.py     # Core search implementation
├── indexer.py          # Book indexing logic
├── gemini_client.py    # Gemini API integration
├── vector_store.py     # Vector database management
├── ui.py               # Qt-based user interface
├── utils.py            # Helper functions
├── plugin-import-name-rag_search.txt  # Empty file (enables multi-file)
├── resources/
│   ├── icon.png        # Plugin icon
│   └── styles.css      # UI styling
└── sqlite_vec/
    └── vec0.so         # SQLite extension (platform specific)
```

### Key Calibre APIs

#### Database Access
```python
from calibre.library import db

class SearchPlugin:
    def get_database(self):
        # Get current library database
        db = self.gui.current_db
        return db.new_api  # Use new API for better performance
    
    def get_all_books(self):
        db_api = self.get_database()
        book_ids = db_api.all_book_ids()
        for book_id in book_ids:
            metadata = db_api.get_metadata(book_id)
            yield {
                'id': book_id,
                'title': metadata.title,
                'authors': metadata.authors,
                'tags': metadata.tags,
                'comments': metadata.comments,  # Description
                'series': metadata.series,
                'publisher': metadata.publisher,
                'languages': metadata.languages,
                'date': metadata.timestamp,
                'rating': metadata.rating,
                'identifiers': metadata.identifiers  # ISBN, etc.
            }
```

#### Plugin Base Classes
```python
from calibre.customize import InterfaceActionBase

class RAGSearchPlugin(InterfaceActionBase):
    name = 'RAG Search'
    description = 'Advanced semantic search using RAG'
    supported_platforms = ['windows', 'osx', 'linux']
    author = 'Your Name'
    version = (1, 0, 0)
    minimum_calibre_version = (8, 0, 0)
    
    actual_plugin = 'calibre_plugins.rag_search.ui:InterfacePlugin'
    
    def is_customizable(self):
        return True
    
    def config_widget(self):
        from calibre_plugins.rag_search.config import ConfigWidget
        return ConfigWidget()
```

#### GUI Integration
```python
from calibre.gui2.actions import InterfaceAction
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLineEdit
from PyQt6.QtCore import Qt

class InterfacePlugin(InterfaceAction):
    name = 'RAG Search'
    action_spec = ('RAG Search', None, 'Advanced semantic search', 'Ctrl+Shift+S')
    
    def genesis(self):
        # Called once when plugin loads
        icon = get_icons('icon.png')
        self.qaction.setIcon(icon)
        self.qaction.triggered.connect(self.show_dialog)
        
        # Initialize search engine
        from calibre_plugins.rag_search.search_engine import SearchEngine
        self.search_engine = SearchEngine(self.gui)
    
    def show_dialog(self):
        # Show search dialog
        d = SearchDialog(self.gui, self.search_engine)
        d.exec()
```

#### Configuration Storage
```python
from calibre.utils.config import JSONConfig

prefs = JSONConfig('plugins/rag_search')

# Set defaults
prefs.defaults['gemini_api_key'] = ''
prefs.defaults['embedding_model'] = 'all-MiniLM-L6-v2'
prefs.defaults['max_results'] = 50

# Access config
api_key = prefs['gemini_api_key']

# Update config
prefs['gemini_api_key'] = 'new-key'
```

#### Progress Dialogs
```python
from calibre.gui2.progress_indicator import ProgressIndicator

def index_books(self, books):
    with ProgressIndicator(self.gui, _('Indexing books...'), 
                          max=len(books)) as pi:
        for i, book in enumerate(books):
            if pi.canceled:
                break
            self.index_single_book(book)
            pi.increment()
            if i % 10 == 0:
                pi.set_msg(f'Indexed {i}/{len(books)} books')
```

#### Background Jobs
```python
from calibre.gui2.threaded_jobs import ThreadedJob

def start_indexing(self):
    job = ThreadedJob('rag_search_index', 
                     _('Building search index'),
                     self.do_index_work,
                     self.index_complete,
                     callback=self.index_progress)
    self.gui.job_manager.run_threaded_job(job)

def do_index_work(self, job):
    # Runs in background thread
    books = list(self.get_all_books())
    for i, book in enumerate(books):
        self.index_book(book)
        job.percent = int((i / len(books)) * 100)
        
def index_complete(self, job):
    # Runs in GUI thread when done
    if job.failed:
        self.gui.status_bar.show_message(
            f'Indexing failed: {job.exception}', 5000)
```

### SQLite Integration

#### Loading sqlite-vec
```python
import sqlite3
import os
from calibre.constants import iswindows, ismacos

def load_sqlite_vec(conn):
    # Determine platform-specific extension
    if iswindows:
        ext_path = 'vec0.dll'
    elif ismacos:
        ext_path = 'vec0.dylib'
    else:
        ext_path = 'vec0.so'
    
    # Load from plugin resources
    plugin_path = os.path.dirname(os.path.abspath(__file__))
    vec_path = os.path.join(plugin_path, 'sqlite_vec', ext_path)
    
    conn.enable_load_extension(True)
    conn.load_extension(vec_path)
    conn.enable_load_extension(False)
```

#### Creating Vector Tables
```python
def setup_database(self):
    # Use separate database for vectors
    db_path = os.path.join(
        self.gui.current_db.library_path, 
        'rag_search_index.db'
    )
    conn = sqlite3.connect(db_path)
    load_sqlite_vec(conn)
    
    # Create tables
    conn.execute('''
        CREATE TABLE IF NOT EXISTS book_metadata (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            authors TEXT,
            tags TEXT,
            description TEXT,
            indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS book_vectors
        USING vec0(
            book_id INTEGER PRIMARY KEY,
            title_embedding FLOAT[384],
            description_embedding FLOAT[384],
            combined_embedding FLOAT[384]
        )
    ''')
    
    # Create FTS5 for keyword search
    conn.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS book_fts
        USING fts5(
            book_id, title, authors, tags, description,
            content=book_metadata
        )
    ''')
```

### Platform Considerations

#### File Paths
```python
from calibre.utils.config_base import config_dir

# Plugin config directory
plugin_dir = os.path.join(config_dir, 'plugins', 'rag_search')
os.makedirs(plugin_dir, exist_ok=True)

# Model cache
model_cache = os.path.join(plugin_dir, 'models')

# Index database
index_db = os.path.join(plugin_dir, 'search_index.db')
```

#### Memory Management
```python
import gc

def process_large_library(self):
    batch_size = 100
    books = self.get_all_books()
    
    for i in range(0, len(books), batch_size):
        batch = books[i:i+batch_size]
        self.process_batch(batch)
        
        # Force garbage collection every 1000 books
        if i % 1000 == 0:
            gc.collect()
```

### Error Handling

```python
from calibre import prints
from calibre.gui2 import error_dialog

def safe_search(self, query):
    try:
        return self.search_engine.search(query)
    except Exception as e:
        # Log to console
        prints(f'RAG Search error: {e}')
        
        # Show user-friendly error
        error_dialog(
            self.gui,
            _('Search Failed'),
            _('An error occurred during search: {}').format(str(e)),
            show=True
        )
        
        # Fallback to basic search
        return self.fallback_search(query)
```

### Testing & Debugging

#### Debug Mode
```python
# Run Calibre in debug mode
# calibre-debug -g

# In plugin code
from calibre.constants import DEBUG

if DEBUG:
    print(f'[RAG Search] Processing book: {book.title}')
```

#### Plugin Development Workflow
```bash
# 1. Create plugin structure
mkdir calibre-rag-search
cd calibre-rag-search

# 2. Develop plugin files
# ... edit files ...

# 3. Create plugin ZIP
zip -r ../rag_search.zip *

# 4. Install in Calibre
calibre-customize -b .

# 5. Test in debug mode
calibre-debug -g

# 6. Reload plugin (after changes)
calibre-customize -r "RAG Search"
calibre-customize -b .
```

### Performance Tips

1. **Use new_api** - Much faster than legacy API
2. **Batch operations** - Process books in groups
3. **Lazy loading** - Don't load all metadata at once
4. **Caching** - Cache frequently accessed data
5. **Threading** - Use ThreadedJob for long operations
6. **Generators** - Use yield for large datasets
7. **SQLite optimizations** - Use transactions, prepared statements

### Common Pitfalls

1. **GUI thread blocking** - Always use background jobs
2. **Memory leaks** - Clean up large objects
3. **Path issues** - Use calibre's path utilities
4. **Database locks** - Close connections properly
5. **Plugin conflicts** - Use unique namespaces
6. **Version compatibility** - Test with min version

### Resources

- **Official Docs**: https://manual.calibre-ebook.com/creating_plugins.html
- **API Reference**: https://manual.calibre-ebook.com/plugins.html
- **DB API**: https://manual.calibre-ebook.com/db_api.html
- **MobileRead Forum**: https://www.mobileread.com/forums/forumdisplay.php?f=237
- **Example Plugins**: https://github.com/kiwidude68/calibre_plugins