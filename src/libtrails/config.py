"""Configuration settings for libtrails."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IPAD_DB_PATH = DATA_DIR / "ipad_library.db"

# Calibre library (read-only)
CALIBRE_LIBRARY_PATH = Path.home() / "Calibre_Main_Library"
CALIBRE_DB_PATH = CALIBRE_LIBRARY_PATH / "metadata.db"

# LLM settings
DEFAULT_MODEL = "gemma3:27b"
OLLAMA_HOST = "http://localhost:11434"

# Chunking settings
CHUNK_TARGET_WORDS = 500
CHUNK_MIN_WORDS = 100

# Topic extraction
TOPICS_PER_CHUNK = 5

# Embedding settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, fast
EMBEDDING_DIMENSION = 384

# Deduplication settings
DEDUP_SIMILARITY_THRESHOLD = 0.85

# Graph/clustering settings
EMBEDDING_EDGE_THRESHOLD = 0.5
COOCCURRENCE_MIN_COUNT = 2
PMI_MIN_THRESHOLD = 0.0
