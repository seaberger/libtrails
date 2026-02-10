"""Configuration settings for libtrails."""

import os
from pathlib import Path
from typing import Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# DB variant toggle: set LIBTRAILS_DB=v2 to use ipad_library_v2.db
_db_variant = os.environ.get("LIBTRAILS_DB", "")
IPAD_DB_PATH = (
    DATA_DIR / f"ipad_library_{_db_variant}.db" if _db_variant else DATA_DIR / "ipad_library.db"
)

# User config directory
USER_CONFIG_DIR = Path.home() / ".libtrails"
USER_CONFIG_FILE = USER_CONFIG_DIR / "config.yaml"


def get_user_config() -> dict:
    """Load user configuration from ~/.libtrails/config.yaml"""
    if not USER_CONFIG_FILE.exists():
        return {}

    try:
        import yaml

        with open(USER_CONFIG_FILE) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def save_user_config(config: dict):
    """Save user configuration to ~/.libtrails/config.yaml"""
    import yaml

    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(USER_CONFIG_FILE, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def get_ipad_url() -> Optional[str]:
    """Get default iPad URL from user config."""
    config = get_user_config()
    return config.get("ipad", {}).get("default_url")


def set_ipad_url(url: str):
    """Save iPad URL to user config."""
    config = get_user_config()
    if "ipad" not in config:
        config["ipad"] = {}
    config["ipad"]["default_url"] = url
    save_user_config(config)


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
DEDUP_HIGH_CONFIDENCE_THRESHOLD = 0.95  # Merge unconditionally above this

# Graph/clustering settings
EMBEDDING_EDGE_THRESHOLD = 0.7  # For full mode only
COOCCURRENCE_MIN_COUNT = 2  # Lowered for better connectivity
PMI_MIN_THRESHOLD = 1.0

# Clustering defaults (optimized for ~300-400 coherent clusters)
CLUSTER_MODE = "knn"  # "cooccurrence", "knn", or "full"
CLUSTER_KNN_K = 10  # k neighbors for knn mode
CLUSTER_PARTITION_TYPE = "cpm"  # "modularity", "surprise", or "cpm"
CLUSTER_RESOLUTION = 0.001  # Resolution for CPM (lower = fewer clusters)

# KNN graph: minimum cosine similarity for embedding edges
KNN_MIN_SIMILARITY = 0.65

# Topic extraction models — override with env vars to use Gemini API
# Gemini models use "gemini/" prefix, e.g. "gemini/gemini-3-flash-preview"
THEME_MODEL = os.environ.get("LIBTRAILS_THEME_MODEL", "gemma3:27b")
CHUNK_MODEL = os.environ.get("LIBTRAILS_CHUNK_MODEL", "gemma3:4b")
BATCH_SIZE = int(os.environ.get("LIBTRAILS_BATCH_SIZE", "5"))

# Ollama context window — both gemma3 models support 128K tokens
# Default of 2048 is far too small for batched extraction
OLLAMA_NUM_CTX = 8192

# Generic topic stoplist — filtered during normalization
TOPIC_STOPLIST = frozenset(
    {
        "manipulation",
        "relationships",
        "technology",
        "society",
        "nature",
        "life",
        "death",
        "love",
        "power",
        "time",
        "people",
        "world",
        "change",
        "future",
        "communication",
        "conflict",
        "loss",
        "survival",
        "identity",
        "freedom",
        "control",
        "trust",
        "fear",
        "growth",
        "knowledge",
        "science",
        "culture",
        "politics",
        "art",
        "history",
        "topics",  # LM Studio 4b echoes the prompt's "Topics:" label
    }
)

# Universe visualization
UNIVERSE_JSON_PATH = DATA_DIR / "universe_coords.json"
