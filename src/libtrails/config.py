"""Configuration settings for libtrails."""

from pathlib import Path
from typing import Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
IPAD_DB_PATH = DATA_DIR / "ipad_library.db"

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
    with open(USER_CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def get_ipad_url() -> Optional[str]:
    """Get default iPad URL from user config."""
    config = get_user_config()
    return config.get('ipad', {}).get('default_url')


def set_ipad_url(url: str):
    """Save iPad URL to user config."""
    config = get_user_config()
    if 'ipad' not in config:
        config['ipad'] = {}
    config['ipad']['default_url'] = url
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

# Graph/clustering settings
EMBEDDING_EDGE_THRESHOLD = 0.5
COOCCURRENCE_MIN_COUNT = 2
PMI_MIN_THRESHOLD = 0.0
