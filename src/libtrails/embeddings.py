"""Embedding generation using sentence-transformers."""

import os
import logging
import numpy as np
from pathlib import Path
from typing import Optional

from .config import PROJECT_ROOT

# Suppress verbose HuggingFace/transformers logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

# Model configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Top MTEB performer for STS, 384 dims
EMBEDDING_DIMENSION = 384
MODEL_CACHE_DIR = PROJECT_ROOT / "models"

# Lazy load the model to avoid import-time overhead
_model = None


def get_model():
    """
    Get the sentence transformer model, loading it lazily.

    Uses BGE-small-en-v1.5 which excels at semantic textual similarity.
    Model is cached locally in the project's models/ directory.
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        # Ensure cache directory exists
        MODEL_CACHE_DIR.mkdir(exist_ok=True)

        # Check if model is already cached locally
        local_model_path = MODEL_CACHE_DIR / MODEL_NAME.replace("/", "_")

        if local_model_path.exists():
            # Load from local cache
            _model = SentenceTransformer(str(local_model_path))
        else:
            # Download and cache locally
            _model = SentenceTransformer(MODEL_NAME)
            _model.save(str(local_model_path))

    return _model


def embed_text(text: str) -> np.ndarray:
    """Generate an embedding for a single text string."""
    model = get_model()
    return model.encode(text, normalize_embeddings=True)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding

    Returns:
        np.ndarray of shape (len(texts), 384)
    """
    model = get_model()
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100
    )


def embedding_to_bytes(embedding: np.ndarray) -> bytes:
    """Convert a numpy embedding to bytes for SQLite storage."""
    return embedding.astype(np.float32).tobytes()


def bytes_to_embedding(data: bytes) -> np.ndarray:
    """Convert bytes from SQLite back to numpy embedding."""
    return np.frombuffer(data, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized embeddings."""
    return float(np.dot(a, b))


def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity matrix.

    For normalized embeddings, this is just the dot product.

    Args:
        embeddings: np.ndarray of shape (n, dim)

    Returns:
        np.ndarray of shape (n, n) with similarity scores
    """
    return np.dot(embeddings, embeddings.T)


def find_similar(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    top_k: int = 10
) -> list[tuple[int, float]]:
    """
    Find the most similar embeddings to a query.

    Args:
        query_embedding: The query embedding
        embeddings: Matrix of embeddings to search
        top_k: Number of results to return

    Returns:
        List of (index, similarity_score) tuples
    """
    similarities = np.dot(embeddings, query_embedding)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(int(idx), float(similarities[idx])) for idx in top_indices]


def get_embedding_dimension() -> int:
    """Return the embedding dimension (384 for BGE-small-en-v1.5)."""
    return EMBEDDING_DIMENSION


def get_model_info() -> dict:
    """Get information about the current embedding model."""
    return {
        "name": MODEL_NAME,
        "dimension": EMBEDDING_DIMENSION,
        "cache_dir": str(MODEL_CACHE_DIR),
        "cached_locally": (MODEL_CACHE_DIR / MODEL_NAME.replace("/", "_")).exists(),
    }
