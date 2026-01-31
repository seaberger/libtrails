"""Embedding generation using sentence-transformers."""

import numpy as np
from typing import Optional
from functools import lru_cache

# Lazy load the model to avoid import-time overhead
_model = None


def get_model():
    """Get the sentence transformer model, loading it lazily."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # all-MiniLM-L6-v2: 384 dimensions, fast, good quality
        _model = SentenceTransformer('all-MiniLM-L6-v2')
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
    """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
    return 384
