"""Embedding generation using ONNX Runtime (lightweight) or sentence-transformers (full).

ONNX Runtime is preferred for server deployment — it avoids loading PyTorch (~300 MB)
and keeps the uvicorn process under 200 MB RSS. sentence-transformers is used as a
fallback for local development and bulk indexing where PyTorch is already installed.
"""

import logging
import os

import numpy as np

from .config import PROJECT_ROOT

# Suppress verbose HuggingFace/transformers logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Top MTEB performer for STS, 384 dims
EMBEDDING_DIMENSION = 384
MODEL_CACHE_DIR = PROJECT_ROOT / "models"
ONNX_MODEL_DIR = MODEL_CACHE_DIR / "bge-small-onnx"

# Lazy-loaded backends
_onnx_session = None
_onnx_tokenizer = None
_st_model = None
_backend = None  # "onnx" or "sentence-transformers"


def _try_load_onnx() -> bool:
    """Attempt to load the ONNX model and tokenizer. Returns True on success."""
    global _onnx_session, _onnx_tokenizer, _backend

    onnx_path = ONNX_MODEL_DIR / "model.onnx"
    tokenizer_path = ONNX_MODEL_DIR / "tokenizer.json"

    if not onnx_path.exists() or not tokenizer_path.exists():
        return False

    try:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        _onnx_session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        _onnx_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        _onnx_tokenizer.enable_truncation(max_length=512)
        _onnx_tokenizer.enable_padding()
        _backend = "onnx"
        logger.info("Loaded ONNX embedding model (%s)", onnx_path)
        return True
    except ImportError:
        _onnx_session = None
        _onnx_tokenizer = None
        logger.debug(
            "onnxruntime or tokenizers not installed, falling back to sentence-transformers"
        )
        return False
    except Exception:
        _onnx_session = None
        _onnx_tokenizer = None
        logger.warning(
            "Failed to load ONNX model, falling back to sentence-transformers", exc_info=True
        )
        return False


def _ensure_st_loaded():
    """Load the sentence-transformers model without changing the active backend."""
    global _st_model

    if _st_model is not None:
        return

    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_DIR.mkdir(exist_ok=True)
    local_model_path = MODEL_CACHE_DIR / MODEL_NAME.replace("/", "_")

    if local_model_path.exists():
        _st_model = SentenceTransformer(str(local_model_path))
    else:
        _st_model = SentenceTransformer(MODEL_NAME)
        _st_model.save(str(local_model_path))

    logger.info("Loaded sentence-transformers embedding model")


def _ensure_backend():
    """Ensure an embedding backend is loaded (ONNX preferred, ST fallback)."""
    global _backend
    if _backend is not None:
        return

    if not _try_load_onnx():
        _ensure_st_loaded()
        _backend = "sentence-transformers"


def _onnx_encode(texts: list[str]) -> np.ndarray:
    """Encode texts using ONNX Runtime. Returns L2-normalized embeddings."""
    encodings = _onnx_tokenizer.encode_batch(texts)

    input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64)

    outputs = _onnx_session.run(
        None,
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        },
    )

    # CLS pooling (index 0) — matches BGE model's default pooling strategy
    cls_embeddings = outputs[0][:, 0, :]

    # L2 normalize
    norms = np.linalg.norm(cls_embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)  # avoid division by zero
    return cls_embeddings / norms


def get_model():
    """Get the sentence-transformers model for bulk operations.

    This loads sentence-transformers without overriding the active backend,
    so embed_text()/embed_texts() continue using ONNX if it was loaded first.
    """
    _ensure_st_loaded()
    return _st_model


def embed_text(text: str) -> np.ndarray:
    """Generate an embedding for a single text string."""
    _ensure_backend()

    if _backend == "onnx":
        return _onnx_encode([text])[0]

    return _st_model.encode(text, normalize_embeddings=True)


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of text strings to embed
        batch_size: Batch size for encoding

    Returns:
        np.ndarray of shape (len(texts), 384)
    """
    if len(texts) == 0:
        return np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)

    _ensure_backend()

    if _backend == "onnx":
        # Process in batches to avoid OOM on large inputs
        if len(texts) <= batch_size:
            return _onnx_encode(texts)

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.append(_onnx_encode(batch))
        return np.vstack(all_embeddings)

    return _st_model.encode(
        texts, normalize_embeddings=True, batch_size=batch_size, show_progress_bar=len(texts) > 100
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
    """Compute pairwise cosine similarity matrix.

    For normalized embeddings, this is just the dot product.

    Args:
        embeddings: np.ndarray of shape (n, dim)

    Returns:
        np.ndarray of shape (n, n) with similarity scores
    """
    return np.dot(embeddings, embeddings.T)


def find_similar(
    query_embedding: np.ndarray, embeddings: np.ndarray, top_k: int = 10
) -> list[tuple[int, float]]:
    """Find the most similar embeddings to a query.

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
    _ensure_backend()
    return {
        "name": MODEL_NAME,
        "dimension": EMBEDDING_DIMENSION,
        "cache_dir": str(MODEL_CACHE_DIR),
        "backend": _backend,
        "onnx_available": (ONNX_MODEL_DIR / "model.onnx").exists(),
        "cached_locally": (MODEL_CACHE_DIR / MODEL_NAME.replace("/", "_")).exists(),
    }
