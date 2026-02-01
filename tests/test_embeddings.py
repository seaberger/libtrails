"""Tests for embedding generation functionality."""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from libtrails.embeddings import (
    embed_text,
    embed_texts,
    embedding_to_bytes,
    bytes_to_embedding,
    cosine_similarity,
    cosine_similarity_matrix,
    get_embedding_dimension,
    get_model_info,
)


class TestEmbeddingConversion:
    """Tests for embedding serialization/deserialization."""

    def test_embedding_to_bytes(self):
        """Test converting numpy array to bytes."""
        embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        result = embedding_to_bytes(embedding)

        assert isinstance(result, bytes)
        assert len(result) == 3 * 4  # 3 floats * 4 bytes each

    def test_bytes_to_embedding(self):
        """Test converting bytes back to numpy array."""
        original = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        as_bytes = embedding_to_bytes(original)
        result = bytes_to_embedding(as_bytes)

        np.testing.assert_array_almost_equal(result, original)

    def test_roundtrip_conversion(self):
        """Test that conversion is lossless."""
        original = np.random.randn(384).astype(np.float32)
        as_bytes = embedding_to_bytes(original)
        recovered = bytes_to_embedding(as_bytes)

        np.testing.assert_array_almost_equal(recovered, original)


class TestCosineSimilarity:
    """Tests for cosine similarity calculations.

    Note: cosine_similarity expects normalized vectors (|v| = 1).
    """

    def test_identical_vectors(self):
        """Test similarity of identical normalized vectors is 1.0."""
        vec = np.array([1.0, 0.0, 0.0])  # Already normalized
        result = cosine_similarity(vec, vec)

        assert abs(result - 1.0) < 0.0001

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors is 0.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        result = cosine_similarity(vec1, vec2)

        assert abs(result) < 0.0001

    def test_opposite_vectors(self):
        """Test similarity of opposite normalized vectors is -1.0."""
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([-1.0, 0.0, 0.0])
        result = cosine_similarity(vec1, vec2)

        assert abs(result + 1.0) < 0.0001

    def test_similarity_matrix_shape(self):
        """Test similarity matrix has correct shape."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        result = cosine_similarity_matrix(embeddings)

        assert result.shape == (5, 5)

    def test_similarity_matrix_diagonal(self):
        """Test similarity matrix diagonal is all 1s."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        # Normalize for proper cosine similarity
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        result = cosine_similarity_matrix(embeddings)

        np.testing.assert_array_almost_equal(np.diag(result), np.ones(5))

    def test_similarity_matrix_symmetric(self):
        """Test similarity matrix is symmetric."""
        embeddings = np.random.randn(5, 384).astype(np.float32)
        result = cosine_similarity_matrix(embeddings)

        np.testing.assert_array_almost_equal(result, result.T)


class TestEmbedText:
    """Tests for text embedding with mocked model."""

    @patch('libtrails.embeddings.get_model')
    def test_embed_single_text(self, mock_get_model):
        """Test embedding a single text."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1, 0.2, 0.3]])
        mock_get_model.return_value = mock_model

        result = embed_text("test text")

        assert isinstance(result, np.ndarray)
        mock_model.encode.assert_called_once()

    @patch('libtrails.embeddings.get_model')
    def test_embed_multiple_texts(self, mock_get_model):
        """Test embedding multiple texts."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ])
        mock_get_model.return_value = mock_model

        result = embed_texts(["text one", "text two"])

        assert result.shape[0] == 2
        mock_model.encode.assert_called_once()


class TestModelInfo:
    """Tests for model information retrieval."""

    @patch('libtrails.embeddings.get_model')
    @patch('libtrails.embeddings.MODEL_CACHE_DIR')
    def test_get_model_info(self, mock_cache_dir, mock_get_model):
        """Test retrieving model info."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_get_model.return_value = mock_model
        mock_cache_dir.exists.return_value = True

        result = get_model_info()

        assert 'name' in result
        assert 'dimension' in result
        assert result['dimension'] == 384

    @patch('libtrails.embeddings._model', None)
    def test_get_embedding_dimension(self):
        """Test getting embedding dimension."""
        # Should return the configured dimension
        dim = get_embedding_dimension()

        assert isinstance(dim, int)
        assert dim > 0
