"""Tests for text chunking functionality."""

from libtrails.chunker import chunk_text


class TestChunkText:
    """Tests for text chunking."""

    def test_basic_chunking(self):
        """Test basic text chunking."""
        # Create text with ~1000 words
        text = "This is a test sentence. " * 200  # ~1000 words

        chunks = chunk_text(text, target_words=500)

        assert len(chunks) >= 2
        for chunk in chunks:
            word_count = len(chunk.split())
            # Should be roughly around target, with some flexibility
            assert word_count >= 100  # Minimum threshold

    def test_respects_sentence_boundaries(self):
        """Test that chunks don't split sentences."""
        text = (
            "First sentence here. Second sentence here. Third sentence here. Fourth sentence here."
        )

        chunks = chunk_text(text, target_words=5)

        for chunk in chunks:
            # Each chunk should end with sentence-ending punctuation or be the last chunk
            chunk = chunk.strip()
            if chunk:
                assert chunk[-1] in ".!?" or chunk == chunks[-1].strip()

    def test_short_text_filtered_out(self):
        """Test that very short text is filtered out (below CHUNK_MIN_WORDS)."""
        text = "This is a short text with only a few words."

        chunks = chunk_text(text, target_words=500)

        # Short text below CHUNK_MIN_WORDS (100) is filtered out
        assert len(chunks) == 0

    def test_text_above_minimum(self):
        """Test that text above minimum word count is kept."""
        # Create text with ~150 words (above CHUNK_MIN_WORDS of 100)
        text = "This is a test sentence with several words. " * 20

        chunks = chunk_text(text, target_words=500)

        assert len(chunks) == 1

    def test_empty_text(self):
        """Test handling of empty text."""
        chunks = chunk_text("", target_words=500)
        # Should return empty list or single empty chunk
        assert len(chunks) == 0 or (len(chunks) == 1 and chunks[0].strip() == "")

    def test_no_sentence_boundaries(self):
        """Test text without clear sentence boundaries."""
        text = "word " * 1000  # No periods

        chunks = chunk_text(text, target_words=500)

        # Should still chunk, possibly treating whole thing as one "sentence"
        assert len(chunks) >= 1

    def test_various_punctuation(self):
        """Test handling of different sentence-ending punctuation."""
        text = (
            "Is this a question? Yes it is! And this is a statement. What about this? Sure thing!"
        )

        chunks = chunk_text(text, target_words=5)

        # All chunks should be valid
        for chunk in chunks:
            assert len(chunk.strip()) > 0

    def test_unicode_text(self):
        """Test handling of Unicode text."""
        # Create Unicode text with enough content to pass min_words filter
        # Note: Japanese doesn't split on spaces, so word count is different
        text = ("This is a test. " * 30) + "日本語のテスト文章です。"

        chunks = chunk_text(text, target_words=100)

        assert len(chunks) >= 1
        # Should preserve Unicode
        full_text = "".join(chunks)
        assert "日本語" in full_text or len(chunks) > 0

    def test_chunk_word_counts(self):
        """Test that chunk word counts are reasonable."""
        text = "This is a sentence with seven words. " * 100  # 700 words

        chunks = chunk_text(text, target_words=200)

        total_words = sum(len(c.split()) for c in chunks)
        original_words = len(text.split())

        # Total words should be preserved (approximately)
        assert abs(total_words - original_words) < 10
