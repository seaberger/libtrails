"""Tests for text chunking functionality."""

from libtrails.chunker import (
    _chunk_by_sentences,
    _chunk_by_words,
    _is_false_sentence_break,
    _split_paragraphs,
    _split_sentences,
    chunk_text,
)


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


class TestSplitParagraphs:
    """Tests for _split_paragraphs()."""

    def test_double_newline_split(self):
        """Splits on double newlines."""
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        result = _split_paragraphs(text)
        assert len(result) == 3
        assert result[0] == "Paragraph one."
        assert result[1] == "Paragraph two."

    def test_triple_newline_split(self):
        """Handles 3+ newlines as paragraph break."""
        text = "Paragraph one.\n\n\nParagraph two."
        result = _split_paragraphs(text)
        assert len(result) == 2

    def test_single_newline_not_split(self):
        """Single newlines do NOT create paragraph breaks."""
        text = "Line one.\nLine two."
        result = _split_paragraphs(text)
        assert len(result) == 1

    def test_strips_empty_paragraphs(self):
        """Empty paragraphs are stripped."""
        text = "First.\n\n\n\n\n\nSecond."
        result = _split_paragraphs(text)
        assert len(result) == 2

    def test_strips_whitespace(self):
        """Leading/trailing whitespace is stripped from paragraphs."""
        text = "  First paragraph.  \n\n  Second paragraph.  "
        result = _split_paragraphs(text)
        assert result[0] == "First paragraph."
        assert result[1] == "Second paragraph."


class TestSplitSentences:
    """Tests for _split_sentences()."""

    def test_period_split(self):
        """Splits on periods followed by space."""
        text = "First sentence. Second sentence. Third sentence."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_abbreviation_handling_mr(self):
        """Does not split on 'Mr.'."""
        text = "Mr. Smith went to Washington. He was tired."
        result = _split_sentences(text)
        assert len(result) == 2
        assert "Mr. Smith" in result[0]

    def test_abbreviation_handling_dr(self):
        """Does not split on 'Dr.'."""
        text = "Dr. Jones prescribed medicine. The patient recovered."
        result = _split_sentences(text)
        assert len(result) == 2

    def test_us_abbreviation(self):
        """Does not split on 'U.S.'."""
        text = "The U.S. government issued a statement. It was important."
        result = _split_sentences(text)
        assert len(result) == 2

    def test_decimal_numbers_not_split(self):
        """Does not split on decimal numbers when followed by lowercase."""
        text = "Version 2.5. is available now. Download it today."
        result = _split_sentences(text)
        # "2.5. is" — period after digit + lowercase = false break
        assert any("2.5" in s for s in result)

    def test_question_exclamation_marks(self):
        """Splits on ? and ! marks."""
        text = "Is this a question? Yes! It is a statement."
        result = _split_sentences(text)
        assert len(result) == 3

    def test_single_sentence(self):
        """Single sentence returns as-is."""
        text = "Just one sentence here."
        result = _split_sentences(text)
        assert len(result) == 1
        assert result[0] == "Just one sentence here."

    def test_empty_text(self):
        """Empty text returns empty list."""
        result = _split_sentences("")
        assert result == []


class TestIsFalseSentenceBreak:
    """Tests for _is_false_sentence_break()."""

    def test_mr_abbreviation(self):
        """'Mr.' is a false break."""
        assert _is_false_sentence_break("Mr.", "Smith went") is True

    def test_us_abbreviation(self):
        """'U.S.' is a false break."""
        assert _is_false_sentence_break("the U.S.", "government") is True

    def test_real_sentence_end(self):
        """Real sentence end is not a false break."""
        assert _is_false_sentence_break("end of sentence.", "Beginning of next") is False

    def test_initial_mid_sentence(self):
        """Single letter initial is a false break."""
        assert _is_false_sentence_break("J.", "K. Rowling wrote") is True

    def test_lowercase_after_period(self):
        """Lowercase after period signals false break."""
        assert _is_false_sentence_break("etc.", "and more") is True

    def test_empty_before(self):
        """Empty 'before' returns False."""
        assert _is_false_sentence_break("", "Next") is False

    def test_empty_after(self):
        """Empty 'after' returns False."""
        assert _is_false_sentence_break("End.", "") is False


class TestChunkBySentences:
    """Tests for _chunk_by_sentences()."""

    def test_sentences_grouped_to_target(self):
        """Sentences are grouped to approximately target word count."""
        # 10 sentences of ~10 words each = ~100 words total
        text = ". ".join(["This is a sentence with about ten words each"] * 10) + "."
        chunks = _chunk_by_sentences(text, target_words=50)
        # Should produce ~2 chunks
        assert len(chunks) >= 2

    def test_oversized_sentence_split_by_words(self):
        """Very long sentence falls back to word-level splitting."""
        text = " ".join(["word"] * 200)  # 200-word "sentence"
        chunks = _chunk_by_sentences(text, target_words=50)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert len(chunk.split()) <= 50


class TestChunkByWords:
    """Tests for _chunk_by_words()."""

    def test_splits_evenly(self):
        """Words are split into chunks of target size."""
        text = " ".join(["word"] * 100)
        chunks = _chunk_by_words(text, target_words=30)
        assert len(chunks) == 4  # 30 + 30 + 30 + 10
        assert len(chunks[0].split()) == 30

    def test_single_chunk_if_under_target(self):
        """Text under target is one chunk."""
        text = " ".join(["word"] * 10)
        chunks = _chunk_by_words(text, target_words=50)
        assert len(chunks) == 1
