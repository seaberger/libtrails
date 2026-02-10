"""Tests for topic extraction functionality."""

from unittest.mock import MagicMock, patch

from libtrails.topic_extractor import (
    _parse_topics,
    extract_topics,
    extract_topics_batch,
    normalize_topic,
)


class TestNormalizeTopic:
    """Tests for topic normalization."""

    def test_lowercase(self):
        assert normalize_topic("Philosophy") == "philosophy"
        assert normalize_topic("SCIENCE FICTION") == "science fiction"

    def test_strip_whitespace(self):
        assert normalize_topic("  topic  ") == "topic"
        assert normalize_topic("\ttopic\n") == "topic"

    def test_replace_underscores(self):
        assert normalize_topic("science_fiction") == "science fiction"
        assert normalize_topic("self_discovery_journey") == "self discovery journey"

    def test_collapse_spaces(self):
        assert normalize_topic("too   many   spaces") == "too many spaces"

    def test_combined_normalization(self):
        assert normalize_topic("  Self_Discovery  ") == "self discovery"
        # Double underscore becomes double space, then collapsed
        result = normalize_topic("SCIENCE_FICTION__ADVENTURE")
        assert "science fiction" in result
        assert "adventure" in result


class TestParseTopics:
    """Tests for parsing LLM output."""

    def test_parse_json_array(self):
        output = '["Topic One", "Topic Two", "Topic Three"]'
        topics = _parse_topics(output)
        assert topics == ["Topic One", "Topic Two", "Topic Three"]

    def test_parse_json_with_prefix(self):
        output = 'Here are the topics:\n["Topic One", "Topic Two"]'
        topics = _parse_topics(output)
        assert topics == ["Topic One", "Topic Two"]

    def test_parse_json_with_suffix(self):
        output = '["Topic One", "Topic Two"]\n\nThese are the main topics.'
        topics = _parse_topics(output)
        assert topics == ["Topic One", "Topic Two"]

    def test_parse_quoted_strings_fallback(self):
        output = 'Topics: "Topic One", "Topic Two", "Topic Three"'
        topics = _parse_topics(output)
        assert "Topic One" in topics
        assert "Topic Two" in topics

    def test_parse_empty_output(self):
        topics = _parse_topics("")
        assert topics == []

    def test_parse_invalid_json(self):
        output = "This is not valid JSON at all"
        topics = _parse_topics(output)
        # Should handle gracefully
        assert isinstance(topics, list)

    def test_parse_nested_json(self):
        output = '{"topics": ["Topic One", "Topic Two"]}'
        topics = _parse_topics(output)
        # May or may not extract - just shouldn't crash
        assert isinstance(topics, list)


class TestExtractTopics:
    """Tests for topic extraction with mocked Ollama."""

    @patch("libtrails.topic_extractor._get_client")
    def test_extract_topics_success(self, mock_get_client):
        """Test successful topic extraction."""
        # Mock the HTTP response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '["Philosophy", "Ethics", "Morality"]'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        text = "This is a philosophical text about ethics and morality."
        topics = extract_topics(text, model="gemma3:4b")

        assert len(topics) == 3
        assert "Philosophy" in topics
        assert "Ethics" in topics

    @patch("libtrails.topic_extractor._get_client")
    def test_extract_topics_timeout(self, mock_get_client):
        """Test handling of timeout."""
        import httpx

        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_get_client.return_value = mock_client

        topics = extract_topics("Some text", model="gemma3:4b")

        assert topics == []

    @patch("libtrails.topic_extractor._get_client")
    def test_extract_topics_error(self, mock_get_client):
        """Test handling of HTTP errors."""
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection error")
        mock_get_client.return_value = mock_client

        topics = extract_topics("Some text", model="gemma3:4b")

        assert topics == []


class TestExtractTopicsBatch:
    """Tests for batch topic extraction."""

    @patch("libtrails.topic_extractor.extract_topics")
    def test_batch_extraction(self, mock_extract):
        """Test batch extraction calls extract_topics for each chunk."""
        mock_extract.return_value = ["Topic A", "Topic B"]

        chunks = ["Chunk 1 text", "Chunk 2 text", "Chunk 3 text"]
        results = extract_topics_batch(chunks, model="gemma3:4b")

        assert len(results) == 3
        assert mock_extract.call_count == 3

    @patch("libtrails.topic_extractor.extract_topics")
    def test_batch_preserves_order(self, mock_extract):
        """Test that batch results are in correct order."""
        # Return different topics for each chunk
        mock_extract.side_effect = [
            ["Topic 1"],
            ["Topic 2"],
            ["Topic 3"],
        ]

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        results = extract_topics_batch(chunks, model="gemma3:4b")

        assert results[0] == ["topic 1"]
        assert results[1] == ["topic 2"]
        assert results[2] == ["topic 3"]

    @patch("libtrails.topic_extractor.extract_topics")
    def test_batch_progress_callback(self, mock_extract):
        """Test that progress callback is called."""
        mock_extract.return_value = ["Topic"]

        progress_calls = []

        def callback(completed, total):
            progress_calls.append((completed, total))

        chunks = ["Chunk 1", "Chunk 2"]
        extract_topics_batch(chunks, model="gemma3:4b", progress_callback=callback)

        assert len(progress_calls) == 2
        assert progress_calls[-1] == (2, 2)

    @patch("libtrails.topic_extractor.extract_topics")
    def test_batch_handles_failures(self, mock_extract):
        """Test that batch handles individual failures gracefully."""
        # Second chunk fails
        mock_extract.side_effect = [
            ["Topic 1"],
            [],  # Failed/empty
            ["Topic 3"],
        ]

        chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        results = extract_topics_batch(chunks, model="gemma3:4b")

        assert len(results) == 3
        assert results[0] == ["topic 1"]
        assert results[1] == []
        assert results[2] == ["topic 3"]
