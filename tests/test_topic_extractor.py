"""Tests for topic extraction functionality."""

import json
import subprocess
from unittest.mock import MagicMock, patch

import httpx

from libtrails.topic_extractor import (
    ContentFilterError,
    _build_batch_schema,
    _extract_json_from_text,
    _filter_topics,
    _is_litellm_model,
    _parse_topics,
    _unwrap_topic,
    check_ollama_available,
    clean_calibre_tags,
    extract_book_themes,
    extract_topics,
    extract_topics_batch,
    extract_topics_batched,
    extract_topics_single_optimized,
    get_available_models,
    normalize_topic,
    strip_html,
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


# ---- Pure function tests (no mocks needed) ----


class TestExtractJsonFromText:
    """Tests for _extract_json_from_text()."""

    def test_valid_json_passthrough(self):
        """Already-valid JSON is returned unchanged."""
        text = '{"topics": ["a", "b"]}'
        assert _extract_json_from_text(text) == text

    def test_valid_json_array(self):
        """A bare JSON array passes through."""
        text = '["a", "b", "c"]'
        assert _extract_json_from_text(text) == text

    def test_fenced_code_block(self):
        """Extracts JSON from ```json fences."""
        text = 'Here are topics:\n```json\n{"topics": ["a"]}\n```\nDone.'
        result = _extract_json_from_text(text)
        assert json.loads(result) == {"topics": ["a"]}

    def test_fenced_code_block_no_lang(self):
        """Extracts JSON from ``` fences without language tag."""
        text = 'Result:\n```\n{"topics": ["x"]}\n```'
        result = _extract_json_from_text(text)
        assert json.loads(result) == {"topics": ["x"]}

    def test_bare_json_object_in_prose(self):
        """Extracts bare {...} from surrounding prose."""
        text = 'The result is {"key": "value"} as expected.'
        result = _extract_json_from_text(text)
        assert json.loads(result) == {"key": "value"}

    def test_no_json_returns_original(self):
        """When no JSON is found, original text is returned."""
        text = "This is just plain text with no JSON at all."
        assert _extract_json_from_text(text) == text

    def test_nested_braces(self):
        """Handles nested braces in JSON."""
        text = '{"outer": {"inner": "value"}}'
        result = _extract_json_from_text(text)
        assert json.loads(result) == {"outer": {"inner": "value"}}

    def test_multiple_json_blocks_first_wins(self):
        """When multiple fenced blocks exist, first valid one wins."""
        text = '```json\n{"first": true}\n```\nAnd:\n```json\n{"second": true}\n```'
        result = _extract_json_from_text(text)
        assert json.loads(result) == {"first": True}


class TestUnwrapTopic:
    """Tests for _unwrap_topic()."""

    def test_plain_string(self):
        """Plain string passes through."""
        assert _unwrap_topic("machine learning") == "machine learning"

    def test_dict_with_topic_key(self):
        """Dict with 'topic' key extracts value."""
        assert _unwrap_topic({"topic": "neural networks"}) == "neural networks"

    def test_dict_with_name_key(self):
        """Dict with 'name' key extracts value."""
        assert _unwrap_topic({"name": "deep learning"}) == "deep learning"

    def test_dict_with_topic_label_key(self):
        """Dict with 'topic label' key extracts value."""
        assert _unwrap_topic({"topic label": "NLP"}) == "NLP"

    def test_dict_with_unknown_key(self):
        """Dict with unknown key uses first value."""
        assert _unwrap_topic({"unknown": "some topic"}) == "some topic"

    def test_string_encoded_dict(self):
        """String-encoded Python dict extracts value via regex."""
        result = _unwrap_topic("{'topic': 'string encoded'}")
        assert result == "string encoded"

    def test_string_encoded_double_quotes(self):
        """String-encoded dict with double quotes."""
        result = _unwrap_topic('{"topic": "double quoted"}')
        # JSON-valid — loads as a dict first
        assert "double quoted" in result

    def test_int_returns_string(self):
        """Non-string non-dict returns str()."""
        assert _unwrap_topic(42) == "42"

    def test_none_returns_string(self):
        """None returns 'None'."""
        assert _unwrap_topic(None) == "None"


class TestFilterTopics:
    """Tests for _filter_topics()."""

    def test_removes_stoplist_words(self):
        """Stoplist single words are removed."""
        result = _filter_topics(["machine learning", "Power", "neural networks"])
        assert "machine learning" in result
        assert "neural networks" in result
        assert "power" not in result

    def test_removes_empty(self):
        """Empty strings are removed."""
        result = _filter_topics(["valid topic", "", "  ", "another topic"])
        assert result == ["valid topic", "another topic"]

    def test_allows_duplicates(self):
        """_filter_topics normalizes but does not deduplicate — that's the caller's job."""
        result = _filter_topics(["Machine Learning", "machine learning", "deep learning"])
        # _filter_topics normalizes and removes stoplist, but keeps duplicates
        assert "machine learning" in result
        assert "deep learning" in result

    def test_preserves_first_occurrence_order(self):
        """Order of first occurrence is preserved."""
        result = _filter_topics(["B topic", "A topic", "C topic"])
        assert result == ["b topic", "a topic", "c topic"]

    def test_all_filtered_returns_empty(self):
        """All stoplist words returns empty list."""
        result = _filter_topics(["Power", "Love", "Death"])
        assert result == []


class TestStripHtml:
    """Tests for strip_html()."""

    def test_basic_tags(self):
        """Strips basic HTML tags."""
        assert strip_html("<p>Hello</p>") == "Hello"

    def test_nested_tags(self):
        """Strips nested tags."""
        result = strip_html("<div><p>Some <em>emphasized</em> text</p></div>")
        assert "Some" in result
        assert "emphasized" in result
        assert "text" in result
        assert "<" not in result

    def test_html_entities(self):
        """Non-breaking spaces are handled (strip_html handles &nbsp; as \xa0)."""
        result = strip_html("A\xa0B&amp;C")
        assert "A" in result
        # \xa0 → space
        assert "B" in result

    def test_already_clean_text(self):
        """Clean text passes through."""
        assert strip_html("Just plain text") == "Just plain text"

    def test_empty_string(self):
        """Empty string returns empty."""
        assert strip_html("") == ""

    def test_collapses_whitespace(self):
        """Multiple spaces are collapsed."""
        result = strip_html("too    many    spaces")
        assert "  " not in result


class TestCleanCalibreTags:
    """Tests for clean_calibre_tags()."""

    def test_removes_noise_tags(self):
        """Noise tags like 'General', 'Fiction' are removed."""
        result = clean_calibre_tags(["General", "Science Fiction", "Fiction"])
        assert "Science Fiction" in result
        assert "General" not in result
        assert "Fiction" not in result

    def test_strips_compound_prefix(self):
        """'Fiction - Science Fiction' → 'Science Fiction'."""
        result = clean_calibre_tags(["Fiction - Science Fiction"])
        assert len(result) == 1
        assert result[0] == "Science Fiction"

    def test_strips_compound_suffix(self):
        """'Science Fiction - General' → 'Science Fiction'."""
        result = clean_calibre_tags(["Science Fiction - General"])
        assert len(result) == 1
        assert result[0] == "Science Fiction"

    def test_keeps_real_tags(self):
        """Legitimate tags are preserved."""
        result = clean_calibre_tags(["Philosophy", "Psychology", "Self-Help"])
        assert result == ["Philosophy", "Psychology", "Self-Help"]

    def test_deduplicates_case_insensitive(self):
        """Duplicate tags (case-insensitive) are deduplicated."""
        result = clean_calibre_tags(["science fiction", "Science Fiction"])
        assert len(result) == 1

    def test_removes_substring_duplicates(self):
        """Keeps longer/more-specific tag, removes substring."""
        result = clean_calibre_tags(["Science Fiction", "Science"])
        assert "Science Fiction" in result
        assert "Science" not in result

    def test_empty_input(self):
        """Empty list returns empty."""
        assert clean_calibre_tags([]) == []

    def test_caps_at_10(self):
        """Returns at most 10 tags."""
        tags = [f"Tag {i}" for i in range(15)]
        result = clean_calibre_tags(tags)
        assert len(result) <= 10


class TestBuildBatchSchema:
    """Tests for _build_batch_schema()."""

    def test_single_chunk(self):
        """Schema for 1 chunk, 5 topics."""
        schema = _build_batch_schema(1, 5)
        assert schema["type"] == "object"
        assert "1" in schema["properties"]
        assert schema["properties"]["1"]["items"]["type"] == "string"
        assert schema["properties"]["1"]["minItems"] == 5
        assert schema["properties"]["1"]["maxItems"] == 5
        assert schema["required"] == ["1"]

    def test_multiple_chunks(self):
        """Schema for 5 chunks, 3 topics."""
        schema = _build_batch_schema(5, 3)
        assert len(schema["properties"]) == 5
        for i in range(1, 6):
            assert str(i) in schema["properties"]
            assert schema["properties"][str(i)]["minItems"] == 3
        assert schema["required"] == ["1", "2", "3", "4", "5"]

    def test_schema_structure(self):
        """Schema has correct JSON schema structure."""
        schema = _build_batch_schema(2, 4)
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema
        # Each property is an array of strings
        for prop in schema["properties"].values():
            assert prop["type"] == "array"
            assert prop["items"] == {"type": "string"}


class TestIsLitellmModel:
    """Tests for _is_litellm_model()."""

    def test_gemini_model(self):
        """Gemini models return True."""
        assert _is_litellm_model("gemini/gemini-3-flash-preview") is True

    def test_lm_studio_model(self):
        """LM Studio models return True."""
        assert _is_litellm_model("lm_studio/qwen2.5-7b") is True

    def test_ollama_model(self):
        """Ollama models return False."""
        assert _is_litellm_model("gemma3:4b") is False

    def test_plain_model_name(self):
        """Plain model names return False."""
        assert _is_litellm_model("llama3") is False

    def test_empty_string(self):
        """Empty string returns False."""
        assert _is_litellm_model("") is False


# ---- Mocked tests ----


class TestExtractBookThemes:
    """Tests for extract_book_themes()."""

    @patch("libtrails.topic_extractor._get_client")
    def test_success_returns_themes(self, mock_get_client):
        """Successful extraction returns list of theme strings."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"themes": ["epic fantasy", "magic systems", "political intrigue"]}'
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        themes = extract_book_themes(
            title="Mistborn", author="Brandon Sanderson", model="gemma3:27b"
        )

        assert len(themes) == 3
        assert "epic fantasy" in themes
        assert "magic systems" in themes

    @patch("libtrails.topic_extractor._get_client")
    def test_includes_tags_in_prompt(self, mock_get_client):
        """Tags are included in prompt after cleaning."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"themes": ["sci-fi"]}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        extract_book_themes(
            title="Dune",
            author="Frank Herbert",
            tags=["Science Fiction", "General"],
            model="gemma3:27b",
        )

        call_args = mock_client.post.call_args
        prompt = (
            call_args[1]["json"]["prompt"] if "json" in call_args[1] else call_args[0][1]["prompt"]
        )
        assert "Science Fiction" in prompt
        # "General" should be filtered out by clean_calibre_tags
        assert "Tags:" in prompt, "Expected 'Tags:' section in prompt"
        tags_section = prompt.split("Tags:")[1].split("\n")[0]
        assert "General" not in tags_section

    @patch("libtrails.topic_extractor._get_client")
    def test_timeout_returns_empty(self, mock_get_client):
        """Timeout returns empty list."""
        mock_client = MagicMock()
        mock_client.post.side_effect = httpx.TimeoutException("Timeout")
        mock_get_client.return_value = mock_client

        themes = extract_book_themes(title="Test", author="Author", model="gemma3:27b")
        assert themes == []

    @patch("libtrails.topic_extractor._get_client")
    def test_error_returns_empty(self, mock_get_client):
        """General error returns empty list."""
        mock_client = MagicMock()
        mock_client.post.side_effect = Exception("Connection refused")
        mock_get_client.return_value = mock_client

        themes = extract_book_themes(title="Test", author="Author", model="gemma3:27b")
        assert themes == []

    @patch("libtrails.topic_extractor._get_client")
    def test_description_html_stripped(self, mock_get_client):
        """HTML in description is stripped before inclusion."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"themes": ["adventure"]}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        extract_book_themes(
            title="Test",
            author="Author",
            description="<p>A <b>bold</b> adventure</p>",
            model="gemma3:27b",
        )

        call_args = mock_client.post.call_args
        prompt = call_args[1]["json"]["prompt"]
        assert "<p>" not in prompt
        assert "<b>" not in prompt
        assert "bold" in prompt


class TestExtractTopicsBatched:
    """Tests for extract_topics_batched()."""

    @patch("libtrails.topic_extractor._get_client")
    def test_success_with_batch(self, mock_get_client):
        """Successful batch extraction returns topics per chunk."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": json.dumps(
                {
                    "1": ["topic a", "topic b"],
                    "2": ["topic c", "topic d"],
                }
            )
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        results = extract_topics_batched(
            chunks=["chunk 1 text", "chunk 2 text"],
            book_title="Test Book",
            author="Author",
            model="gemma3:4b",
            batch_size=5,
            num_topics=2,
        )

        assert len(results) == 2
        assert len(results[0]) == 2
        assert len(results[1]) == 2

    @patch("libtrails.topic_extractor._extract_single_contextualized")
    @patch("libtrails.topic_extractor._extract_batch")
    def test_fallback_on_batch_failure(self, mock_batch, mock_single):
        """Falls back to individual extraction when batch fails."""
        mock_batch.return_value = None  # Batch failed
        mock_single.return_value = ["fallback topic"]

        results = extract_topics_batched(
            chunks=["chunk 1"],
            book_title="Test",
            author="Author",
            model="gemma3:4b",
            batch_size=5,
        )

        assert len(results) == 1
        mock_single.assert_called_once()

    @patch("libtrails.topic_extractor._get_client")
    def test_progress_callback_called(self, mock_get_client):
        """Progress callback is called."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"1": ["t1"], "2": ["t2"]}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        progress_calls = []

        def callback(completed, total):
            progress_calls.append((completed, total))

        extract_topics_batched(
            chunks=["c1", "c2"],
            book_title="Test",
            author="Author",
            model="gemma3:4b",
            batch_size=5,
            num_topics=1,
            progress_callback=callback,
        )

        assert len(progress_calls) >= 1
        assert progress_calls[-1][0] == 2  # All completed

    @patch("libtrails.topic_extractor._get_client")
    def test_themes_in_context(self, mock_get_client):
        """Book themes are included in context."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"1": ["t1"]}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        extract_topics_batched(
            chunks=["chunk text"],
            book_title="Dune",
            author="Herbert",
            book_themes=["desert ecology", "spice economy"],
            model="gemma3:4b",
            batch_size=5,
            num_topics=1,
        )

        call_args = mock_client.post.call_args
        prompt = call_args[1]["json"]["prompt"]
        assert "desert ecology" in prompt
        assert "spice economy" in prompt


class TestExtractTopicsSingleOptimized:
    """Tests for extract_topics_single_optimized()."""

    @patch("libtrails.topic_extractor._get_client")
    def test_success_ollama_path(self, mock_get_client):
        """Successful extraction via Ollama returns normalized topics."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "response": '{"topics": ["desert ecology", "spice trade", "fremen culture"]}'
        }
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        topics = extract_topics_single_optimized(
            text="A passage about Dune",
            context="Book: Dune by Frank Herbert",
            model="gemma3:4b",
            num_topics=3,
        )

        assert len(topics) == 3
        assert "desert ecology" in topics

    @patch("libtrails.topic_extractor._call_litellm")
    def test_litellm_path(self, mock_call):
        """Uses litellm for gemini/ models."""
        mock_call.return_value = '{"topics": ["topic a", "topic b"]}'

        topics = extract_topics_single_optimized(
            text="A passage",
            context="Book context",
            model="gemini/flash",
            num_topics=2,
        )

        mock_call.assert_called_once()
        assert len(topics) == 2

    @patch("libtrails.topic_extractor._call_litellm")
    def test_content_filter_returns_empty(self, mock_call):
        """ContentFilterError returns empty list without retry."""
        mock_call.side_effect = ContentFilterError("blocked")

        topics = extract_topics_single_optimized(
            text="Sensitive passage",
            context="Context",
            model="gemini/flash",
        )

        assert topics == []
        # Should NOT retry on content filter
        assert mock_call.call_count == 1

    @patch("libtrails.topic_extractor._call_litellm")
    def test_retry_on_general_error(self, mock_call):
        """General errors trigger retries with backoff."""
        mock_call.side_effect = [
            Exception("rate limit"),
            Exception("rate limit"),
            '{"topics": ["recovered topic"]}',
        ]

        with patch("time.sleep"):  # Don't actually sleep
            topics = extract_topics_single_optimized(
                text="A passage",
                context="Context",
                model="gemini/flash",
            )

        assert mock_call.call_count == 3
        assert "recovered topic" in topics

    @patch("libtrails.topic_extractor._call_litellm")
    def test_max_retries_exhausted_returns_empty(self, mock_call):
        """After 3 retries, returns empty list."""
        mock_call.side_effect = Exception("persistent error")

        with patch("time.sleep"):
            topics = extract_topics_single_optimized(
                text="A passage",
                context="Context",
                model="gemini/flash",
            )

        assert mock_call.call_count == 3
        assert topics == []

    @patch("libtrails.topic_extractor._get_client")
    def test_extended_prompt_uses_more_demos(self, mock_get_client):
        """Extended prompt includes additional demos."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": '{"topics": ["t1"]}'}
        mock_response.raise_for_status = MagicMock()
        mock_client.post.return_value = mock_response
        mock_get_client.return_value = mock_client

        extract_topics_single_optimized(
            text="passage",
            context="context",
            model="gemma3:4b",
            use_extended_prompt=True,
        )

        call_args = mock_client.post.call_args
        prompt = call_args[1]["json"]["prompt"]
        # Extended prompt has extra demos (e.g., Lean Startup passage)
        assert "Lean Startup" in prompt


class TestCheckOllamaAvailable:
    """Tests for check_ollama_available()."""

    @patch("subprocess.run")
    def test_model_available(self, mock_run):
        """Returns True when model is in ollama list output."""
        mock_run.return_value = MagicMock(stdout="NAME\ngemma3:4b\nllama3:8b\n", returncode=0)
        assert check_ollama_available("gemma3:4b") is True

    @patch("subprocess.run")
    def test_model_not_available(self, mock_run):
        """Returns False when model is not in output."""
        mock_run.return_value = MagicMock(stdout="NAME\nllama3:8b\n", returncode=0)
        assert check_ollama_available("gemma3:4b") is False

    @patch("subprocess.run")
    def test_connection_error(self, mock_run):
        """Returns False on connection error."""
        mock_run.side_effect = Exception("Connection refused")
        assert check_ollama_available("gemma3:4b") is False

    @patch("subprocess.run")
    def test_timeout(self, mock_run):
        """Returns False on timeout."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ollama", timeout=5)
        assert check_ollama_available("gemma3:4b") is False


class TestGetAvailableModels:
    """Tests for get_available_models()."""

    @patch("subprocess.run")
    def test_returns_model_names(self, mock_run):
        """Returns list of model names."""
        mock_run.return_value = MagicMock(
            stdout="NAME            SIZE\ngemma3:4b       2.5GB\nllama3:8b       4.7GB\n",
            returncode=0,
        )
        models = get_available_models()
        assert "gemma3:4b" in models
        assert "llama3:8b" in models

    @patch("subprocess.run")
    def test_connection_error_returns_empty(self, mock_run):
        """Returns empty list on error."""
        mock_run.side_effect = Exception("Connection refused")
        assert get_available_models() == []

    @patch("subprocess.run")
    def test_nonzero_returncode_returns_empty(self, mock_run):
        """Returns empty list when command fails."""
        mock_run.return_value = MagicMock(stdout="", returncode=1)
        assert get_available_models() == []
