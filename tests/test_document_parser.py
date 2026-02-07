"""Tests for document parsing functionality."""

import zipfile
from unittest.mock import patch

import pytest

from libtrails.document_parser import (
    _clean_markdown,
    extract_text,
    extract_text_from_epub,
)


class TestExtractText:
    """Tests for the main extract_text router."""

    def test_routes_epub(self, tmp_path):
        """Test that .epub files route to EPUB parser."""
        epub_path = tmp_path / "test.epub"
        epub_path.touch()

        with patch('libtrails.document_parser.extract_text_from_epub') as mock:
            mock.return_value = "EPUB content"
            result = extract_text(epub_path)

        assert result == "EPUB content"
        mock.assert_called_once_with(epub_path)

    def test_routes_pdf(self, tmp_path):
        """Test that .pdf files route to PDF parser."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with patch('libtrails.document_parser.extract_text_from_pdf') as mock:
            mock.return_value = "PDF content"
            result = extract_text(pdf_path)

        assert result == "PDF content"
        mock.assert_called_once_with(pdf_path)

    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise ValueError."""
        txt_path = tmp_path / "test.txt"
        txt_path.touch()

        with pytest.raises(ValueError) as exc:
            extract_text(txt_path)

        assert "Unsupported format" in str(exc.value)


class TestExtractTextFromEpub:
    """Tests for EPUB text extraction."""

    def test_extracts_html_content(self, tmp_path):
        """Test extraction from EPUB with HTML content."""
        epub_path = tmp_path / "test.epub"

        # Create a minimal EPUB (ZIP with HTML content)
        with zipfile.ZipFile(epub_path, 'w') as zf:
            html_content = """
            <html>
            <body>
            <h1>Chapter 1</h1>
            <p>This is the first paragraph.</p>
            <p>This is the second paragraph.</p>
            </body>
            </html>
            """
            zf.writestr("chapter1.xhtml", html_content)

        result = extract_text_from_epub(epub_path)

        assert "Chapter 1" in result
        assert "first paragraph" in result
        assert "second paragraph" in result

    def test_skips_toc_files(self, tmp_path):
        """Test that TOC files are skipped."""
        epub_path = tmp_path / "test.epub"

        # Content needs to be > 50 chars to be included
        long_content = "This is the real content of the chapter that should be extracted from the EPUB file."

        with zipfile.ZipFile(epub_path, 'w') as zf:
            zf.writestr("toc.xhtml", "<html><body>Table of Contents with navigation links</body></html>")
            zf.writestr("chapter1.xhtml", f"<html><body>{long_content}</body></html>")

        result = extract_text_from_epub(epub_path)

        assert "Table of Contents" not in result
        assert "real content" in result

    def test_handles_multiple_chapters(self, tmp_path):
        """Test extraction from multiple chapter files."""
        epub_path = tmp_path / "test.epub"

        # Content needs to be > 50 chars to be included
        ch1 = "Chapter One contains lots of interesting text about the beginning of our story."
        ch2 = "Chapter Two continues the narrative with more exciting developments and plot twists."
        ch3 = "Chapter Three wraps up the story with a satisfying conclusion to all storylines."

        with zipfile.ZipFile(epub_path, 'w') as zf:
            zf.writestr("chapter1.xhtml", f"<html><body>{ch1}</body></html>")
            zf.writestr("chapter2.xhtml", f"<html><body>{ch2}</body></html>")
            zf.writestr("chapter3.html", f"<html><body>{ch3}</body></html>")

        result = extract_text_from_epub(epub_path)

        assert "Chapter One" in result
        assert "Chapter Two" in result
        assert "Chapter Three" in result

    def test_handles_empty_epub(self, tmp_path):
        """Test handling of EPUB with no content."""
        epub_path = tmp_path / "test.epub"

        with zipfile.ZipFile(epub_path, 'w') as zf:
            zf.writestr("metadata.opf", "<package></package>")

        result = extract_text_from_epub(epub_path)

        assert result == ""


class TestCleanMarkdown:
    """Tests for markdown cleaning utility.

    Note: _clean_markdown only removes images and heading markers,
    not links or horizontal rules.
    """

    def test_removes_images(self):
        """Test that image references are removed."""
        text = "Some text ![image](path/to/image.png) more text"
        result = _clean_markdown(text)

        assert "![image]" not in result
        assert "Some text" in result
        assert "more text" in result

    def test_removes_heading_markers(self):
        """Test that heading markers are removed."""
        text = "## Heading Two\n### Heading Three"
        result = _clean_markdown(text)

        assert "##" not in result
        assert "Heading Two" in result
        assert "Heading Three" in result

    def test_collapses_whitespace(self):
        """Test that excessive whitespace is collapsed."""
        text = "Line one\n\n\n\n\nLine two"
        result = _clean_markdown(text)

        # Should have at most 2 consecutive newlines
        assert "\n\n\n" not in result

    def test_preserves_links(self):
        """Test that links are preserved (not cleaned)."""
        text = "Click [here](http://example.com) for more"
        result = _clean_markdown(text)

        # Links are preserved in current implementation
        assert "[here]" in result
