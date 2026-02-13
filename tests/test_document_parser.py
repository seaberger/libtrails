"""Tests for document parsing functionality."""

import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from libtrails.document_parser import (
    _clean_markdown,
    _html_to_structured_text,
    extract_text,
    extract_text_from_epub,
    extract_text_from_pdf,
    get_book_metadata_from_epub,
)


class TestExtractText:
    """Tests for the main extract_text router."""

    def test_routes_epub(self, tmp_path):
        """Test that .epub files route to EPUB parser."""
        epub_path = tmp_path / "test.epub"
        epub_path.touch()

        with patch("libtrails.document_parser.extract_text_from_epub") as mock:
            mock.return_value = "EPUB content"
            result = extract_text(epub_path)

        assert result == "EPUB content"
        mock.assert_called_once_with(epub_path)

    def test_routes_pdf(self, tmp_path):
        """Test that .pdf files route to PDF parser."""
        pdf_path = tmp_path / "test.pdf"
        pdf_path.touch()

        with patch("libtrails.document_parser.extract_text_from_pdf") as mock:
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
        with zipfile.ZipFile(epub_path, "w") as zf:
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
        long_content = (
            "This is the real content of the chapter that should be extracted from the EPUB file."
        )

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr(
                "toc.xhtml", "<html><body>Table of Contents with navigation links</body></html>"
            )
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

        with zipfile.ZipFile(epub_path, "w") as zf:
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

        with zipfile.ZipFile(epub_path, "w") as zf:
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


class TestHtmlToStructuredText:
    """Tests for _html_to_structured_text()."""

    def test_p_tags_create_paragraph_breaks(self):
        """<p> tags become \\n\\n."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = _html_to_structured_text(html)
        assert "First paragraph." in result
        assert "Second paragraph." in result
        # Should have paragraph break between them
        assert "\n\n" in result

    def test_br_creates_newline(self):
        """<br> becomes single newline."""
        html = "Line one.<br>Line two."
        result = _html_to_structured_text(html)
        assert "Line one." in result
        assert "Line two." in result
        assert "\n" in result

    def test_hr_creates_paragraph_break(self):
        """<hr> becomes double newline."""
        html = "Before.<hr>After."
        result = _html_to_structured_text(html)
        assert "Before." in result
        assert "After." in result

    def test_div_creates_paragraph_break(self):
        """<div> creates paragraph break."""
        html = "<div>Block one.</div><div>Block two.</div>"
        result = _html_to_structured_text(html)
        assert "Block one." in result
        assert "Block two." in result
        assert "\n\n" in result

    def test_heading_tags_create_breaks(self):
        """<h1>-<h6> create paragraph breaks."""
        html = "<h1>Title</h1><p>Content here.</p>"
        result = _html_to_structured_text(html)
        assert "Title" in result
        assert "Content here." in result

    def test_strips_inline_tags(self):
        """Inline tags (<em>, <strong>, <span>, <a>) are stripped."""
        html = "<p>Some <em>emphasized</em> and <strong>bold</strong> text.</p>"
        result = _html_to_structured_text(html)
        assert "Some" in result
        assert "emphasized" in result
        assert "bold" in result
        assert "<em>" not in result
        assert "<strong>" not in result

    def test_preserves_text_content(self):
        """All text content is preserved."""
        html = "<div><p>Hello <span>world</span>!</p></div>"
        result = _html_to_structured_text(html)
        assert "Hello" in result
        assert "world" in result

    def test_decodes_html_entities(self):
        """Decodes common HTML entities."""
        html = "<p>A &amp; B &lt; C &gt; D &quot;E&quot;</p>"
        result = _html_to_structured_text(html)
        assert "A & B" in result
        assert "< C" in result
        assert "> D" in result

    def test_collapses_excessive_newlines(self):
        """3+ consecutive newlines collapse to 2."""
        html = "<p>A</p><p></p><p></p><p>B</p>"
        result = _html_to_structured_text(html)
        assert "\n\n\n" not in result

    def test_empty_html(self):
        """Empty HTML returns empty string."""
        assert _html_to_structured_text("") == ""

    def test_nbsp_handling(self):
        """Non-breaking spaces are converted to regular spaces."""
        html = "<p>Word&nbsp;word</p>"
        result = _html_to_structured_text(html)
        assert "Word word" in result


class TestExtractTextFromPdf:
    """Tests for extract_text_from_pdf()."""

    @patch("libtrails.document_parser.fitz")
    def test_extracts_text(self, mock_fitz):
        """Extracts text from PDF pages."""
        # Text must be >= 100 words to avoid docling OCR fallback
        page1_text = "Page one content. " + " ".join(f"word{i}" for i in range(60))
        page2_text = "Page two content. " + " ".join(f"word{i}" for i in range(60))
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = page1_text
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = page2_text

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page1, mock_page2])
        mock_fitz.open.return_value = mock_doc

        result = extract_text_from_pdf(Path("/fake/book.pdf"))
        assert "Page one content." in result
        assert "Page two content." in result
        assert "\n\n" in result

    @patch("libtrails.document_parser.fitz")
    def test_skips_empty_pages(self, mock_fitz):
        """Skips pages with no text."""
        # Text must be >= 100 words to avoid docling OCR fallback
        page1_text = "Content. " + " ".join(f"word{i}" for i in range(60))
        page3_text = "More content. " + " ".join(f"word{i}" for i in range(60))
        mock_page1 = MagicMock()
        mock_page1.get_text.return_value = page1_text
        mock_page2 = MagicMock()
        mock_page2.get_text.return_value = ""
        mock_page3 = MagicMock()
        mock_page3.get_text.return_value = page3_text

        mock_doc = MagicMock()
        mock_doc.__iter__ = lambda self: iter([mock_page1, mock_page2, mock_page3])
        mock_fitz.open.return_value = mock_doc

        result = extract_text_from_pdf(Path("/fake/book.pdf"))
        assert "Content." in result
        assert "More content." in result
        # Empty page should not leave extra whitespace
        parts = [p for p in result.split("\n\n") if p.strip()]
        assert len(parts) == 2


class TestGetBookMetadataFromEpub:
    """Tests for get_book_metadata_from_epub()."""

    def test_extracts_title_and_author(self, tmp_path):
        """Extracts title and author from OPF file."""
        epub_path = tmp_path / "test.epub"
        opf_content = """<?xml version="1.0"?>
        <package>
            <metadata>
                <title>Siddhartha</title>
                <creator>Hermann Hesse</creator>
            </metadata>
        </package>"""

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("content.opf", opf_content)

        metadata = get_book_metadata_from_epub(epub_path)
        assert metadata.get("title") == "Siddhartha"
        assert metadata.get("author") == "Hermann Hesse"

    def test_handles_missing_metadata(self, tmp_path):
        """Returns empty dict when no OPF file exists."""
        epub_path = tmp_path / "test.epub"
        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("chapter1.xhtml", "<html><body>Content</body></html>")

        metadata = get_book_metadata_from_epub(epub_path)
        assert metadata == {}

    def test_handles_partial_metadata(self, tmp_path):
        """Returns available fields when some are missing."""
        epub_path = tmp_path / "test.epub"
        opf_content = """<?xml version="1.0"?>
        <package>
            <metadata>
                <title>Only Title</title>
            </metadata>
        </package>"""

        with zipfile.ZipFile(epub_path, "w") as zf:
            zf.writestr("content.opf", opf_content)

        metadata = get_book_metadata_from_epub(epub_path)
        assert "title" in metadata
        assert "author" not in metadata
