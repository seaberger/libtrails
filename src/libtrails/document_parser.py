"""Document parsing for EPUB and PDF files."""

import zipfile
import re
from pathlib import Path
from selectolax.parser import HTMLParser


def extract_text(file_path: Path) -> str:
    """
    Extract plain text from a document (EPUB or PDF).

    Routes to the appropriate parser based on file extension.
    """
    suffix = file_path.suffix.lower()

    if suffix == '.epub':
        return extract_text_from_epub(file_path)
    elif suffix == '.pdf':
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def extract_text_from_epub(epub_path: Path) -> str:
    """
    Extract plain text from an EPUB file.

    Uses selectolax for fast HTML parsing.
    """
    text_parts = []

    with zipfile.ZipFile(epub_path, 'r') as zf:
        # Get content files in reading order if possible
        content_files = []

        # Try to read the spine from content.opf
        for name in zf.namelist():
            if name.endswith('.opf'):
                try:
                    opf_content = zf.read(name).decode('utf-8', errors='ignore')
                    # Extract itemrefs from spine
                    # This is a simple extraction - could be improved
                    pass
                except:
                    pass

        # Fallback: get all HTML files
        for name in sorted(zf.namelist()):
            if name.endswith(('.xhtml', '.html', '.htm')) and 'toc' not in name.lower():
                content_files.append(name)

        for name in content_files:
            try:
                content = zf.read(name).decode('utf-8', errors='ignore')
                tree = HTMLParser(content)

                # Remove non-content elements
                for tag in tree.css('script, style, nav, header, footer'):
                    tag.decompose()

                # Extract text
                text = tree.text(separator=' ')
                text = _clean_text(text)

                if len(text) > 50:
                    text_parts.append(text)

            except Exception as e:
                continue

    return '\n\n'.join(text_parts)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF file using docling.

    Docling handles complex layouts, tables, and reading order.
    Returns markdown-formatted text which works well for chunking.
    """
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # Export to markdown for clean text with structure
    markdown_text = result.document.export_to_markdown()

    # Clean up the markdown for our purposes
    text = _clean_markdown(markdown_text)

    return text


def _clean_text(text: str) -> str:
    """Clean extracted text from EPUB."""
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove common EPUB artifacts
    text = re.sub(r'\[\d+\]', '', text)  # Footnote markers
    text = text.strip()
    return text


def _clean_markdown(text: str) -> str:
    """Clean markdown from docling for topic extraction."""
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove markdown image references
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    # Keep headers but simplify them for chunking
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = text.strip()
    return text


def get_book_metadata_from_epub(epub_path: Path) -> dict:
    """Extract metadata from EPUB."""
    metadata = {}

    with zipfile.ZipFile(epub_path, 'r') as zf:
        for name in zf.namelist():
            if name.endswith('.opf'):
                try:
                    content = zf.read(name).decode('utf-8', errors='ignore')
                    tree = HTMLParser(content)

                    title = tree.css_first('dc\\:title, title')
                    if title:
                        metadata['title'] = title.text()

                    creator = tree.css_first('dc\\:creator, creator')
                    if creator:
                        metadata['author'] = creator.text()

                except:
                    pass

    return metadata
