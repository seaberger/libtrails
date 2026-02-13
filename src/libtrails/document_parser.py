"""Document parsing for EPUB and PDF files."""

import logging
import re
import zipfile
from pathlib import Path

import fitz
from selectolax.parser import HTMLParser

logger = logging.getLogger(__name__)


def extract_text(file_path: Path) -> str:
    """
    Extract plain text from a document (EPUB or PDF).

    Routes to the appropriate parser based on file extension.
    """
    suffix = file_path.suffix.lower()

    if suffix == ".epub":
        return extract_text_from_epub(file_path)
    elif suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def extract_text_from_epub(epub_path: Path) -> str:
    """
    Extract plain text from an EPUB file.

    Uses selectolax for fast HTML parsing.
    """
    text_parts = []

    with zipfile.ZipFile(epub_path, "r") as zf:
        # Get content files in reading order if possible
        content_files = []

        # Get all HTML/XML content files (sorted for consistent ordering)
        for name in sorted(zf.namelist()):
            # Match standard extensions OR Calibre's .html_split_NNN format
            filename = name.split("/")[-1]
            is_html_content = name.endswith(
                (".xhtml", ".html", ".htm", ".xml")
            ) or filename.startswith(".html_split_")
            if not is_html_content:
                continue
            # Skip TOC, NCX, and OPF files by checking filename (not full path)
            filename = name.split("/")[-1].lower()
            if filename.startswith("toc") or ".ncx" in filename or ".opf" in filename:
                continue
            content_files.append(name)

        for name in content_files:
            try:
                content = zf.read(name).decode("utf-8", errors="ignore")
                # Strip XML namespaces that break selectolax parsing
                # (e.g., xmlns:epub="http://www.idpf.org/2007/ops" causes empty text extraction)
                content = re.sub(r'\s+xmlns(?::[a-zA-Z]+)?="[^"]*"', "", content)
                # Convert XHTML self-closing tags to proper HTML
                # (e.g., <title/> becomes <title></title> - void elements excluded)
                void_elements = {
                    "area",
                    "base",
                    "br",
                    "col",
                    "embed",
                    "hr",
                    "img",
                    "input",
                    "link",
                    "meta",
                    "param",
                    "source",
                    "track",
                    "wbr",
                }

                def fix_self_closing(match):
                    tag = match.group(1).lower()
                    if tag in void_elements:
                        return match.group(0)  # Keep void elements as-is
                    return f"<{match.group(1)}{match.group(2)}></{match.group(1)}>"

                content = re.sub(r"<([a-zA-Z][a-zA-Z0-9]*)([^>]*)/>", fix_self_closing, content)
                tree = HTMLParser(content)

                # Remove non-content elements
                for tag in tree.css("script, style, nav, header, footer"):
                    tag.decompose()

                # Extract text preserving paragraph structure
                cleaned_html = tree.html or ""
                text = _html_to_structured_text(cleaned_html)

                if len(text.split()) > 10:
                    text_parts.append(text)

            except Exception:
                continue

    return "\n\n".join(text_parts)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract plain text from a PDF file.

    Tries PyMuPDF first (fast). If fewer than 100 words are extracted
    (likely a scanned-image PDF), falls back to docling OCR.
    """
    text = _extract_pdf_pymupdf(pdf_path)
    word_count = len(text.split())

    if word_count >= 100:
        logger.info("PyMuPDF extracted %d words from %s", word_count, pdf_path.name)
        return text

    logger.info(
        "PyMuPDF extracted only %d words from %s, falling back to docling OCR",
        word_count,
        pdf_path.name,
    )
    return _extract_pdf_docling(pdf_path)


def _extract_pdf_pymupdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF (fast path)."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for page in doc:
        text = page.get_text().strip()
        if text:
            pages.append(text)
    doc.close()
    return "\n\n".join(pages)


def _extract_pdf_docling(pdf_path: Path) -> str:
    """Extract text from scanned PDF using docling OCR (slow path)."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    ocr_options = _pick_ocr_engine()
    pipeline_options = PdfPipelineOptions(do_ocr=True, ocr_options=ocr_options)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    result = converter.convert(str(pdf_path))
    md_text = result.document.export_to_markdown()
    text = _clean_markdown(md_text)
    word_count = len(text.split())
    logger.info("Docling OCR extracted %d words from %s", word_count, pdf_path.name)
    return text


def _pick_ocr_engine():
    """Pick the best available OCR engine: Tesseract > EasyOCR > macOS native."""
    import shutil

    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrMacOptions,
        TesseractOcrOptions,
    )

    if shutil.which("tesseract"):
        logger.info("Using Tesseract OCR engine")
        return TesseractOcrOptions()

    try:
        import easyocr  # noqa: F401

        logger.info("Using EasyOCR engine")
        return EasyOcrOptions()
    except ImportError:
        pass

    logger.info("Using macOS native OCR engine")
    return OcrMacOptions()


# Block-level HTML elements that should produce paragraph breaks
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|h[1-6]|li|blockquote|tr|table|section|article"
    r"|aside|figcaption|figure|main|details|summary|dd|dt)\b[^>]*>",
    re.IGNORECASE,
)


def _html_to_structured_text(html: str) -> str:
    """
    Convert HTML to plain text preserving paragraph boundaries.

    Block-level elements (<p>, <div>, <h1>-<h6>, etc.) become paragraph
    breaks (double newlines). Inline elements are stripped, leaving their
    text content. Horizontal whitespace is collapsed but paragraph
    structure is preserved.
    """
    # Convert <br> to single newline
    text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    # Convert <hr> to paragraph break
    text = re.sub(r"<hr\s*/?>", "\n\n", text, flags=re.IGNORECASE)
    # Convert block-level tags to paragraph breaks
    text = _BLOCK_TAG_RE.sub("\n\n", text)
    # Strip remaining HTML tags (inline: <em>, <strong>, <span>, <a>, etc.)
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode common HTML entities
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&rsquo;", "\u2019")
        .replace("&lsquo;", "\u2018")
        .replace("&rdquo;", "\u201d")
        .replace("&ldquo;", "\u201c")
        .replace("&mdash;", "\u2014")
        .replace("&ndash;", "\u2013")
        .replace("&hellip;", "\u2026")
        .replace("&nbsp;", " ")
        .replace("\xa0", " ")
    )
    # Remove EPUB artifacts
    text = re.sub(r"\[\d+\]", "", text)  # Footnote markers
    # Collapse horizontal whitespace (spaces/tabs) but preserve newlines
    text = re.sub(r"[ \t]+", " ", text)
    # Clean spaces around newlines
    text = re.sub(r" ?\n ?", "\n", text)
    # Collapse 3+ consecutive newlines to exactly 2 (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove leading/trailing whitespace from each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Final cleanup of excessive paragraph breaks
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_markdown(text: str) -> str:
    """Clean markdown from docling for topic extraction."""
    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove markdown image references
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    # Keep headers but simplify them for chunking
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = text.strip()
    return text


def get_book_metadata_from_epub(epub_path: Path) -> dict:
    """Extract metadata from EPUB."""
    metadata = {}

    with zipfile.ZipFile(epub_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(".opf"):
                try:
                    content = zf.read(name).decode("utf-8", errors="ignore")
                    tree = HTMLParser(content)

                    title = tree.css_first("dc\\:title, title")
                    if title:
                        metadata["title"] = title.text()

                    creator = tree.css_first("dc\\:creator, creator")
                    if creator:
                        metadata["author"] = creator.text()

                except Exception:
                    pass

    return metadata
