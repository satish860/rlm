"""
RLM Document Reader - Unified document reading with format auto-detection.

This module provides a single entry point for reading documents of any
supported format. It auto-detects the format and uses the appropriate reader.

Example:
    from rlm.document.reader import read_document

    # Auto-detect format
    text = read_document("report.pdf")
    text = read_document("notes.md")
    text = read_document("data.txt")

    # Or use DocumentReader class for more control
    reader = DocumentReader(converter="markitdown")
    text = reader.read("document.pdf")
"""

from pathlib import Path
from typing import Optional, Union

from rlm.exceptions import DocumentError
from rlm.document.converter import get_converter, BaseConverter
from rlm.document.formats.pdf import PDFReader
from rlm.document.formats.markdown import MarkdownReader
from rlm.document.formats.text import TextReader


class DocumentReader:
    """
    Unified document reader with format auto-detection.

    Supports:
    - PDF files (.pdf) - via markitdown, pymupdf, or pdfplumber
    - Markdown files (.md, .markdown)
    - Text files (.txt, .text, .rst)
    - Office documents (.docx, .pptx, .xlsx) - via markitdown
    - HTML files (.html, .htm) - via markitdown

    Example:
        reader = DocumentReader()

        # Auto-detect format (uses Mistral OCR for PDFs)
        text = reader.read("document.pdf")

        # Specify PDF method
        text = reader.read("document.pdf", method="pymupdf")

        # Use markitdown instead
        reader = DocumentReader(converter="markitdown")
        text = reader.read("document.pdf")
    """

    # Format mappings
    PDF_EXTENSIONS = [".pdf"]
    MARKDOWN_EXTENSIONS = [".md", ".markdown"]
    TEXT_EXTENSIONS = [".txt", ".text", ".rst", ".log"]
    OFFICE_EXTENSIONS = [".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls"]
    HTML_EXTENSIONS = [".html", ".htm"]

    def __init__(self, converter: Union[str, BaseConverter, None] = None):
        """
        Initialize document reader.

        Args:
            converter: Converter name (e.g., "mistral", "markitdown") or
                      BaseConverter instance. Default: "mistral"
        """
        if isinstance(converter, str):
            self._converter = get_converter(converter)
        elif isinstance(converter, BaseConverter):
            self._converter = converter
        else:
            self._converter = None  # Will use default

        self._pdf_reader = PDFReader(self._converter)
        self._md_reader = MarkdownReader()
        self._text_reader = TextReader()

    def read(self, path: str, method: str = None, encoding: str = None) -> str:
        """
        Read a document, auto-detecting format.

        Args:
            path: Path to document
            method: For PDFs - extraction method ("markitdown", "pymupdf", "pdfplumber")
            encoding: For text files - specific encoding

        Returns:
            Document text content

        Raises:
            DocumentError: If format unsupported or reading fails
        """
        path = Path(path)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")

        ext = path.suffix.lower()

        # PDF files
        if ext in self.PDF_EXTENSIONS:
            return self._pdf_reader.read(str(path), method=method or "mistral")

        # Markdown files
        if ext in self.MARKDOWN_EXTENSIONS:
            return self._md_reader.read(str(path), encoding=encoding)

        # Plain text files
        if ext in self.TEXT_EXTENSIONS:
            return self._text_reader.read(str(path), encoding=encoding)

        # Office documents - use converter (markitdown supports these)
        if ext in self.OFFICE_EXTENSIONS + self.HTML_EXTENSIONS:
            converter = self._converter or get_converter("markitdown")
            return converter.convert(str(path))

        raise DocumentError(
            f"Unsupported file format: {ext}. "
            f"Supported: {self._list_supported_formats()}"
        )

    def _list_supported_formats(self) -> str:
        """List all supported formats."""
        all_exts = (
            self.PDF_EXTENSIONS +
            self.MARKDOWN_EXTENSIONS +
            self.TEXT_EXTENSIONS +
            self.OFFICE_EXTENSIONS +
            self.HTML_EXTENSIONS
        )
        return ", ".join(all_exts)

    def supports(self, path: str) -> bool:
        """Check if this reader supports the given file format."""
        ext = Path(path).suffix.lower()
        return ext in (
            self.PDF_EXTENSIONS +
            self.MARKDOWN_EXTENSIONS +
            self.TEXT_EXTENSIONS +
            self.OFFICE_EXTENSIONS +
            self.HTML_EXTENSIONS
        )


# Default reader instance
_default_reader: Optional[DocumentReader] = None


def read_document(
    path: str,
    method: str = None,
    encoding: str = None,
    converter: str = None
) -> str:
    """
    Read a document with auto-format detection.

    This is the main entry point for reading documents.

    Args:
        path: Path to document file
        method: For PDFs - "mistral" (default), "markitdown", "pymupdf", or "pdfplumber"
        encoding: For text files - specific encoding (default: auto-detect)
        converter: Converter to use (default: "mistral")

    Returns:
        Document text content

    Raises:
        DocumentError: If format unsupported or reading fails

    Example:
        # Simple usage (uses Mistral OCR)
        text = read_document("report.pdf")

        # Specify PDF extraction method
        text = read_document("report.pdf", method="pymupdf")

        # Use markitdown converter
        text = read_document("report.pdf", converter="markitdown")
    """
    reader = DocumentReader(converter=converter)
    return reader.read(path, method=method, encoding=encoding)


def get_document_reader(converter: str = None) -> DocumentReader:
    """
    Get a DocumentReader instance.

    Args:
        converter: Converter name (default: "markitdown")

    Returns:
        DocumentReader instance
    """
    return DocumentReader(converter=converter)
