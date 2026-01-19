"""
RLM PDF Reader - Read PDF documents using markitdown or fallback methods.

Primary method: markitdown (high quality, converts to Markdown)
Fallback methods: pymupdf, pdfplumber (basic text extraction)

Example:
    from rlm.document.formats.pdf import PDFReader

    reader = PDFReader()
    text = reader.read("document.pdf")

    # Use specific method
    text = reader.read("document.pdf", method="pymupdf")
"""

from pathlib import Path
from typing import Optional

from rlm.exceptions import DocumentError
from rlm.document.converter import get_converter, BaseConverter


class PDFReader:
    """
    PDF document reader with multiple backend support.

    Methods:
    - markitdown: High quality conversion to Markdown (default)
    - pymupdf: Fast text extraction using PyMuPDF/fitz
    - pdfplumber: Alternative text extraction
    """

    def __init__(self, converter: Optional[BaseConverter] = None):
        """
        Initialize PDF reader.

        Args:
            converter: Custom converter to use (default: markitdown)
        """
        self.converter = converter

    def read(self, path: str, method: str = "markitdown") -> str:
        """
        Read PDF and convert to text.

        Args:
            path: Path to PDF file
            method: Extraction method - "markitdown", "pymupdf", or "pdfplumber"

        Returns:
            Extracted text content

        Raises:
            DocumentError: If extraction fails
        """
        path = Path(path)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise DocumentError(f"Not a PDF file: {path}")

        if method == "markitdown":
            return self._read_with_markitdown(path)
        elif method == "pymupdf":
            return self._read_with_pymupdf(path)
        elif method == "pdfplumber":
            return self._read_with_pdfplumber(path)
        else:
            raise DocumentError(f"Unknown PDF method: {method}")

    def _read_with_markitdown(self, path: Path) -> str:
        """Read PDF using markitdown converter."""
        converter = self.converter or get_converter("markitdown")
        return converter.convert(str(path))

    def _read_with_pymupdf(self, path: Path) -> str:
        """Read PDF using PyMuPDF (fitz) - basic text extraction."""
        try:
            import fitz
        except ImportError:
            raise DocumentError(
                "pymupdf not installed. Install with: pip install pymupdf"
            )

        try:
            doc = fitz.open(str(path))
            text_parts = []
            for page_num, page in enumerate(doc, 1):
                text = page.get_text()
                text_parts.append(f"--- Page {page_num} ---\n{text}")
            doc.close()
            return "\n\n".join(text_parts)
        except Exception as e:
            raise DocumentError(f"PyMuPDF extraction failed: {e}")

    def _read_with_pdfplumber(self, path: Path) -> str:
        """Read PDF using pdfplumber - alternative text extraction."""
        try:
            import pdfplumber
        except ImportError:
            raise DocumentError(
                "pdfplumber not installed. Install with: pip install pdfplumber"
            )

        try:
            text_parts = []
            with pdfplumber.open(str(path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    text_parts.append(f"--- Page {page_num} ---\n{text}")
            return "\n\n".join(text_parts)
        except Exception as e:
            raise DocumentError(f"pdfplumber extraction failed: {e}")


def read_pdf(path: str, method: str = "markitdown") -> str:
    """
    Convenience function to read a PDF file.

    Args:
        path: Path to PDF file
        method: Extraction method (default: "markitdown")

    Returns:
        Extracted text content
    """
    return PDFReader().read(path, method=method)
