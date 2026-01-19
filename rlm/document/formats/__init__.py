"""
RLM Document Formats - Format-specific readers.

- PDFReader: Read PDF files (markitdown, pymupdf, pdfplumber)
- MarkdownReader: Read Markdown files
- TextReader: Read plain text files
"""

from rlm.document.formats.pdf import PDFReader, read_pdf
from rlm.document.formats.markdown import MarkdownReader, read_markdown
from rlm.document.formats.text import TextReader, read_text

__all__ = [
    "PDFReader",
    "MarkdownReader",
    "TextReader",
    "read_pdf",
    "read_markdown",
    "read_text",
]
