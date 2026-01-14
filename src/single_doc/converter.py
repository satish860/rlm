"""Document converter using markitdown.

Converts various document formats (PDF, DOCX, HTML, TXT) to markdown.
"""

import os
from pathlib import Path
from typing import Optional

from markitdown import MarkItDown


class DocumentConverter:
    """Convert documents to markdown format."""

    SUPPORTED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".doc",
        ".pptx",
        ".ppt",
        ".xlsx",
        ".xls",
        ".html",
        ".htm",
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
    }

    def __init__(self):
        """Initialize the converter."""
        self._converter = MarkItDown()

    def convert(
        self,
        source_path: str,
        output_path: Optional[str] = None,
    ) -> tuple[str, str]:
        """
        Convert a document to markdown.

        Args:
            source_path: Path to the source document (PDF, DOCX, HTML, etc.)
            output_path: Optional path to save the markdown file.
                        If not provided, uses source_path with .md extension.

        Returns:
            Tuple of (markdown_content, output_path)

        Raises:
            FileNotFoundError: If source file doesn't exist
            ValueError: If file extension is not supported
        """
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        ext = source.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file extension: {ext}. "
                f"Supported: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}"
            )

        # If already markdown, just read it
        if ext == ".md":
            markdown_content = source.read_text(encoding="utf-8")
        else:
            # Convert using markitdown
            result = self._converter.convert(str(source))
            markdown_content = result.text_content

        # Determine output path
        if output_path is None:
            output_path = str(source.with_suffix(".converted.md"))

        # Save markdown file
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown_content, encoding="utf-8")

        return markdown_content, str(output)

    def convert_to_string(self, source_path: str) -> str:
        """
        Convert a document to markdown string without saving.

        Args:
            source_path: Path to the source document

        Returns:
            Markdown content as string
        """
        source = Path(source_path)

        if not source.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        ext = source.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        if ext == ".md":
            return source.read_text(encoding="utf-8")

        result = self._converter.convert(str(source))
        return result.text_content

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.SUPPORTED_EXTENSIONS


def convert_document(
    source_path: str,
    output_path: Optional[str] = None,
) -> tuple[str, str]:
    """
    Convenience function to convert a document.

    Args:
        source_path: Path to source document
        output_path: Optional output path for markdown

    Returns:
        Tuple of (markdown_content, output_path)
    """
    converter = DocumentConverter()
    return converter.convert(source_path, output_path)
