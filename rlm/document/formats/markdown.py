"""
RLM Markdown Reader - Read Markdown documents.

Example:
    from rlm.document.formats.markdown import MarkdownReader

    reader = MarkdownReader()
    text = reader.read("document.md")
"""

from pathlib import Path
from typing import Optional

from rlm.exceptions import DocumentError


class MarkdownReader:
    """
    Markdown document reader.

    Reads .md and .markdown files with proper encoding handling.
    """

    def read(self, path: str, encoding: str = None) -> str:
        """
        Read Markdown file.

        Args:
            path: Path to Markdown file
            encoding: Specific encoding (auto-detect if None)

        Returns:
            Markdown text content

        Raises:
            DocumentError: If reading fails
        """
        path = Path(path)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")

        if path.suffix.lower() not in [".md", ".markdown"]:
            raise DocumentError(f"Not a Markdown file: {path}")

        # Try encodings in order
        encodings = [encoding] if encoding else ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for enc in encodings:
            if enc is None:
                continue
            try:
                with open(path, "r", encoding=enc) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        raise DocumentError(f"Could not decode {path} with any supported encoding")


def read_markdown(path: str, encoding: str = None) -> str:
    """
    Convenience function to read a Markdown file.

    Args:
        path: Path to Markdown file
        encoding: Specific encoding (default: auto-detect)

    Returns:
        Markdown text content
    """
    return MarkdownReader().read(path, encoding=encoding)
