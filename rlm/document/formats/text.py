"""
RLM Text Reader - Read plain text documents.

Example:
    from rlm.document.formats.text import TextReader

    reader = TextReader()
    text = reader.read("document.txt")
"""

from pathlib import Path
from typing import Optional

from rlm.exceptions import DocumentError


class TextReader:
    """
    Plain text document reader.

    Reads .txt and other plain text files with encoding detection.
    """

    SUPPORTED_EXTENSIONS = [".txt", ".text", ".rst", ".log"]

    def read(self, path: str, encoding: str = None) -> str:
        """
        Read plain text file.

        Args:
            path: Path to text file
            encoding: Specific encoding (auto-detect if None)

        Returns:
            Text content

        Raises:
            DocumentError: If reading fails
        """
        path = Path(path)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")

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


def read_text(path: str, encoding: str = None) -> str:
    """
    Convenience function to read a text file.

    Args:
        path: Path to text file
        encoding: Specific encoding (default: auto-detect)

    Returns:
        Text content
    """
    return TextReader().read(path, encoding=encoding)
