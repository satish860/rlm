"""
RLM Document Converter - Abstract interface for document conversion.

This module provides an open interface for document conversion that can be
extended with different backends (markitdown, Mistral, custom LLMs, etc.)

Example:
    # Use default markitdown converter
    converter = get_converter("markitdown")
    text = converter.convert("document.pdf")

    # Register custom converter
    register_converter("mistral", MistralConverter)
    converter = get_converter("mistral")
"""

from abc import ABC, abstractmethod
from typing import Dict, Type, Optional, Any
from pathlib import Path

from rlm.exceptions import DocumentError


class BaseConverter(ABC):
    """
    Abstract base class for document converters.

    Implement this interface to add support for new conversion backends
    like Mistral, GPT-4V, or custom LLM-based converters.

    Example:
        class MistralConverter(BaseConverter):
            def __init__(self, api_key: str = None):
                self.api_key = api_key or os.getenv("MISTRAL_API_KEY")

            def convert(self, path: str, **kwargs) -> str:
                # Call Mistral API to convert document
                ...

            @property
            def name(self) -> str:
                return "mistral"

            @property
            def supported_formats(self) -> list[str]:
                return [".pdf", ".docx", ".pptx", ".xlsx"]
    """

    @abstractmethod
    def convert(self, path: str, **kwargs) -> str:
        """
        Convert a document to text/markdown.

        Args:
            path: Path to the document file
            **kwargs: Converter-specific options

        Returns:
            Converted text content

        Raises:
            DocumentError: If conversion fails
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this converter."""
        pass

    @property
    @abstractmethod
    def supported_formats(self) -> list:
        """Return list of supported file extensions (e.g., ['.pdf', '.docx'])."""
        pass

    def supports(self, path: str) -> bool:
        """Check if this converter supports the given file."""
        ext = Path(path).suffix.lower()
        return ext in self.supported_formats


class MarkitdownConverter(BaseConverter):
    """
    Document converter using Microsoft's markitdown library.

    markitdown supports: PDF, Word, PowerPoint, Excel, Images, HTML, and more.
    It converts documents to clean Markdown format.

    Install: pip install markitdown
    """

    def __init__(self):
        self._md = None

    def _get_markitdown(self):
        """Lazy load markitdown to avoid import errors if not installed."""
        if self._md is None:
            try:
                from markitdown import MarkItDown
                self._md = MarkItDown()
            except ImportError:
                raise DocumentError(
                    "markitdown not installed. Install with: pip install markitdown"
                )
        return self._md

    def convert(self, path: str, **kwargs) -> str:
        """
        Convert document to markdown using markitdown.

        Args:
            path: Path to document
            **kwargs: Additional options (passed to markitdown)

        Returns:
            Markdown text content
        """
        path = Path(path)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")

        md = self._get_markitdown()
        try:
            result = md.convert(str(path))
            return result.text_content
        except Exception as e:
            raise DocumentError(f"Failed to convert {path}: {e}")

    @property
    def name(self) -> str:
        return "markitdown"

    @property
    def supported_formats(self) -> list:
        return [
            ".pdf", ".docx", ".doc", ".pptx", ".ppt",
            ".xlsx", ".xls", ".html", ".htm",
            ".jpg", ".jpeg", ".png", ".gif", ".bmp",
            ".csv", ".json", ".xml", ".zip"
        ]


class PlainTextConverter(BaseConverter):
    """
    Simple converter that reads plain text files.

    Handles encoding detection and fallback.
    """

    def convert(self, path: str, encoding: str = None, **kwargs) -> str:
        """
        Read plain text file with encoding detection.

        Args:
            path: Path to text file
            encoding: Specific encoding (auto-detect if None)

        Returns:
            Text content
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

    @property
    def name(self) -> str:
        return "plaintext"

    @property
    def supported_formats(self) -> list:
        return [".txt", ".md", ".markdown", ".rst", ".text"]


# Registry of available converters
_converters: Dict[str, Type[BaseConverter]] = {
    "markitdown": MarkitdownConverter,
    "plaintext": PlainTextConverter,
}

# Default converter to use
_default_converter: str = "markitdown"


def register_converter(name: str, converter_class: Type[BaseConverter]) -> None:
    """
    Register a custom converter.

    Example:
        class MistralConverter(BaseConverter):
            ...

        register_converter("mistral", MistralConverter)
    """
    _converters[name] = converter_class


def get_converter(name: str = None) -> BaseConverter:
    """
    Get a converter instance by name.

    Args:
        name: Converter name (default: "markitdown")

    Returns:
        BaseConverter instance

    Raises:
        DocumentError: If converter not found
    """
    name = name or _default_converter
    if name not in _converters:
        available = ", ".join(_converters.keys())
        raise DocumentError(f"Unknown converter: {name}. Available: {available}")
    return _converters[name]()


def set_default_converter(name: str) -> None:
    """Set the default converter to use."""
    global _default_converter
    if name not in _converters:
        raise DocumentError(f"Unknown converter: {name}")
    _default_converter = name


def list_converters() -> list:
    """List all registered converter names."""
    return list(_converters.keys())
