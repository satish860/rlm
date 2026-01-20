"""
RLM Document Converter - Abstract interface for document conversion.

This module provides an open interface for document conversion that can be
extended with different backends (Mistral OCR, markitdown, custom LLMs, etc.)

Example:
    # Use default Mistral converter
    converter = get_converter("mistral")
    text = converter.convert("document.pdf")

    # Use markitdown
    converter = get_converter("markitdown")
    text = converter.convert("document.pdf")
"""

import os
import base64
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


class MistralConverter(BaseConverter):
    """
    Document converter using Mistral's dedicated OCR API.

    Uses mistral-ocr-latest model for high-quality document OCR.
    Supports PDFs and images with table extraction and layout preservation.

    Requires: MISTRAL_API_KEY environment variable or pass api_key to constructor.
    Install: pip install mistralai
    """

    # Default API key (can be overridden via env var or constructor)
    DEFAULT_API_KEY = "S5FhVwZ42xTGYrXo29xmbGpDegHY4zHh"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY", self.DEFAULT_API_KEY)
        self.model = "mistral-ocr-latest"
        self._client = None

    def _get_client(self):
        """Lazy load Mistral client."""
        if self._client is None:
            try:
                from mistralai import Mistral
                self._client = Mistral(api_key=self.api_key)
            except ImportError:
                raise DocumentError(
                    "mistralai not installed. Install with: pip install mistralai"
                )
        return self._client

    def convert(self, path: str, **kwargs) -> str:
        """
        Convert document to text using Mistral OCR API.

        Args:
            path: Path to document (PDF or image)
            **kwargs: Additional options

        Returns:
            Extracted text content in markdown format
        """
        path = Path(path)
        if not path.exists():
            raise DocumentError(f"File not found: {path}")

        ext = path.suffix.lower()

        if ext == ".pdf":
            return self._convert_pdf(path)
        elif ext in [".jpg", ".jpeg", ".png", ".avif", ".gif", ".bmp", ".webp"]:
            return self._convert_image(path)
        else:
            raise DocumentError(f"Unsupported format for Mistral OCR: {ext}")

    def _convert_pdf(self, path: Path) -> str:
        """Convert PDF using Mistral OCR API."""
        client = self._get_client()

        # Read and encode PDF as base64
        with open(path, "rb") as f:
            pdf_bytes = f.read()
        pdf_b64 = base64.b64encode(pdf_bytes).decode("utf-8")

        try:
            from mistralai.models import DocumentURLChunk

            ocr_response = client.ocr.process(
                model=self.model,
                document=DocumentURLChunk(
                    document_url=f"data:application/pdf;base64,{pdf_b64}"
                ),
                include_image_base64=False
            )

            # Extract text from all pages
            all_text = []
            for i, page in enumerate(ocr_response.pages, 1):
                page_text = page.markdown if hasattr(page, 'markdown') else str(page)
                all_text.append(f"--- Page {i} ---\n{page_text}")

            return "\n\n".join(all_text)

        except Exception as e:
            raise DocumentError(f"Mistral OCR failed: {e}")

    def _convert_image(self, path: Path) -> str:
        """Convert image using Mistral OCR API."""
        client = self._get_client()

        # Read and encode image as base64
        with open(path, "rb") as f:
            img_bytes = f.read()
        img_b64 = base64.b64encode(img_bytes).decode("utf-8")

        # Determine mime type
        ext = path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".avif": "image/avif",
            ".bmp": "image/bmp",
        }
        mime_type = mime_types.get(ext, "image/png")

        try:
            from mistralai.models import ImageURLChunk

            ocr_response = client.ocr.process(
                model=self.model,
                document=ImageURLChunk(
                    image_url=f"data:{mime_type};base64,{img_b64}"
                ),
                include_image_base64=False
            )

            # Extract text from response
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                return ocr_response.pages[0].markdown
            elif hasattr(ocr_response, 'markdown'):
                return ocr_response.markdown
            else:
                return str(ocr_response)

        except Exception as e:
            raise DocumentError(f"Mistral OCR failed: {e}")

    @property
    def name(self) -> str:
        return "mistral"

    @property
    def supported_formats(self) -> list:
        return [".pdf", ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".avif"]


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
    "mistral": MistralConverter,
    "markitdown": MarkitdownConverter,
    "plaintext": PlainTextConverter,
}

# Default converter to use
_default_converter: str = "mistral"


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
        name: Converter name (default: "mistral")

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
