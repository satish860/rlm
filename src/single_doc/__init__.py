"""Single document RLM implementation."""

from .rlm import SingleDocRLM, query_document
from .indexer import SingleDocIndex, Section, build_index
from .converter import DocumentConverter

# Extraction models
from .models import Paper, PaperList, Figure, FigureList, StructuredSummary

# Custom exceptions
from .errors import (
    ExtractionError,
    SchemaGenerationError,
    JSONConversionError,
    UserCancelledError,
)

__all__ = [
    # Core classes
    "SingleDocRLM",
    "query_document",
    "SingleDocIndex",
    "Section",
    "build_index",
    "DocumentConverter",
    # Extraction models
    "Paper",
    "PaperList",
    "Figure",
    "FigureList",
    "StructuredSummary",
    # Exceptions
    "ExtractionError",
    "SchemaGenerationError",
    "JSONConversionError",
    "UserCancelledError",
]
