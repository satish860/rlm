"""
RLM Core Types - Pydantic models for extraction results and metadata.

These types are used throughout the library and returned by the public API.
"""

from typing import Generic, TypeVar, List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


T = TypeVar('T')


class Citation(BaseModel):
    """
    A verbatim citation from the source document.

    Every extracted fact should have supporting citations that can be
    verified against the original document.

    Attributes:
        snippet: Exact verbatim text from the source document
        page: Page number where the snippet was found (1-indexed)
        note: Optional interpretation or context about the citation
    """
    snippet: str = Field(..., description="Exact verbatim text from source")
    page: int = Field(..., description="Page number (1-indexed)")
    note: str = Field(default="", description="Interpretation or context")


class ThinkingEntry(BaseModel):
    """
    A recorded reasoning step from the extraction process.

    The root model's thinking is logged for transparency and debugging.

    Attributes:
        timestamp: When this thought was recorded
        thought: The reasoning or observation text
    """
    timestamp: datetime = Field(default_factory=datetime.now)
    thought: str = Field(..., description="Reasoning or observation")


class ConfidenceEntry(BaseModel):
    """
    A progress evaluation snapshot during extraction.

    Tracks extraction progress and confidence over time.

    Attributes:
        records: Number of records extracted so far
        pages_covered: Number of pages processed
        total_pages: Total pages in document
        coverage: Fraction of document covered (0.0 to 1.0)
        issues: Any problems encountered
        notes: Additional observations
        confidence: Overall confidence score (0.0 to 1.0)
    """
    records: int = Field(default=0, description="Records extracted")
    pages_covered: int = Field(default=0, description="Pages processed")
    total_pages: int = Field(default=0, description="Total pages")
    coverage: float = Field(default=0.0, description="Coverage fraction")
    issues: str = Field(default="", description="Problems encountered")
    notes: str = Field(default="", description="Additional observations")
    confidence: float = Field(default=0.0, description="Confidence 0.0-1.0")


class ExtractionResult(BaseModel):
    """
    Result of an extraction operation.

    Contains the extracted data, citations, reasoning trace, and metadata.

    Example:
        result = rlm.extract("invoice.pdf", schema=Invoice)
        print(f"Extracted {len(result.data)} invoices")
        print(f"Confidence: {result.confidence_history[-1].confidence:.0%}")
        for citation in result.citations:
            print(f"  p.{citation.page}: {citation.snippet[:50]}...")
    """
    # Extracted data
    data: List[Any] = Field(default_factory=list, description="Extracted records")
    schema_info: Dict[str, Any] = Field(default_factory=dict, description="Schema metadata")
    verification: Dict[str, Any] = Field(default_factory=dict, description="Verification status")

    # Reasoning trace
    citations: List[Citation] = Field(default_factory=list, description="Source evidence")
    thinking_log: List[ThinkingEntry] = Field(default_factory=list, description="Model reasoning")
    confidence_history: List[ConfidenceEntry] = Field(default_factory=list, description="Progress over time")

    # Metadata
    iterations: int = Field(default=0, description="Iterations used")
    data_file: Optional[str] = Field(default=None, description="Path if saved to file")
    document_path: Optional[str] = Field(default=None, description="Source document path")
    extraction_time: Optional[float] = Field(default=None, description="Time taken in seconds")

    class Config:
        arbitrary_types_allowed = True


class QueryResult(BaseModel):
    """
    Result of a query operation.

    Contains the answer, supporting citations, and reasoning trace.

    Example:
        result = rlm.query("report.pdf", "What was Q3 revenue?")
        print(f"Answer: {result.answer}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Source: page {result.citations[0].page}")
    """
    answer: str = Field(..., description="The answer to the question")
    citations: List[Citation] = Field(default_factory=list, description="Supporting evidence")
    confidence: float = Field(default=0.0, description="Confidence in answer")
    thinking_log: List[ThinkingEntry] = Field(default_factory=list, description="Reasoning trace")
    document_path: Optional[str] = Field(default=None, description="Source document path")
