"""Pydantic models for structured extraction and summarization.

These models define the schemas for common extraction tasks (papers, figures)
and structured summaries. They are used with Instructor for guaranteed
JSON output with validation.
"""

from typing import Optional
from pydantic import BaseModel, Field


# =============================================================================
# Paper/Citation Extraction Models
# =============================================================================

class Paper(BaseModel):
    """A referenced paper or citation."""
    title: str = Field(description="Paper title")
    authors: list[str] = Field(description="List of author names")
    year: str = Field(description="Publication year")
    venue: Optional[str] = Field(default=None, description="Journal or conference name")
    url: Optional[str] = Field(default=None, description="URL or arXiv link")


class PaperList(BaseModel):
    """Container for extracted papers."""
    items: list[Paper] = Field(description="List of extracted papers")


# =============================================================================
# Figure Extraction Models
# =============================================================================

class Figure(BaseModel):
    """A figure in the document."""
    number: str = Field(description="Figure number (e.g., '1', '2a')")
    caption: str = Field(description="Figure caption text")
    section: Optional[str] = Field(default=None, description="Section where figure appears")


class FigureList(BaseModel):
    """Container for extracted figures."""
    items: list[Figure] = Field(description="List of extracted figures")


# =============================================================================
# Summary Models
# =============================================================================

class StructuredSummary(BaseModel):
    """Structured document summary for programmatic use."""
    title: str = Field(description="Document title")
    main_contribution: str = Field(description="Main contribution or thesis")
    problem: str = Field(description="Problem being addressed")
    approach: str = Field(description="Approach or methodology used")
    key_results: list[str] = Field(description="Key results or findings")
    limitations: list[str] = Field(default_factory=list, description="Limitations mentioned")


# =============================================================================
# Export all models
# =============================================================================

__all__ = [
    "Paper",
    "PaperList",
    "Figure",
    "FigureList",
    "StructuredSummary",
]
