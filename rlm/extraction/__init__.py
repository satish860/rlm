"""
RLM Extraction - Structured data extraction with Pydantic.

Main functions:
- llm_extract: Extract structured data from document sections
- llm_extract_parallel: Parallel extraction across multiple sections
- get_section: Get content for specific pages
- ask_about_section: Ask questions about document sections
"""

from rlm.extraction.structured import (
    llm_extract,
    llm_extract_parallel,
    get_section,
    ask_about_section,
)
from rlm.extraction.schemas import (
    Contact,
    Invoice,
    Entity,
    TableRow,
    KeyValuePair,
    FinancialFigure,
)

__all__ = [
    # Extraction functions
    "llm_extract",
    "llm_extract_parallel",
    "get_section",
    "ask_about_section",
    # Built-in schemas
    "Contact",
    "Invoice",
    "Entity",
    "TableRow",
    "KeyValuePair",
    "FinancialFigure",
]
