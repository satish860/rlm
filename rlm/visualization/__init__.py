"""
RLM Visualization - HTML report generation.

Main functions:
- generate_html_report(): Create full extraction report
- highlight_citations(): Highlight cited text in documents
- create_citation_index(): Generate citation list
"""

from rlm.visualization.html import generate_html_report
from rlm.visualization.citations import (
    highlight_citations,
    highlight_by_page,
    create_citation_index,
)

__all__ = [
    "generate_html_report",
    "highlight_citations",
    "highlight_by_page",
    "create_citation_index",
]
