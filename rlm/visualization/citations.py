"""
RLM Citation Highlighting - Highlight cited text in documents.

Provides functions to:
- Find citation snippets in document text
- Wrap matches in HTML mark tags
- Handle overlapping citations
- Link citations to extracted records
"""

import html
import re
from typing import List, Dict, Any, Tuple


def escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(text)


def find_citation_positions(
    text: str,
    citations: List[Dict[str, Any]]
) -> List[Tuple[int, int, int, str]]:
    """
    Find positions of citation snippets in text.

    Args:
        text: Document text to search
        citations: List of citation dicts with 'snippet' and 'page' keys

    Returns:
        List of (start, end, citation_id, snippet) tuples, sorted by start position
    """
    positions = []

    for i, citation in enumerate(citations):
        snippet = citation.get("snippet", "")
        if not snippet or len(snippet) < 3:
            continue

        # Escape regex special chars but allow some flexibility
        # Normalize whitespace for matching
        pattern = re.escape(snippet)
        pattern = re.sub(r'\\ +', r'\\s+', pattern)  # Flexible whitespace

        try:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                positions.append((match.start(), match.end(), i, snippet))
        except re.error:
            # If regex fails, try simple substring search
            idx = text.lower().find(snippet.lower())
            if idx >= 0:
                positions.append((idx, idx + len(snippet), i, snippet))

    # Sort by start position
    positions.sort(key=lambda x: x[0])
    return positions


def merge_overlapping_positions(
    positions: List[Tuple[int, int, int, str]]
) -> List[Tuple[int, int, List[int], str]]:
    """
    Merge overlapping citation positions.

    Args:
        positions: List of (start, end, citation_id, snippet) tuples

    Returns:
        List of (start, end, [citation_ids], combined_snippet) tuples
    """
    if not positions:
        return []

    merged = []
    current_start, current_end, current_ids, current_snippet = (
        positions[0][0], positions[0][1], [positions[0][2]], positions[0][3]
    )

    for start, end, cid, snippet in positions[1:]:
        if start <= current_end:
            # Overlapping - extend and merge
            current_end = max(current_end, end)
            if cid not in current_ids:
                current_ids.append(cid)
        else:
            # No overlap - save current and start new
            merged.append((current_start, current_end, current_ids, current_snippet))
            current_start, current_end, current_ids, current_snippet = (
                start, end, [cid], snippet
            )

    # Don't forget the last one
    merged.append((current_start, current_end, current_ids, current_snippet))
    return merged


def highlight_citations(
    text: str,
    citations: List[Dict[str, Any]],
    mark_class: str = "citation",
    add_ids: bool = True
) -> str:
    """
    Highlight citation snippets in text with HTML mark tags.

    Args:
        text: Document text to highlight
        citations: List of citation dicts with 'snippet' key
        mark_class: CSS class for mark elements
        add_ids: Whether to add citation IDs as data attributes

    Returns:
        HTML string with citations wrapped in <mark> tags

    Example:
        html = highlight_citations(
            "John Smith is the CEO.",
            [{"snippet": "John Smith", "page": 1}]
        )
        # Returns: '<mark class="citation" data-citation="0">John Smith</mark> is the CEO.'
    """
    if not citations:
        return escape_html(text)

    # Find all citation positions
    positions = find_citation_positions(text, citations)
    if not positions:
        return escape_html(text)

    # Merge overlapping positions
    merged = merge_overlapping_positions(positions)

    # Build highlighted text
    result = []
    last_end = 0

    for start, end, citation_ids, _ in merged:
        # Add text before this citation
        if start > last_end:
            result.append(escape_html(text[last_end:start]))

        # Add highlighted citation
        cited_text = escape_html(text[start:end])
        ids_attr = f' data-citations="{",".join(map(str, citation_ids))}"' if add_ids else ''
        result.append(f'<mark class="{mark_class}"{ids_attr}>{cited_text}</mark>')

        last_end = end

    # Add remaining text
    if last_end < len(text):
        result.append(escape_html(text[last_end:]))

    return ''.join(result)


def highlight_by_page(
    pages: List[str],
    citations: List[Dict[str, Any]],
    mark_class: str = "citation"
) -> List[str]:
    """
    Highlight citations in each page separately.

    Args:
        pages: List of page texts
        citations: List of citation dicts with 'snippet' and 'page' keys
        mark_class: CSS class for mark elements

    Returns:
        List of HTML strings, one per page
    """
    result = []

    for page_num, page_text in enumerate(pages, 1):
        # Filter citations for this page
        page_citations = [
            c for c in citations
            if c.get("page") == page_num
        ]

        if page_citations:
            highlighted = highlight_citations(page_text, page_citations, mark_class)
        else:
            highlighted = escape_html(page_text)

        result.append(highlighted)

    return result


def create_citation_index(citations: List[Dict[str, Any]]) -> str:
    """
    Create an HTML index of all citations.

    Args:
        citations: List of citation dicts

    Returns:
        HTML string with citation list
    """
    if not citations:
        return "<p>No citations recorded.</p>"

    lines = ['<ol class="citation-index">']

    for i, citation in enumerate(citations):
        snippet = escape_html(citation.get("snippet", "")[:100])
        page = citation.get("page", "?")
        note = escape_html(citation.get("note", ""))

        lines.append(f'<li id="citation-{i}">')
        lines.append(f'  <span class="citation-snippet">"{snippet}"</span>')
        lines.append(f'  <span class="citation-page">(Page {page})</span>')
        if note:
            lines.append(f'  <span class="citation-note"> - {note}</span>')
        lines.append('</li>')

    lines.append('</ol>')
    return '\n'.join(lines)
