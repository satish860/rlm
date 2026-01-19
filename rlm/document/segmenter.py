"""
RLM Document Segmenter - Split documents into logical sections with page ranges.

Uses LLM to identify document structure and create a table of contents.
Supports parallel processing for large documents.

Example:
    from rlm.document.segmenter import segment_document, split_into_pages

    # Split into pages
    pages = split_into_pages(text, lines_per_page=50)

    # Segment into logical sections (requires LLM)
    segments = segment_document(text)
    for seg in segments:
        print(f"[{seg.page_range.start}-{seg.page_range.end}] {seg.heading}")
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field

from rlm.exceptions import DocumentError


class PageRange(BaseModel):
    """Page range for a document segment."""
    start: int = Field(..., description="Starting page number (1-indexed)")
    end: int = Field(..., description="Ending page number (1-indexed)")


class Segment(BaseModel):
    """A logical segment/section of a document."""
    heading: str = Field(..., description="Section heading or title")
    description: str = Field(default="", description="Brief description of section content")
    page_range: PageRange = Field(..., description="Page range covered by this section")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "heading": self.heading,
            "description": self.description,
            "page_range": {
                "start": self.page_range.start,
                "end": self.page_range.end
            }
        }


def split_into_pages(text: str, lines_per_page: int = 50) -> List[str]:
    """
    Split text into pages by line count.

    Each page is prefixed with a page number marker for LLM reference.

    Args:
        text: Document text content
        lines_per_page: Number of lines per page (default: 50)

    Returns:
        List of page texts with page number markers

    Example:
        pages = split_into_pages(document_text)
        print(f"Document has {len(pages)} pages")
        print(pages[0])  # First page with marker
    """
    lines = text.split('\n')
    pages = []

    for i in range(0, len(lines), lines_per_page):
        page_lines = lines[i:i + lines_per_page]
        page_num = i // lines_per_page + 1
        page_text = f"### Page Number: [PG:{page_num}]\n" + '\n'.join(page_lines)
        pages.append(page_text)

    return pages


def segment_document(
    text: str,
    lines_per_page: int = 50,
    chunk_size: int = 10,
    max_workers: int = 3,
    llm_call: Callable = None,
    verbose: bool = False
) -> List[Segment]:
    """
    Segment a document into logical sections using LLM analysis.

    Splits the document into chunks and processes them in parallel to
    identify section boundaries and create a table of contents.

    Args:
        text: Document text content
        lines_per_page: Lines per page for splitting (default: 50)
        chunk_size: Pages per chunk for LLM processing (default: 10)
        max_workers: Parallel workers for LLM calls (default: 3)
        llm_call: Function to call LLM for segmentation. Signature:
                  llm_call(chunk_text, start_page, end_page) -> List[Segment]
                  If None, returns simple page-based segments.
        verbose: Print progress messages

    Returns:
        List of Segment objects sorted by page range

    Example:
        # Without LLM (simple page-based segments)
        segments = segment_document(text)

        # With LLM (requires provider setup)
        from rlm.providers import get_provider
        provider = get_provider("openrouter")

        def llm_segment(chunk, start, end):
            # Call LLM to identify sections
            ...

        segments = segment_document(text, llm_call=llm_segment)
    """
    pages = split_into_pages(text, lines_per_page)

    if verbose:
        print(f"Split into {len(pages)} pages")

    # If no LLM call provided, return simple page-based segments
    if llm_call is None:
        return _create_simple_segments(pages, chunk_size)

    # Create chunks for parallel processing
    chunks = []
    for i in range(0, len(pages), chunk_size):
        chunk_pages = pages[i:i + chunk_size]
        chunk_text = '\n\n'.join(chunk_pages)
        start_page = i + 1
        end_page = min(i + chunk_size, len(pages))
        chunks.append((chunk_text, start_page, end_page))

    if verbose:
        print(f"Created {len(chunks)} chunks for parallel processing")

    # Process chunks in parallel
    all_segments = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(llm_call, chunk_text, start, end): (start, end)
            for chunk_text, start, end in chunks
        }

        for future in as_completed(futures):
            start, end = futures[future]
            try:
                segments = future.result()
                all_segments.extend(segments)
                if verbose:
                    print(f"  Processed pages {start}-{end}: {len(segments)} segments")
            except Exception as e:
                if verbose:
                    print(f"  Error processing pages {start}-{end}: {e}")
                # Create fallback segment for this range
                all_segments.append(Segment(
                    heading=f"Pages {start}-{end}",
                    description="(Segmentation failed)",
                    page_range=PageRange(start=start, end=end)
                ))

    # Sort by page range
    all_segments.sort(key=lambda s: s.page_range.start)

    return all_segments


def _create_simple_segments(pages: List[str], chunk_size: int) -> List[Segment]:
    """Create simple page-based segments without LLM."""
    segments = []
    total_pages = len(pages)

    for i in range(0, total_pages, chunk_size):
        start = i + 1
        end = min(i + chunk_size, total_pages)
        segments.append(Segment(
            heading=f"Pages {start}-{end}",
            description=f"Document pages {start} through {end}",
            page_range=PageRange(start=start, end=end)
        ))

    return segments


def get_section_content(pages: List[str], segment: Segment, padding: int = 0) -> str:
    """
    Extract content for a specific segment from pages.

    Args:
        pages: List of page texts (from split_into_pages)
        segment: Segment to extract
        padding: Extra pages before/after (default: 0)

    Returns:
        Combined text for the segment's page range
    """
    start = max(0, segment.page_range.start - 1 - padding)
    end = min(len(pages), segment.page_range.end + padding)
    return '\n\n'.join(pages[start:end])


def build_toc(segments: List[Segment]) -> str:
    """
    Build a table of contents string from segments.

    Args:
        segments: List of Segment objects

    Returns:
        Formatted TOC string

    Example:
        toc = build_toc(segments)
        print(toc)
        # [1-5] Introduction
        # [6-15] Methods
        # [16-25] Results
    """
    lines = []
    for seg in segments:
        pr = seg.page_range
        lines.append(f"[{pr.start}-{pr.end}] {seg.heading}")
    return "\n".join(lines)
