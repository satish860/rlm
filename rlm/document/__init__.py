"""
RLM Document - Document readers and segmentation.

Main entry points:
- read_document(): Read any supported document format
- segment_document(): Split document into logical sections
- split_into_pages(): Split text into pages

Converter interface:
- BaseConverter: Abstract base for custom converters
- register_converter(): Register custom converter (e.g., Mistral)
- get_converter(): Get converter instance

Example:
    from rlm.document import read_document, segment_document

    # Read a PDF
    text = read_document("report.pdf")

    # Segment into sections
    segments = segment_document(text)
    for seg in segments:
        print(f"[{seg.page_range.start}-{seg.page_range.end}] {seg.heading}")
"""

from rlm.document.reader import (
    read_document,
    DocumentReader,
    get_document_reader,
)

from rlm.document.segmenter import (
    segment_document,
    split_into_pages,
    get_section_content,
    build_toc,
    Segment,
    PageRange,
)

from rlm.document.converter import (
    BaseConverter,
    MarkitdownConverter,
    PlainTextConverter,
    register_converter,
    get_converter,
    set_default_converter,
    list_converters,
)

__all__ = [
    # Main functions
    "read_document",
    "segment_document",
    "split_into_pages",
    "get_section_content",
    "build_toc",
    # Classes
    "DocumentReader",
    "Segment",
    "PageRange",
    # Converter interface
    "BaseConverter",
    "MarkitdownConverter",
    "PlainTextConverter",
    "register_converter",
    "get_converter",
    "set_default_converter",
    "list_converters",
    "get_document_reader",
]
