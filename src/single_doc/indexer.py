"""Document indexer for single document RLM.

Robust segmentation with:
- Line markers for verifiable references
- Chunked parallel processing
- Coverage validation
- Title verification
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field

from ..config import SUB_MODEL, MAX_SECTION_CHARS_FOR_SUMMARY


# =============================================================================
# OpenRouter Client Setup
# =============================================================================

def _get_openrouter_client():
    """Get OpenAI client configured for OpenRouter."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )


def _get_instructor_client():
    """Get Instructor client with JSON mode for structured outputs."""
    return instructor.from_openai(
        _get_openrouter_client(),
        mode=instructor.Mode.JSON,
    )


def _strip_openrouter_prefix(model: str) -> str:
    """Strip openrouter/ prefix if present."""
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


# =============================================================================
# Pydantic Models
# =============================================================================

class LineRange(BaseModel):
    """Line range for a section."""
    start: int = Field(..., description="Starting line number (0-indexed)")
    end: int = Field(..., description="Ending line number (inclusive)")


class DocumentSection(BaseModel):
    """A section identified by LLM."""
    title: str = Field(..., description="Section heading exactly as it appears")
    description: str = Field(default="", description="Brief summary of section content")
    line_range: LineRange = Field(..., description="Line range for this section")


class Section(BaseModel):
    """A document section with all metadata."""
    title: str = Field(description="Section header/title")
    start_index: int = Field(description="Start line number")
    end_index: int = Field(description="End line number")
    summary: str = Field(default="", description="Contextual summary of this section")
    content: str = Field(default="", description="Actual text content of section")


class StructuredDocument(BaseModel):
    """Complete structured document with sections."""
    source_path: str = Field(description="Original document path")
    markdown_path: str = Field(description="Converted markdown path")
    total_lines: int = Field(description="Total lines in document")
    total_chars: int = Field(description="Total characters in document")
    sections: list[Section] = Field(default_factory=list, description="Document sections")

    @property
    def summaries(self) -> dict[str, str]:
        """Backward compatibility: summaries as dict."""
        return self.get_all_summaries()

    def get_section(self, title: str) -> Optional[Section]:
        """Get section by title."""
        for section in self.sections:
            if section.title == title:
                return section
        return None

    def get_section_names(self) -> list[str]:
        """Get list of section titles."""
        return [s.title for s in self.sections]

    def get_all_summaries(self) -> dict[str, str]:
        """Get all summaries as dict."""
        return {s.title: s.summary for s in self.sections}

    def to_json(self) -> str:
        """Serialize to JSON."""
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "StructuredDocument":
        """Deserialize from JSON."""
        return cls.model_validate_json(json_str)

    def save(self, path: str) -> None:
        """Save to file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "StructuredDocument":
        """Load from file."""
        json_str = Path(path).read_text(encoding="utf-8")
        return cls.from_json(json_str)


# =============================================================================
# LLM-Based Section Detection (with validation)
# =============================================================================

def add_line_markers(text: str, interval: int = 100) -> str:
    """Add line markers for verifiable LLM references."""
    lines = text.split('\n')
    marked = []
    for i, line in enumerate(lines):
        if i % interval == 0:
            marked.append(f"[LINE:{i}] {line}")
        else:
            marked.append(line)
    return '\n'.join(marked)


def segment_chunk_llm(
    chunk_text: str,
    start_line: int,
    end_line: int,
    model: str,
) -> list[DocumentSection]:
    """Segment a chunk of the document using LLM."""

    client = _get_instructor_client()
    model = _strip_openrouter_prefix(model)

    SYSTEM_PROMPT = """You are a document structure analyzer. Identify logical sections in the provided text.

CRITICAL REQUIREMENTS:
1. Use the [LINE:X] markers to determine accurate line numbers
2. Every line from start to end must be covered - NO GAPS
3. Section titles must EXACTLY match headers/headings in the document
4. Return consecutive line ranges that cover all lines
5. Look for: chapter titles, section headers, numbered sections, topic changes"""

    USER_PROMPT = f"""Analyze this document chunk (lines {start_line} to {end_line}) and identify all sections.

The text contains line markers in format [LINE:X] - use these for accurate line references.

<document>
{chunk_text}
</document>

Requirements:
- Cover ALL lines from {start_line} to {end_line} with no gaps
- Section titles must exactly match headers in the text
- Use [LINE:X] markers for accurate line numbers
- Include a brief description of each section's content"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            response_model=list[DocumentSection],
            max_tokens=4000,
            temperature=0,
        )
        return response
    except Exception as e:
        print(f"  [ERROR] Chunk segmentation failed: {e}")
        return [DocumentSection(
            title=f"Content (lines {start_line}-{end_line})",
            description="Unsegmented content",
            line_range=LineRange(start=start_line, end=end_line)
        )]


def validate_coverage(sections: list[DocumentSection], total_lines: int) -> list[str]:
    """Check for gaps in coverage."""
    errors = []
    if not sections:
        return ["No sections found"]

    sorted_sections = sorted(sections, key=lambda s: s.line_range.start)

    # Check first section starts at 0 (or close to it)
    if sorted_sections[0].line_range.start > 10:
        errors.append(f"Gap at start: lines 0-{sorted_sections[0].line_range.start-1}")

    # Check for gaps between sections
    for i in range(len(sorted_sections) - 1):
        current_end = sorted_sections[i].line_range.end
        next_start = sorted_sections[i + 1].line_range.start
        gap = next_start - current_end - 1
        if gap > 5:  # Allow small gaps (empty lines)
            errors.append(f"Gap: lines {current_end+1}-{next_start-1} ({gap} lines)")

    # Check last section ends near total_lines
    if sorted_sections[-1].line_range.end < total_lines - 20:
        errors.append(f"Gap at end: lines {sorted_sections[-1].line_range.end+1}-{total_lines-1}")

    return errors


def validate_titles_exist(
    sections: list[DocumentSection],
    lines: list[str],
) -> list[DocumentSection]:
    """Verify section titles exist in document, fix line numbers if needed."""
    validated = []

    for section in sections:
        title = section.title
        claimed_start = section.line_range.start

        # Search for title near claimed location (+/- 20 lines)
        found_at = None
        title_lower = title.lower().strip()

        # First check exact line
        if 0 <= claimed_start < len(lines):
            if title_lower in lines[claimed_start].lower():
                found_at = claimed_start

        # Then search nearby
        if found_at is None:
            for offset in range(1, 30):
                for direction in [1, -1]:
                    check_line = claimed_start + (offset * direction)
                    if 0 <= check_line < len(lines):
                        if title_lower in lines[check_line].lower():
                            found_at = check_line
                            break
                if found_at is not None:
                    break

        if found_at is not None and found_at != claimed_start:
            # Fix the start line
            offset = found_at - claimed_start
            section.line_range.start = found_at
            section.line_range.end = section.line_range.end + offset
            print(f"  [FIX] '{title}' moved from line {claimed_start} to {found_at}")

        validated.append(section)

    return validated


def deduplicate_sections(sections: list[DocumentSection]) -> list[DocumentSection]:
    """Remove duplicate sections from overlapping chunks."""
    if not sections:
        return []

    seen_titles = {}
    unique = []

    for section in sorted(sections, key=lambda s: s.line_range.start):
        # Normalize title for comparison
        key = re.sub(r'\s+', ' ', section.title.lower().strip())

        if key not in seen_titles:
            seen_titles[key] = section
            unique.append(section)
        else:
            # Keep existing (first occurrence)
            pass

    return sorted(unique, key=lambda s: s.line_range.start)


def fix_section_boundaries(sections: list[DocumentSection], total_lines: int) -> list[DocumentSection]:
    """Ensure sections have proper boundaries (no overlaps, no gaps)."""
    if not sections:
        return []

    sorted_sections = sorted(sections, key=lambda s: s.line_range.start)

    # Fix first section to start at 0
    if sorted_sections[0].line_range.start > 0:
        sorted_sections[0].line_range.start = 0

    # Fix boundaries between adjacent sections
    for i in range(len(sorted_sections) - 1):
        current = sorted_sections[i]
        next_sec = sorted_sections[i + 1]

        # Current section ends just before next section starts
        current.line_range.end = next_sec.line_range.start - 1

    # Fix last section to end at total_lines
    sorted_sections[-1].line_range.end = total_lines - 1

    return sorted_sections


def segment_document_llm(
    markdown_text: str,
    model: str,
    chunk_size: int = 800,
    overlap: int = 100,
    max_workers: int = 3,
) -> list[DocumentSection]:
    """
    LLM-based segmentation with chunking and validation.
    """
    # Add line markers
    marked_text = add_line_markers(markdown_text, interval=50)
    lines = markdown_text.split('\n')
    marked_lines = marked_text.split('\n')
    total_lines = len(lines)

    # For small documents, process in one chunk
    if total_lines <= chunk_size:
        print(f"  Processing as single chunk ({total_lines} lines)")
        sections = segment_chunk_llm(marked_text, 0, total_lines - 1, model)
        sections = validate_titles_exist(sections, lines)
        sections = fix_section_boundaries(sections, total_lines)
        return sections

    # Prepare chunks for large documents
    step = chunk_size - overlap
    chunks = []
    for i in range(0, total_lines, step):
        chunk_end = min(i + chunk_size, total_lines)
        chunk_text = '\n'.join(marked_lines[i:chunk_end])
        chunks.append((chunk_text, i, chunk_end - 1))

    print(f"  Processing {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")

    # Process chunks (parallel or sequential based on count)
    all_sections = []

    if len(chunks) <= 2 or max_workers <= 1:
        # Sequential processing for small number of chunks
        for idx, (chunk_text, start, end) in enumerate(chunks):
            print(f"  Chunk {idx+1}/{len(chunks)}: lines {start}-{end}")
            chunk_sections = segment_chunk_llm(chunk_text, start, end, model)
            all_sections.extend(chunk_sections)
    else:
        # Parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(segment_chunk_llm, chunk[0], chunk[1], chunk[2], model): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    chunk_sections = future.result()
                    all_sections.extend(chunk_sections)
                    print(f"  Chunk {idx+1}/{len(chunks)} completed: {len(chunk_sections)} sections")
                except Exception as e:
                    print(f"  Chunk {idx+1} failed: {e}")

    # Post-processing
    print(f"  Deduplicating {len(all_sections)} raw sections...")
    all_sections = deduplicate_sections(all_sections)
    print(f"  After dedup: {len(all_sections)} sections")

    # Validate titles exist
    all_sections = validate_titles_exist(all_sections, lines)

    # Fix boundaries
    all_sections = fix_section_boundaries(all_sections, total_lines)

    # Validate coverage
    gaps = validate_coverage(all_sections, total_lines)
    if gaps:
        print(f"  [WARNING] Coverage gaps: {gaps}")

    return all_sections


# =============================================================================
# Main Segmentation Function
# =============================================================================

def segment_document(
    markdown_text: str,
    model: str = SUB_MODEL,
) -> list[DocumentSection]:
    """
    LLM-based document segmentation with validation.

    Uses line markers for accurate references and validates
    that detected sections actually exist in the document.
    """
    lines = markdown_text.split('\n')
    total_lines = len(lines)

    print(f"Segmenting document ({total_lines} lines, {len(markdown_text):,} chars)")

    # LLM-based detection with validation
    print("  Using LLM-based detection with line markers...")
    sections = segment_document_llm(markdown_text, model)

    if sections and len(sections) >= 1:
        print(f"  SUCCESS: Found {len(sections)} sections")
        return sections

    # Fallback - single section
    print("  FALLBACK: Creating single section for entire document")
    return [DocumentSection(
        title="Document",
        description="Full document content",
        line_range=LineRange(start=0, end=total_lines - 1)
    )]


# =============================================================================
# Summary Generation
# =============================================================================

def generate_summary(
    title: str,
    content: str,
    document_context: str,
    model: str = SUB_MODEL,
) -> str:
    """Generate contextual summary for a section."""
    if len(content) > MAX_SECTION_CHARS_FOR_SUMMARY:
        content = content[:MAX_SECTION_CHARS_FOR_SUMMARY] + "\n...[truncated]"

    prompt = f"""<document>
{document_context[:5000]}
</document>

<section title="{title}">
{content}
</section>

Write a 1-2 sentence summary of this section that captures its key content. Be specific and factual.

Summary:"""

    try:
        client = _get_openrouter_client()
        model = _strip_openrouter_prefix(model)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [ERROR] Summary failed for '{title}': {e}")
        return f"Section about {title}"


# =============================================================================
# Main Index Builder
# =============================================================================

def build_index(
    source_path: str,
    output_dir: Optional[str] = None,
    model: str = SUB_MODEL,
    generate_summaries: bool = True,
) -> StructuredDocument:
    """
    Build complete structured index for a document.

    Uses tiered segmentation: regex -> LLM -> fallback
    """
    from .converter import DocumentConverter

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Document not found: {source_path}")

    # Set output directory
    if output_dir is None:
        out_dir = source.parent
    else:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to markdown
    print(f"\n{'='*60}")
    print(f"Building index for: {source.name}")
    print(f"{'='*60}")

    print(f"\nStep 1: Converting to markdown...")
    converter = DocumentConverter()
    markdown_path = str(out_dir / f"{source.stem}.md")
    markdown_text, markdown_path = converter.convert(str(source_path), markdown_path)

    lines = markdown_text.split('\n')
    print(f"  Result: {len(lines)} lines, {len(markdown_text):,} chars")

    # Step 2: Segment document (tiered approach)
    print(f"\nStep 2: Segmenting document...")
    doc_sections = segment_document(markdown_text, model=model)
    print(f"  Result: {len(doc_sections)} sections found")

    # Step 3: Build sections with content and summaries
    print(f"\nStep 3: Building section content and summaries...")
    sections = []

    for idx, sec_data in enumerate(doc_sections):
        title = sec_data.title
        start_idx = max(0, sec_data.line_range.start)
        end_idx = min(len(lines) - 1, sec_data.line_range.end)

        # Extract content
        content = '\n'.join(lines[start_idx:end_idx + 1])

        # Generate summary if requested
        summary = sec_data.description or ""
        if generate_summaries and len(content) > 100:
            print(f"  [{idx+1}/{len(doc_sections)}] Summarizing: {title[:50]}...")
            summary = generate_summary(
                title=title,
                content=content,
                document_context=markdown_text[:5000],
                model=model,
            )

        sections.append(Section(
            title=title,
            start_index=start_idx,
            end_index=end_idx,
            summary=summary,
            content=content,
        ))

    # Build final document
    doc = StructuredDocument(
        source_path=str(source_path),
        markdown_path=markdown_path,
        total_lines=len(lines),
        total_chars=len(markdown_text),
        sections=sections,
    )

    print(f"\n{'='*60}")
    print(f"Index complete: {len(sections)} sections")
    for i, s in enumerate(sections):
        print(f"  {i+1}. [{s.start_index}-{s.end_index}] {s.title[:60]}")
    print(f"{'='*60}\n")

    return doc


def build_and_save_index(
    source_path: str,
    output_dir: Optional[str] = None,
    **kwargs,
) -> tuple[StructuredDocument, str]:
    """Build and save index to JSON file."""
    doc = build_index(source_path, output_dir, **kwargs)

    source = Path(source_path)
    if output_dir is None:
        out_dir = source.parent
    else:
        out_dir = Path(output_dir)

    index_path = str(out_dir / f"{source.stem}.index.json")
    doc.save(index_path)
    print(f"Index saved: {index_path}")

    return doc, index_path


def build_index_from_text(
    text: str,
    source_name: str = "raw_text",
    model: str = SUB_MODEL,
    generate_summaries: bool = True,
) -> StructuredDocument:
    """
    Build complete structured index from raw text string.

    Skips file conversion step - useful for benchmarks or in-memory documents.

    Args:
        text: Raw text content (markdown/plain text)
        source_name: Name to identify this document
        model: Model for segmentation/summaries
        generate_summaries: Whether to generate contextual summaries

    Returns:
        StructuredDocument with sections and summaries
    """
    lines = text.split('\n')
    total_lines = len(lines)

    print(f"\n{'='*60}")
    print(f"Building index from text: {source_name}")
    print(f"{'='*60}")
    print(f"  Size: {total_lines} lines, {len(text):,} chars")

    # Step 1: Segment document
    print(f"\nStep 1: Segmenting document...")
    doc_sections = segment_document(text, model=model)
    print(f"  Result: {len(doc_sections)} sections found")

    # Step 2: Build sections with content and summaries
    print(f"\nStep 2: Building section content and summaries...")
    sections = []

    for idx, sec_data in enumerate(doc_sections):
        title = sec_data.title
        start_idx = max(0, sec_data.line_range.start)
        end_idx = min(len(lines) - 1, sec_data.line_range.end)

        # Extract content
        content = '\n'.join(lines[start_idx:end_idx + 1])

        # Generate summary if requested
        summary = sec_data.description or ""
        if generate_summaries and len(content) > 100:
            print(f"  [{idx+1}/{len(doc_sections)}] Summarizing: {title[:50]}...")
            summary = generate_summary(
                title=title,
                content=content,
                document_context=text[:5000],
                model=model,
            )

        sections.append(Section(
            title=title,
            start_index=start_idx,
            end_index=end_idx,
            summary=summary,
            content=content,
        ))

    # Build final document
    doc = StructuredDocument(
        source_path=source_name,
        markdown_path=source_name,  # No markdown file for raw text
        total_lines=total_lines,
        total_chars=len(text),
        sections=sections,
    )

    print(f"\n{'='*60}")
    print(f"Index complete: {len(sections)} sections")
    for i, s in enumerate(sections):
        print(f"  {i+1}. [{s.start_index}-{s.end_index}] {s.title[:60]}")
    print(f"{'='*60}\n")

    return doc


# Backward compatibility alias
SingleDocIndex = StructuredDocument
