"""Document indexer for single document RLM.

Handles TOC parsing, section mapping, and contextual summary generation.
"""

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import litellm

from ..config import TOC_MODEL, SUB_MODEL, MAX_SECTION_CHARS_FOR_SUMMARY, MAX_DOC_CHARS_FOR_CONTEXT


@dataclass
class Section:
    """A document section with hierarchy info."""
    title: str              # "3.2 Model Architecture"
    level: int              # Heading level (1-6)
    parent: Optional[str]   # Parent section title or None for top-level
    start_char: int         # Start offset in markdown
    end_char: int           # End offset in markdown

    def __post_init__(self):
        """Validate section data."""
        if self.end_char < self.start_char:
            raise ValueError(f"end_char ({self.end_char}) must be >= start_char ({self.start_char})")


@dataclass
class SingleDocIndex:
    """Document index containing structure and summaries."""

    source_path: str                           # Original document path
    markdown_path: str                         # Path to converted markdown
    total_chars: int                           # Total markdown size
    sections: dict[str, Section] = field(default_factory=dict)  # title -> Section
    summaries: dict[str, str] = field(default_factory=dict)     # title -> contextual summary
    keywords: dict[str, list[str]] = field(default_factory=dict)  # title -> keywords

    def get_toc(self) -> list[dict]:
        """Return TOC as list of dicts."""
        return [
            {
                "title": s.title,
                "level": s.level,
                "parent": s.parent,
                "start_char": s.start_char,
                "end_char": s.end_char,
            }
            for s in self.sections.values()
        ]

    def get_section_names(self) -> list[str]:
        """Return list of section titles."""
        return list(self.sections.keys())

    def to_json(self) -> str:
        """Serialize to JSON string."""
        data = {
            "source_path": self.source_path,
            "markdown_path": self.markdown_path,
            "total_chars": self.total_chars,
            "sections": {
                title: asdict(section)
                for title, section in self.sections.items()
            },
            "summaries": self.summaries,
            "keywords": self.keywords,
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "SingleDocIndex":
        """Deserialize from JSON string."""
        data = json.loads(json_str)
        sections = {
            title: Section(**section_data)
            for title, section_data in data["sections"].items()
        }
        return cls(
            source_path=data["source_path"],
            markdown_path=data["markdown_path"],
            total_chars=data["total_chars"],
            sections=sections,
            summaries=data.get("summaries", {}),
            keywords=data.get("keywords", {}),
        )

    def save(self, path: str) -> None:
        """Save index to file."""
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "SingleDocIndex":
        """Load index from file."""
        json_str = Path(path).read_text(encoding="utf-8")
        return cls.from_json(json_str)


# =============================================================================
# TOC Parsing - Regex
# =============================================================================

# Regex pattern for markdown headings: # Title, ## Title, etc.
HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def parse_toc_regex(markdown_text: str) -> list[Section]:
    """
    Parse TOC from markdown using regex.

    Extracts headings (#, ##, ###, etc.) and maps them to sections
    with character offsets.

    Args:
        markdown_text: Markdown content

    Returns:
        List of Section objects (empty if no headings found)
    """
    sections = []

    # Find all headings
    matches = list(HEADING_PATTERN.finditer(markdown_text))

    if not matches:
        return []

    # Build section list with start positions
    for i, match in enumerate(matches):
        level = len(match.group(1))  # Number of # characters
        title = match.group(2).strip()
        start_char = match.start()

        # End is start of next section, or end of document
        if i + 1 < len(matches):
            end_char = matches[i + 1].start()
        else:
            end_char = len(markdown_text)

        sections.append(Section(
            title=title,
            level=level,
            parent=None,  # Will be filled in by _assign_parents
            start_char=start_char,
            end_char=end_char,
        ))

    # Assign parent relationships based on hierarchy
    _assign_parents(sections)

    return sections


def _assign_parents(sections: list[Section]) -> None:
    """
    Assign parent sections based on heading levels.

    A section's parent is the most recent section with a lower level number.
    Example: ### is child of ##, which is child of #

    Modifies sections in place.
    """
    # Stack of (level, title) for tracking hierarchy
    stack: list[tuple[int, str]] = []

    for section in sections:
        # Pop sections from stack that are same level or deeper
        while stack and stack[-1][0] >= section.level:
            stack.pop()

        # Parent is top of stack (if any)
        if stack:
            section.parent = stack[-1][1]
        else:
            section.parent = None

        # Push current section onto stack
        stack.append((section.level, section.title))


# =============================================================================
# TOC Parsing - LLM Fallback
# =============================================================================

def parse_toc_llm(
    markdown_text: str,
    model: str = TOC_MODEL,
) -> list[Section]:
    """
    Parse TOC using lightweight LLM when regex finds no headings.

    Asks the LLM to identify major sections/chapters in the document.

    Args:
        markdown_text: Markdown content
        model: LLM model to use (default: cheap/fast model)

    Returns:
        List of Section objects (empty if LLM can't identify structure)
    """
    # Use first portion of document for context
    preview = markdown_text[:MAX_DOC_CHARS_FOR_CONTEXT]

    prompt = f"""Analyze this document and identify its major sections or chapters.

<document>
{preview}
</document>

Return a JSON array of sections with this format:
[
  {{"title": "Section Title", "line_number": 1}},
  {{"title": "Another Section", "line_number": 50}}
]

Rules:
- Only include major sections (not every paragraph)
- line_number is the approximate line where the section starts
- If the document has no clear sections, return an empty array []
- Return ONLY the JSON array, no other text

JSON:"""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0,
        )

        content = response.choices[0].message.content.strip()

        # Try to parse JSON from response
        # Handle cases where LLM adds extra text
        if "```" in content:
            # Extract from code block
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)

        sections_data = json.loads(content)

        if not sections_data:
            return []

        # Convert line numbers to character offsets
        lines = markdown_text.split('\n')
        line_to_char = _build_line_to_char_map(lines)

        sections = []
        for i, sec in enumerate(sections_data):
            title = sec.get("title", f"Section {i+1}")
            line_num = sec.get("line_number", 1)

            # Convert line number to char offset
            start_char = line_to_char.get(line_num, 0)

            # End is start of next section or end of doc
            if i + 1 < len(sections_data):
                next_line = sections_data[i + 1].get("line_number", len(lines))
                end_char = line_to_char.get(next_line, len(markdown_text))
            else:
                end_char = len(markdown_text)

            sections.append(Section(
                title=title,
                level=1,  # LLM-identified sections are all level 1
                parent=None,
                start_char=start_char,
                end_char=end_char,
            ))

        return sections

    except Exception as e:
        # If LLM fails, return empty list (will fall back to single section)
        print(f"LLM TOC parsing failed: {e}")
        return []


def _build_line_to_char_map(lines: list[str]) -> dict[int, int]:
    """Build mapping from line number (1-indexed) to character offset."""
    line_to_char = {1: 0}
    char_pos = 0

    for i, line in enumerate(lines):
        char_pos += len(line) + 1  # +1 for newline
        line_to_char[i + 2] = char_pos  # Next line starts here

    return line_to_char


# =============================================================================
# TOC Parsing - Tiered (Regex -> LLM -> Single Section)
# =============================================================================

def parse_toc(
    markdown_text: str,
    use_llm_fallback: bool = True,
    llm_model: str = TOC_MODEL,
) -> list[Section]:
    """
    Parse TOC using tiered approach.

    Tier 1: Regex parsing of markdown headings (free, instant)
    Tier 2: LLM fallback if no headings found (cheap, ~2 sec)
    Tier 3: Single section fallback (entire document as one section)

    Args:
        markdown_text: Markdown content
        use_llm_fallback: Whether to try LLM if regex fails
        llm_model: Model for LLM fallback

    Returns:
        List of Section objects (always at least one section)
    """
    # Tier 1: Try regex
    sections = parse_toc_regex(markdown_text)
    if sections:
        return sections

    # Tier 2: Try LLM fallback
    if use_llm_fallback:
        sections = parse_toc_llm(markdown_text, model=llm_model)
        if sections:
            return sections

    # Tier 3: Single section fallback
    return [Section(
        title="Document",
        level=1,
        parent=None,
        start_char=0,
        end_char=len(markdown_text),
    )]


# =============================================================================
# Contextual Summary Generation (Anthropic's Contextual Retrieval)
# =============================================================================

def generate_section_context(
    section: Section,
    markdown_text: str,
    model: str = SUB_MODEL,
) -> str:
    """
    Generate contextual summary for a section using Anthropic's approach.

    Uses the full document context + specific chunk to generate a summary
    that situates the chunk within the broader document.

    Args:
        section: Section to summarize
        markdown_text: Full markdown document
        model: LLM model for summarization

    Returns:
        Contextual summary string
    """
    # Get section content
    section_content = markdown_text[section.start_char:section.end_char]

    # Truncate if too long
    if len(section_content) > MAX_SECTION_CHARS_FOR_SUMMARY:
        section_content = section_content[:MAX_SECTION_CHARS_FOR_SUMMARY] + "\n...[truncated]"

    # Get document context (truncated if needed)
    doc_context = markdown_text[:MAX_DOC_CHARS_FOR_CONTEXT]
    if len(markdown_text) > MAX_DOC_CHARS_FOR_CONTEXT:
        doc_context += "\n...[document truncated]"

    # Anthropic's Contextual Retrieval prompt
    prompt = f"""<document>
{doc_context}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{section_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Summary generation failed for '{section.title}': {e}")
        return f"Section: {section.title}"


def extract_keywords(
    section: Section,
    markdown_text: str,
    model: str = SUB_MODEL,
) -> list[str]:
    """
    Extract keywords from a section for search indexing.

    Args:
        section: Section to extract keywords from
        markdown_text: Full markdown document
        model: LLM model for extraction

    Returns:
        List of keywords
    """
    section_content = markdown_text[section.start_char:section.end_char]

    # Truncate if too long
    if len(section_content) > MAX_SECTION_CHARS_FOR_SUMMARY:
        section_content = section_content[:MAX_SECTION_CHARS_FOR_SUMMARY]

    prompt = f"""Extract 3-7 important keywords or key phrases from this section.
Return ONLY a JSON array of strings, nothing else.

Section Title: {section.title}

Content:
{section_content}

Keywords (JSON array):"""

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        content = response.choices[0].message.content.strip()

        # Parse JSON
        if "```" in content:
            match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', content)
            if match:
                content = match.group(1)

        keywords = json.loads(content)
        if isinstance(keywords, list):
            return [str(k) for k in keywords[:7]]
        return []
    except Exception as e:
        print(f"Keyword extraction failed for '{section.title}': {e}")
        return []


def generate_all_summaries(
    sections: list[Section],
    markdown_text: str,
    model: str = SUB_MODEL,
    include_keywords: bool = True,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """
    Generate contextual summaries for all sections.

    Args:
        sections: List of sections to summarize
        markdown_text: Full markdown document
        model: LLM model for summarization
        include_keywords: Whether to also extract keywords

    Returns:
        Tuple of (summaries dict, keywords dict)
    """
    summaries = {}
    keywords = {}

    for section in sections:
        print(f"  Summarizing: {section.title}")

        # Generate contextual summary
        summaries[section.title] = generate_section_context(
            section, markdown_text, model
        )

        # Extract keywords if requested
        if include_keywords:
            keywords[section.title] = extract_keywords(
                section, markdown_text, model
            )

    return summaries, keywords


# =============================================================================
# Full Indexer Pipeline
# =============================================================================

def build_index(
    source_path: str,
    output_dir: Optional[str] = None,
    generate_summaries: bool = True,
    use_llm_toc_fallback: bool = True,
    toc_model: str = TOC_MODEL,
    summary_model: str = SUB_MODEL,
) -> SingleDocIndex:
    """
    Build complete index for a document.

    Full pipeline:
    1. Convert document to markdown
    2. Parse TOC (regex -> LLM -> single section)
    3. Generate contextual summaries (optional)
    4. Extract keywords (optional)

    Args:
        source_path: Path to source document (PDF, DOCX, etc.)
        output_dir: Directory for output files (default: same as source)
        generate_summaries: Whether to generate summaries via LLM
        use_llm_toc_fallback: Whether to use LLM if regex TOC fails
        toc_model: Model for TOC fallback
        summary_model: Model for summaries and keywords

    Returns:
        SingleDocIndex with all metadata
    """
    from .converter import DocumentConverter

    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"Source document not found: {source_path}")

    # Set output directory
    if output_dir is None:
        output_dir = source.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert to markdown
    print(f"Converting: {source.name}")
    converter = DocumentConverter()
    markdown_path = str(output_dir / f"{source.stem}.md")
    markdown_text, markdown_path = converter.convert(source_path, markdown_path)
    print(f"  -> {markdown_path} ({len(markdown_text):,} chars)")

    # Step 2: Parse TOC
    print("Parsing TOC...")
    sections = parse_toc(
        markdown_text,
        use_llm_fallback=use_llm_toc_fallback,
        llm_model=toc_model,
    )
    print(f"  Found {len(sections)} sections")

    # Build sections dict
    sections_dict = {s.title: s for s in sections}

    # Step 3: Generate summaries (optional)
    summaries = {}
    keywords = {}

    if generate_summaries and len(sections) > 0:
        print("Generating contextual summaries...")
        summaries, keywords = generate_all_summaries(
            sections,
            markdown_text,
            model=summary_model,
            include_keywords=True,
        )
        print(f"  Generated {len(summaries)} summaries")

    # Build index
    index = SingleDocIndex(
        source_path=str(source_path),
        markdown_path=markdown_path,
        total_chars=len(markdown_text),
        sections=sections_dict,
        summaries=summaries,
        keywords=keywords,
    )

    return index


def build_and_save_index(
    source_path: str,
    output_dir: Optional[str] = None,
    **kwargs,
) -> tuple[SingleDocIndex, str]:
    """
    Build index and save to JSON file.

    Args:
        source_path: Path to source document
        output_dir: Directory for output files
        **kwargs: Additional arguments for build_index

    Returns:
        Tuple of (index, index_path)
    """
    index = build_index(source_path, output_dir, **kwargs)

    # Save index
    source = Path(source_path)
    if output_dir is None:
        output_dir = source.parent
    else:
        output_dir = Path(output_dir)

    index_path = str(output_dir / f"{source.stem}.index.json")
    index.save(index_path)
    print(f"Index saved: {index_path}")

    return index, index_path
