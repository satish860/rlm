"""Test indexer functionality.

Run with: pytest tests/test_indexer.py -v
Or directly: python tests/test_indexer.py
"""

import os
from pathlib import Path

from src.single_doc.indexer import (
    Section,
    SingleDocIndex,
    parse_toc_regex,
    parse_toc,
    generate_section_context,
    extract_keywords,
    generate_all_summaries,
)
from src.config import SUB_MODEL


# Sample markdown for testing
SAMPLE_MD = """# Introduction

This is the introduction section. It provides an overview of the document.

## Background

The background section explains the context and history.

### Historical Context

Some historical information here.

## Methods

This section describes the methodology used.

# Results

The results of the study are presented here.

## Data Analysis

Analysis of the collected data.

# Conclusion

Final thoughts and summary.
"""


def test_parse_toc_regex():
    """Test regex-based TOC parsing."""
    sections = parse_toc_regex(SAMPLE_MD)

    assert len(sections) == 7

    # Check first section
    assert sections[0].title == "Introduction"
    assert sections[0].level == 1
    assert sections[0].parent is None

    # Check nested section
    background = next(s for s in sections if s.title == "Background")
    assert background.level == 2
    assert background.parent == "Introduction"

    # Check deeply nested
    historical = next(s for s in sections if s.title == "Historical Context")
    assert historical.level == 3
    assert historical.parent == "Background"

    print(f"Found {len(sections)} sections")
    for s in sections:
        indent = "  " * (s.level - 1)
        print(f"{indent}- {s.title} (L{s.level}, parent={s.parent})")


def test_single_doc_index_serialization():
    """Test index serialization to/from JSON."""
    sections = parse_toc_regex(SAMPLE_MD)

    index = SingleDocIndex(
        source_path="test.pdf",
        markdown_path="test.md",
        total_chars=len(SAMPLE_MD),
        sections={s.title: s for s in sections},
        summaries={"Introduction": "Overview of the document"},
        keywords={"Introduction": ["overview", "document"]},
    )

    # Serialize
    json_str = index.to_json()
    assert "test.pdf" in json_str
    assert "Introduction" in json_str

    # Deserialize
    loaded = SingleDocIndex.from_json(json_str)
    assert loaded.source_path == "test.pdf"
    assert len(loaded.sections) == 7
    assert loaded.summaries["Introduction"] == "Overview of the document"

    print("Serialization test passed")


def has_openrouter_key():
    """Check if OpenRouter API key is set."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def test_contextual_summary_generation():
    """Test contextual summary generation (requires API key)."""
    if not has_openrouter_key():
        print("Skipping: OPENROUTER_API_KEY not set")
        return

    sections = parse_toc_regex(SAMPLE_MD)
    intro = sections[0]

    print(f"Testing contextual summary for: {intro.title}")
    print(f"Using model: {SUB_MODEL}")

    summary = generate_section_context(intro, SAMPLE_MD, model=SUB_MODEL)

    print(f"Summary: {summary}")
    assert len(summary) > 10
    assert summary != f"Section: {intro.title}"  # Not the fallback


def test_keyword_extraction():
    """Test keyword extraction (requires API key)."""
    if not has_openrouter_key():
        print("Skipping: OPENROUTER_API_KEY not set")
        return

    sections = parse_toc_regex(SAMPLE_MD)
    intro = sections[0]

    print(f"Testing keyword extraction for: {intro.title}")
    print(f"Using model: {SUB_MODEL}")

    keywords = extract_keywords(intro, SAMPLE_MD, model=SUB_MODEL)

    print(f"Keywords: {keywords}")
    assert isinstance(keywords, list)


def quick_test():
    """Quick test function to run outside pytest."""
    print("Testing indexer functionality...")
    print()

    # Test regex parsing
    print("1. Testing regex TOC parsing...")
    test_parse_toc_regex()
    print()

    # Test serialization
    print("2. Testing index serialization...")
    test_single_doc_index_serialization()
    print()

    # Test with API if available
    if has_openrouter_key():
        print("3. Testing contextual summary generation...")
        test_contextual_summary_generation()
        print()

        print("4. Testing keyword extraction...")
        test_keyword_extraction()
        print()
    else:
        print("Skipping LLM tests: OPENROUTER_API_KEY not set")

    print("All tests passed!")


if __name__ == "__main__":
    quick_test()
