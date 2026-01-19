"""
RLM Library Example - Ask questions about the RLM paper.

This uses the actual test file in the repo:
  recursive_language_models.pdf

Run from project root:
  python examples/run_query.py
  python examples/run_query.py "What is the REPL environment?"
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import rlm


def main():
    # Use the RLM paper PDF
    pdf_path = Path(__file__).parent.parent / "recursive_language_models.pdf"

    if not pdf_path.exists():
        print(f"Error: Test file not found: {pdf_path}")
        return

    # Get question from args or use default
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What is the key insight of Recursive Language Models?"

    print(f"Document: {pdf_path.name}")
    print(f"Question: {question}")
    print("=" * 60)

    # Query the document
    result = rlm.query(
        str(pdf_path),
        question,
        verbose=True,
        max_iterations=20
    )

    # Show answer
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result.answer)
    print(f"\nConfidence: {result.confidence:.0%}")

    # Show citations
    if result.citations:
        print(f"\nEvidence ({len(result.citations)} citations):")
        for cite in result.citations:
            snippet = cite.snippet if hasattr(cite, 'snippet') else cite.get('snippet', '')
            page = cite.page if hasattr(cite, 'page') else cite.get('page', 0)
            note = cite.note if hasattr(cite, 'note') else cite.get('note', '')
            print(f"\n  Page {page}:")
            print(f"    \"{snippet[:100]}...\"")
            if note:
                print(f"    Note: {note}")


if __name__ == "__main__":
    main()
