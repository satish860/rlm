"""
RLM Library Example - Extract contacts from Agribusiness Companies PDF.

This uses the actual test file in the repo:
  40255083-Agribusiness-Companies.pdf

Run from project root:
  python examples/run_extraction.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import rlm


def main():
    # Use the real PDF in the repo
    pdf_path = Path(__file__).parent.parent / "40255083-Agribusiness-Companies.pdf"

    if not pdf_path.exists():
        print(f"Error: Test file not found: {pdf_path}")
        print("Make sure you're running from the project root.")
        return

    print(f"Extracting contacts from: {pdf_path.name}")
    print("=" * 60)

    # Extract using built-in Contact schema
    result = rlm.extract(
        str(pdf_path),
        schema=rlm.schemas.Contact,
        verbose=True,
        max_iterations=30
    )

    # Show results
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(result.data)} contacts extracted")
    print(f"Iterations: {result.iterations}")
    print("=" * 60)

    for i, contact in enumerate(result.data[:10], 1):  # First 10
        print(f"\n[{i}] {contact.get('name', 'N/A')}")
        if contact.get('company'):
            print(f"    Company: {contact['company']}")
        if contact.get('phone'):
            print(f"    Phone: {contact['phone']}")
        if contact.get('email'):
            print(f"    Email: {contact['email']}")
        if contact.get('address'):
            print(f"    Address: {contact['address'][:50]}...")
        print(f"    Page: {contact.get('page', 'N/A')}")

    if len(result.data) > 10:
        print(f"\n... and {len(result.data) - 10} more contacts")

    # Show citations
    if result.citations:
        print(f"\nCitations ({len(result.citations)} evidence snippets):")
        for cite in result.citations[:3]:
            snippet = cite.snippet if hasattr(cite, 'snippet') else cite.get('snippet', '')
            page = cite.page if hasattr(cite, 'page') else cite.get('page', 0)
            print(f"  Page {page}: \"{snippet[:60]}...\"")

    # Generate visualization
    output_path = Path(__file__).parent / "contacts_report.html"
    rlm.visualize(result, output=str(output_path))
    print(f"\nVisualization saved to: {output_path}")


if __name__ == "__main__":
    main()
