"""
Quick Test - Use simple text file to verify library works.

Run from project root:
  python examples/quick_test.py
"""

import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import rlm
from pydantic import BaseModel
from typing import Optional


class SimpleContact(BaseModel):
    """Simple contact for testing."""
    company: str
    contact_name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    page: int


def test_extraction():
    """Test extraction with sample data."""
    sample_file = Path(__file__).parent / "sample_data.txt"

    print("=== Testing rlm.extract() ===")
    print(f"File: {sample_file.name}")

    result = rlm.extract(
        str(sample_file),
        schema=SimpleContact,
        verbose=True,
        max_iterations=20
    )

    print(f"\nExtracted {len(result.data)} contacts:")
    for contact in result.data:
        print(f"  - {contact.get('company')}: {contact.get('contact_name')} ({contact.get('email')})")

    return result


def test_query():
    """Test Q&A with sample data."""
    sample_file = Path(__file__).parent / "sample_data.txt"

    print("\n=== Testing rlm.query() ===")
    question = "What was the total revenue in 2024?"

    result = rlm.query(
        str(sample_file),
        question,
        verbose=True,
        max_iterations=15
    )

    print(f"\nQuestion: {question}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence:.0%}")

    return result


def test_visualization(result):
    """Test HTML visualization."""
    print("\n=== Testing rlm.visualize() ===")

    output_path = Path(__file__).parent / "test_report.html"
    rlm.visualize(result, output=str(output_path))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    print("RLM Library Quick Test")
    print("=" * 60)

    # Test extraction
    extraction_result = test_extraction()

    # Test query
    query_result = test_query()

    # Test visualization
    test_visualization(extraction_result)

    print("\n" + "=" * 60)
    print("All tests completed!")
