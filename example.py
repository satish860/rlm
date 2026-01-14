"""Example usage of Single-Doc RLM.

Usage:
    python example.py <document_path> <question>

Examples:
    python example.py recursive_language_models.pdf "What benchmarks were used?"
    python example.py myreport.pdf "What are the main conclusions?"
"""

import sys
sys.path.insert(0, '.')

from src import SingleDocRLM


def main():
    if len(sys.argv) < 3:
        print("Usage: python example.py <document_path> <question>")
        print("Example: python example.py recursive_language_models.pdf \"What benchmarks were used?\"")
        sys.exit(1)

    doc_path = sys.argv[1]
    question = sys.argv[2]

    print(f"Document: {doc_path}")
    print(f"Question: {question}")
    print()

    # Create RLM instance
    rlm = SingleDocRLM(doc_path)

    # Check for existing index
    index_path = f"temp/{doc_path.replace('.pdf', '').replace('.docx', '')}.index.json"

    try:
        rlm.load_index(index_path)
        print(f"Loaded existing index: {index_path}")
    except FileNotFoundError:
        print("Building index (this may take a minute)...")
        rlm.build_index(output_dir="./temp")
        rlm.save_index(index_path)

    print(f"Sections: {len(rlm.index.sections)}")
    print()

    # Query
    print("Querying...")
    print("-" * 60)
    answer = rlm.query(question, verbose=True)
    print("-" * 60)
    print()
    print("ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()
