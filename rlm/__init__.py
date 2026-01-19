"""
RLM - Recursive Language Model Library

A production-ready Python library for intelligent document extraction
using the "root model + sub-LLM" architecture with explicit reasoning.

Basic Usage:
    import rlm
    from pydantic import BaseModel

    class Contact(BaseModel):
        name: str
        email: str = None
        phone: str = None
        page: int

    result = rlm.extract("contacts.pdf", schema=Contact)
    for contact in result.data:
        print(contact["name"], contact["email"])

    # Q&A mode
    answer = rlm.query("report.pdf", "What was Q3 revenue?")
    print(answer.answer)

Advanced Usage:
    from rlm import RLMEngine

    engine = RLMEngine(
        root_model="anthropic/claude-sonnet-4.5",
        sub_model="openai/gpt-4o-mini",
        provider="openrouter"
    )
    result = engine.extract("document.pdf", schema=MySchema)
"""

__version__ = "0.1.0"

from typing import Type, Optional

from pydantic import BaseModel

from rlm.core.types import ExtractionResult, QueryResult, Citation
from rlm.core.engine import RLMEngine
from rlm.config import RLMConfig
from rlm import schemas


def extract(
    document: str,
    schema: Type[BaseModel] = None,
    *,
    verbose: bool = False,
    max_iterations: int = 40,
    root_model: str = None,
    sub_model: str = None,
    provider: str = None
) -> ExtractionResult:
    """
    Extract structured data from a document.

    Args:
        document: Path to document file (PDF, Markdown, or text)
        schema: Optional Pydantic model for extraction target
        verbose: Print detailed progress
        max_iterations: Maximum root model iterations
        root_model: Override root model (e.g., "anthropic/claude-sonnet-4.5")
        sub_model: Override sub model (e.g., "openai/gpt-4o-mini")
        provider: Override provider (e.g., "openrouter", "openai", "anthropic")

    Returns:
        ExtractionResult with extracted data, citations, and metadata

    Example:
        from pydantic import BaseModel
        import rlm

        class Contact(BaseModel):
            name: str
            phone: str = None
            email: str = None
            page: int

        result = rlm.extract("directory.pdf", schema=Contact)
        print(f"Found {len(result.data)} contacts")
        for contact in result.data:
            print(contact["name"], contact.get("phone"))
    """
    engine = RLMEngine(
        root_model=root_model,
        sub_model=sub_model,
        provider=provider
    )
    return engine.extract(
        document=document,
        schema=schema,
        max_iterations=max_iterations,
        verbose=verbose
    )


def query(
    document: str,
    question: str,
    *,
    verbose: bool = False,
    max_iterations: int = 20,
    root_model: str = None,
    sub_model: str = None,
    provider: str = None
) -> QueryResult:
    """
    Ask a question about a document.

    Args:
        document: Path to document file (PDF, Markdown, or text)
        question: Question to answer
        verbose: Print detailed progress
        max_iterations: Maximum root model iterations
        root_model: Override root model
        sub_model: Override sub model
        provider: Override provider

    Returns:
        QueryResult with answer, citations, and confidence

    Example:
        import rlm

        result = rlm.query("report.pdf", "What was Q3 revenue?")
        print(result.answer)
        print(f"Confidence: {result.confidence}")
        for citation in result.citations:
            print(f"  Page {citation.page}: {citation.text[:50]}...")
    """
    engine = RLMEngine(
        root_model=root_model,
        sub_model=sub_model,
        provider=provider
    )
    return engine.query(
        document=document,
        question=question,
        max_iterations=max_iterations,
        verbose=verbose
    )


def visualize(
    result: ExtractionResult,
    output: str = None,
    document_text: str = None,
    open_browser: bool = False
) -> str:
    """
    Generate HTML visualization of extraction results.

    Args:
        result: ExtractionResult from extract()
        output: Output file path (if None, returns HTML string)
        document_text: Optional document text for citation highlighting
        open_browser: Open result in browser after generation

    Returns:
        HTML string if output is None, otherwise output path

    Example:
        import rlm

        result = rlm.extract("contacts.pdf", schema=Contact)

        # Save to file
        rlm.visualize(result, output="report.html")

        # Get HTML string
        html = rlm.visualize(result)

        # Open in browser
        rlm.visualize(result, output="report.html", open_browser=True)
    """
    from rlm.visualization.html import generate_html_report

    html_content = generate_html_report(
        result,
        document_text=document_text,
        include_document=document_text is not None
    )

    if output:
        from pathlib import Path
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(f"file://{output_path.absolute()}")

        return str(output_path)

    return html_content


def interactive():
    """Launch interactive CLI session."""
    from rlm.interactive import main as interactive_main
    interactive_main()


__all__ = [
    # Functions
    "extract",
    "query",
    "visualize",
    "interactive",
    # Classes
    "RLMEngine",
    "RLMConfig",
    # Result types
    "ExtractionResult",
    "QueryResult",
    "Citation",
    # Schemas
    "schemas",
    # Version
    "__version__",
]
