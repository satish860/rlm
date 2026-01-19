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

    result = rlm.extract("contacts.pdf", schema=list[Contact])
    for contact in result.data:
        print(contact.name, contact.email)

    # Q&A mode
    answer = rlm.query("report.pdf", "What was Q3 revenue?")
    print(answer.answer)
"""

__version__ = "0.1.0"

# Public API - will be implemented in T-022
from rlm.core.types import ExtractionResult, QueryResult, Citation
from rlm import schemas

# These will be implemented later
def extract(document, schema, *, verbose=False, max_iterations=40,
            root_model=None, sub_model=None, provider=None):
    """Extract structured data from a document."""
    raise NotImplementedError("extract() will be implemented in T-022")


def query(document, question, *, verbose=False):
    """Ask a question about a document."""
    raise NotImplementedError("query() will be implemented in T-022")


def visualize(result, output=None, open_browser=False):
    """Generate HTML visualization of extraction results."""
    raise NotImplementedError("visualize() will be implemented in T-029")


# RLMEngine will be implemented in T-021
class RLMEngine:
    """Advanced API for full control over extraction."""

    def __init__(self, root_model=None, sub_model=None, provider=None,
                 results_dir=None, sessions_dir=None):
        raise NotImplementedError("RLMEngine will be implemented in T-021")


__all__ = [
    "extract",
    "query",
    "visualize",
    "RLMEngine",
    "ExtractionResult",
    "QueryResult",
    "Citation",
    "schemas",
    "__version__",
]
