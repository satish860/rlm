"""RLM - Recursive Language Model for long document processing."""

__version__ = "0.1.0"

from .single_doc import SingleDocRLM, query_document

__all__ = ["SingleDocRLM", "query_document"]
