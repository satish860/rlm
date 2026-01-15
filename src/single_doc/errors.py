"""Custom exceptions for Single Document RLM extraction and summarization."""


class ExtractionError(Exception):
    """
    Raised when structured data extraction fails.

    This error occurs when the extract() method cannot successfully
    extract data from the document, either due to query failures
    or JSON conversion issues.
    """
    pass


class SchemaGenerationError(Exception):
    """
    Raised when automatic schema generation fails.

    This error occurs in Stage 0 when the LLM cannot generate
    a valid Pydantic schema from a plain English description.
    """
    pass


class JSONConversionError(Exception):
    """
    Raised when Instructor fails to convert text to JSON.

    This error occurs in Stage 2 when the free-text output
    from query() cannot be validated against the Pydantic model,
    even after automatic retries.
    """
    pass


class UserCancelledError(Exception):
    """
    Raised when the user cancels an operation.

    This error occurs when the user rejects a generated schema
    during the confirmation step in plain English extraction mode.
    """
    pass
