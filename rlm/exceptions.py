"""
RLM Exceptions - Custom exception hierarchy for the RLM library.

All exceptions inherit from RLMError for easy catching of library-specific errors.
"""


class RLMError(Exception):
    """
    Base exception for all RLM errors.

    All RLM-specific exceptions inherit from this class, making it easy
    to catch any RLM error with a single except clause.

    Example:
        try:
            result = rlm.extract("doc.pdf", schema=MySchema)
        except RLMError as e:
            print(f"RLM error: {e}")
    """
    pass


class DocumentError(RLMError):
    """
    Error reading or parsing a document.

    Raised when:
    - File does not exist
    - File format is unsupported
    - PDF is corrupted or encrypted
    - Encoding issues with text files
    """
    pass


class ProviderError(RLMError):
    """
    Error communicating with LLM provider.

    Raised when:
    - API key is missing or invalid
    - Rate limit exceeded (after retries)
    - Network/connection error
    - Provider returns unexpected response
    """
    pass


class ExtractionError(RLMError):
    """
    Error during the extraction process.

    Raised when:
    - Max iterations reached without final answer
    - Code execution fails repeatedly
    - Root model fails to produce valid tool calls
    """
    pass


class SchemaError(RLMError):
    """
    Invalid or incompatible schema.

    Raised when:
    - Schema is not a valid Pydantic model
    - Schema missing required 'page' field
    - Schema validation fails
    """
    pass


class SessionError(RLMError):
    """
    Error saving or loading session.

    Raised when:
    - Session file not found
    - Session file corrupted
    - Session version incompatible
    """
    pass
