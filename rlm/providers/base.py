"""
RLM Base Provider - Abstract interface for LLM providers.

This module defines the contract that all LLM providers must implement.
Includes retry logic with exponential backoff for transient errors.

Example:
    class MyProvider(BaseProvider):
        def chat(self, messages, model, tools=None):
            # Implement chat completion
            ...

        def extract(self, prompt, response_model, model):
            # Implement structured extraction
            ...
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type, Optional, Callable
from functools import wraps

from pydantic import BaseModel

from rlm.exceptions import ProviderError


logger = logging.getLogger(__name__)


def retry_with_backoff(
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    retryable_exceptions: tuple = None
):
    """
    Decorator for retry with exponential backoff.

    Args:
        max_attempts: Maximum retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for each retry
        retryable_exceptions: Tuple of exceptions to retry on
    """
    if retryable_exceptions is None:
        retryable_exceptions = (ProviderError,)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        # Check if it's a rate limit or transient error
                        error_str = str(e).lower()
                        is_retryable = any(x in error_str for x in [
                            'rate limit', 'timeout', 'connection',
                            '429', '503', '502', '504', 'overloaded'
                        ])

                        if is_retryable:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                                f"Retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                            delay = min(delay * backoff_factor, max_delay)
                        else:
                            # Non-retryable error, raise immediately
                            raise
                    else:
                        raise

            raise last_exception

        return wrapper
    return decorator


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.

    All providers must implement:
    - chat(): Send chat completion request with optional tools
    - extract(): Extract structured data using Pydantic model

    Providers handle:
    - API authentication
    - Request formatting for their specific API
    - Response parsing
    - Error handling and retries
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name (e.g., 'openrouter', 'openai')."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model identifier (e.g., 'gpt-4o-mini')
            tools: Optional list of tool definitions
            **kwargs: Provider-specific options

        Returns:
            Response dict with 'choices' containing message and tool_calls

        Raises:
            ProviderError: If the request fails
        """
        pass

    @abstractmethod
    def extract(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str,
        **kwargs
    ) -> BaseModel:
        """
        Extract structured data using a Pydantic model.

        Uses instructor library for reliable structured extraction.

        Args:
            prompt: The extraction prompt
            response_model: Pydantic model class or List[Model]
            model: Model identifier
            **kwargs: Provider-specific options

        Returns:
            Instance of response_model populated with extracted data

        Raises:
            ProviderError: If extraction fails
        """
        pass

    def validate_api_key(self, api_key: Optional[str], env_var: str) -> str:
        """
        Validate that an API key is available.

        Args:
            api_key: Explicitly provided API key
            env_var: Environment variable name to check

        Returns:
            The API key

        Raises:
            ProviderError: If no API key available
        """
        import os
        key = api_key or os.getenv(env_var)
        if not key:
            raise ProviderError(
                f"No API key provided. Set {env_var} environment variable "
                f"or pass api_key parameter."
            )
        return key


class ChatResponse:
    """
    Normalized chat response wrapper.

    Provides consistent access to response data across different providers.
    """

    def __init__(self, raw_response: Dict[str, Any]):
        self.raw = raw_response
        self._message = None
        self._tool_calls = None

    @property
    def message(self) -> Optional[Dict[str, Any]]:
        """Get the assistant message from the response."""
        if self._message is None:
            choices = self.raw.get('choices', [])
            if choices:
                self._message = choices[0].get('message', {})
        return self._message

    @property
    def content(self) -> Optional[str]:
        """Get the text content of the response."""
        msg = self.message
        return msg.get('content') if msg else None

    @property
    def tool_calls(self) -> List[Dict[str, Any]]:
        """Get tool calls from the response."""
        if self._tool_calls is None:
            msg = self.message
            self._tool_calls = msg.get('tool_calls', []) if msg else []
        return self._tool_calls

    @property
    def has_tool_calls(self) -> bool:
        """Check if response contains tool calls."""
        return len(self.tool_calls) > 0

    @property
    def usage(self) -> Dict[str, int]:
        """Get token usage information."""
        return self.raw.get('usage', {})
