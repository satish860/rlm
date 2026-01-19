"""
RLM OpenRouter Provider - Access multiple LLM models via OpenRouter API.

OpenRouter provides unified access to models from OpenAI, Anthropic, Google,
Meta, and others through a single API.

Example:
    from rlm.providers.openrouter import OpenRouterProvider

    provider = OpenRouterProvider()  # Uses OPENROUTER_API_KEY env var

    # Chat completion
    response = provider.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        model="openai/gpt-4o-mini"
    )

    # Structured extraction
    from pydantic import BaseModel
    class Person(BaseModel):
        name: str
        age: int

    person = provider.extract(
        "John is 30 years old",
        response_model=Person,
        model="openai/gpt-4o-mini"
    )
"""

import os
from typing import List, Dict, Any, Type, Optional

from pydantic import BaseModel

from rlm.exceptions import ProviderError
from rlm.providers.base import BaseProvider, retry_with_backoff


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter API provider.

    Provides access to multiple LLM models through OpenRouter's unified API.
    Supports both chat completions and structured extraction via instructor.

    Models available include:
    - openai/gpt-4o, openai/gpt-4o-mini
    - anthropic/claude-sonnet-4, anthropic/claude-sonnet-4.5
    - google/gemini-pro, meta-llama/llama-3-70b
    - And many more: https://openrouter.ai/models
    """

    BASE_URL = "https://openrouter.ai/api/v1"
    ENV_VAR = "OPENROUTER_API_KEY"

    def __init__(self, api_key: str = None):
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key. If not provided, reads from
                    OPENROUTER_API_KEY environment variable.
        """
        self.api_key = self.validate_api_key(api_key, self.ENV_VAR)
        self._client = None
        self._instructor_client = None

    @property
    def name(self) -> str:
        return "openrouter"

    def _get_client(self):
        """Lazy load OpenAI client configured for OpenRouter."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai"
                )
            self._client = OpenAI(
                base_url=self.BASE_URL,
                api_key=self.api_key
            )
        return self._client

    def _get_instructor_client(self):
        """Lazy load instructor-wrapped client for structured extraction."""
        if self._instructor_client is None:
            try:
                import instructor
            except ImportError:
                raise ProviderError(
                    "instructor package not installed. Install with: pip install instructor"
                )
            self._instructor_client = instructor.from_openai(self._get_client())
        return self._instructor_client

    @retry_with_backoff(max_attempts=3, initial_delay=2.0)
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[Dict] = None,
        timeout: float = 120.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request to OpenRouter.

        Args:
            messages: List of message dicts
            model: Model identifier (e.g., 'openai/gpt-4o-mini')
            tools: Optional tool definitions
            timeout: Request timeout in seconds
            **kwargs: Additional parameters passed to API

        Returns:
            Raw API response as dict

        Raises:
            ProviderError: If request fails
        """
        client = self._get_client()

        try:
            params = {
                "model": model,
                "messages": messages,
                "timeout": timeout,
                **kwargs
            }
            if tools:
                params["tools"] = tools

            response = client.chat.completions.create(**params)
            return response.model_dump()

        except Exception as e:
            error_msg = str(e)
            # Check for specific error types
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(f"Rate limit exceeded: {e}")
            elif "timeout" in error_msg.lower():
                raise ProviderError(f"Request timeout: {e}")
            elif "api key" in error_msg.lower() or "401" in error_msg:
                raise ProviderError(f"Invalid API key: {e}")
            else:
                raise ProviderError(f"OpenRouter API error: {e}")

    @retry_with_backoff(max_attempts=3, initial_delay=2.0)
    def extract(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str,
        timeout: float = 120.0,
        **kwargs
    ) -> BaseModel:
        """
        Extract structured data using instructor.

        Args:
            prompt: The extraction prompt with context
            response_model: Pydantic model class (or List[Model])
            model: Model identifier
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Returns:
            Instance of response_model with extracted data

        Raises:
            ProviderError: If extraction fails
        """
        client = self._get_instructor_client()

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                timeout=timeout,
                **kwargs
            )
            return response

        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(f"Rate limit exceeded: {e}")
            elif "timeout" in error_msg.lower():
                raise ProviderError(f"Request timeout: {e}")
            elif "validation" in error_msg.lower():
                raise ProviderError(f"Schema validation failed: {e}")
            else:
                raise ProviderError(f"Extraction failed: {e}")
