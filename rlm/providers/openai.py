"""
RLM OpenAI Provider - Direct access to OpenAI API.

Use this provider for direct OpenAI API access without going through
OpenRouter. Useful for organizations with direct OpenAI contracts.

Example:
    from rlm.providers.openai import OpenAIProvider

    provider = OpenAIProvider()  # Uses OPENAI_API_KEY env var

    response = provider.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4o-mini"
    )
"""

import os
from typing import List, Dict, Any, Type, Optional

from pydantic import BaseModel

from rlm.exceptions import ProviderError
from rlm.providers.base import BaseProvider, retry_with_backoff


class OpenAIProvider(BaseProvider):
    """
    Direct OpenAI API provider.

    Provides direct access to OpenAI models without routing through OpenRouter.

    Available models:
    - gpt-4o, gpt-4o-mini
    - gpt-4-turbo, gpt-4
    - gpt-3.5-turbo
    - o1, o1-mini (reasoning models)
    """

    ENV_VAR = "OPENAI_API_KEY"

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key. If not provided, reads from
                    OPENAI_API_KEY environment variable.
            base_url: Optional custom base URL (for Azure or proxies)
        """
        self.api_key = self.validate_api_key(api_key, self.ENV_VAR)
        self.base_url = base_url
        self._client = None
        self._instructor_client = None

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Install with: pip install openai"
                )
            kwargs = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def _get_instructor_client(self):
        """Lazy load instructor-wrapped client."""
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
        Send chat completion request to OpenAI.

        Args:
            messages: List of message dicts
            model: Model identifier (e.g., 'gpt-4o-mini')
            tools: Optional tool definitions
            timeout: Request timeout in seconds
            **kwargs: Additional parameters

        Returns:
            Raw API response as dict
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
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(f"Rate limit exceeded: {e}")
            elif "timeout" in error_msg.lower():
                raise ProviderError(f"Request timeout: {e}")
            elif "api key" in error_msg.lower() or "401" in error_msg:
                raise ProviderError(f"Invalid API key: {e}")
            else:
                raise ProviderError(f"OpenAI API error: {e}")

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
            prompt: The extraction prompt
            response_model: Pydantic model class
            model: Model identifier
            timeout: Request timeout

        Returns:
            Instance of response_model
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
            if "rate limit" in error_msg.lower():
                raise ProviderError(f"Rate limit exceeded: {e}")
            elif "validation" in error_msg.lower():
                raise ProviderError(f"Schema validation failed: {e}")
            else:
                raise ProviderError(f"Extraction failed: {e}")
