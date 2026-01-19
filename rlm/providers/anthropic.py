"""
RLM Anthropic Provider - Direct access to Anthropic Claude API.

Use this provider for direct Anthropic API access.

Example:
    from rlm.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider()  # Uses ANTHROPIC_API_KEY env var

    response = provider.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        model="claude-sonnet-4-20250514"
    )
"""

import os
from typing import List, Dict, Any, Type, Optional

from pydantic import BaseModel

from rlm.exceptions import ProviderError
from rlm.providers.base import BaseProvider, retry_with_backoff


class AnthropicProvider(BaseProvider):
    """
    Direct Anthropic API provider.

    Provides direct access to Claude models.

    Available models:
    - claude-sonnet-4-20250514 (Claude Sonnet 4)
    - claude-opus-4-20250514 (Claude Opus 4)
    - claude-3-5-sonnet-20241022
    - claude-3-5-haiku-20241022
    - claude-3-opus-20240229
    """

    ENV_VAR = "ANTHROPIC_API_KEY"

    def __init__(self, api_key: str = None):
        """
        Initialize Anthropic provider.

        Args:
            api_key: Anthropic API key. If not provided, reads from
                    ANTHROPIC_API_KEY environment variable.
        """
        self.api_key = self.validate_api_key(api_key, self.ENV_VAR)
        self._client = None
        self._instructor_client = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_client(self):
        """Lazy load Anthropic client."""
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ProviderError(
                    "anthropic package not installed. Install with: pip install anthropic"
                )
            self._client = anthropic.Anthropic(api_key=self.api_key)
        return self._client

    def _get_instructor_client(self):
        """Lazy load instructor-wrapped client."""
        if self._instructor_client is None:
            try:
                import instructor
                import anthropic
            except ImportError:
                raise ProviderError(
                    "instructor and anthropic packages required. "
                    "Install with: pip install instructor anthropic"
                )
            self._instructor_client = instructor.from_anthropic(
                anthropic.Anthropic(api_key=self.api_key)
            )
        return self._instructor_client

    def _convert_tools_to_anthropic(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Anthropic format."""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools

    def _convert_messages_to_anthropic(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages to Anthropic format.

        Returns:
            Tuple of (system_prompt, messages_list)
        """
        system_prompt = None
        anthropic_messages = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "system":
                system_prompt = content
            elif role == "user":
                anthropic_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # Handle tool calls in assistant messages
                if msg.get("tool_calls"):
                    # Convert tool calls to Anthropic format
                    content_blocks = []
                    if content:
                        content_blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        content_blocks.append({
                            "type": "tool_use",
                            "id": tc["id"],
                            "name": tc["function"]["name"],
                            "input": tc["function"].get("arguments", {})
                        })
                    anthropic_messages.append({"role": "assistant", "content": content_blocks})
                else:
                    anthropic_messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Convert tool results
                anthropic_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.get("tool_call_id"),
                        "content": content
                    }]
                })

        return system_prompt, anthropic_messages

    def _convert_response_to_openai(self, response) -> Dict[str, Any]:
        """Convert Anthropic response to OpenAI-compatible format."""
        content = response.content[0] if response.content else None

        # Build OpenAI-style message
        message = {"role": "assistant"}

        if content:
            if hasattr(content, 'text'):
                message["content"] = content.text
            elif hasattr(content, 'type') and content.type == "tool_use":
                message["content"] = None
                message["tool_calls"] = [{
                    "id": content.id,
                    "type": "function",
                    "function": {
                        "name": content.name,
                        "arguments": str(content.input)
                    }
                }]

        # Handle multiple content blocks (text + tool_use)
        tool_calls = []
        text_content = []
        for block in response.content:
            if hasattr(block, 'text'):
                text_content.append(block.text)
            elif hasattr(block, 'type') and block.type == "tool_use":
                import json
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input) if isinstance(block.input, dict) else str(block.input)
                    }
                })

        if text_content:
            message["content"] = "\n".join(text_content)
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not text_content:
                message["content"] = None

        return {
            "choices": [{"message": message}],
            "usage": {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        }

    @retry_with_backoff(max_attempts=3, initial_delay=2.0)
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[Dict] = None,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send chat completion request to Anthropic.

        Args:
            messages: List of message dicts (OpenAI format, auto-converted)
            model: Model identifier (e.g., 'claude-sonnet-4-20250514')
            tools: Optional tool definitions (OpenAI format, auto-converted)
            timeout: Request timeout in seconds
            max_tokens: Maximum tokens in response
            **kwargs: Additional parameters

        Returns:
            OpenAI-compatible response dict
        """
        client = self._get_client()

        try:
            system_prompt, anthropic_messages = self._convert_messages_to_anthropic(messages)

            params = {
                "model": model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }

            if system_prompt:
                params["system"] = system_prompt

            if tools:
                params["tools"] = self._convert_tools_to_anthropic(tools)

            response = client.messages.create(**params)
            return self._convert_response_to_openai(response)

        except Exception as e:
            error_msg = str(e)
            if "rate limit" in error_msg.lower() or "429" in error_msg:
                raise ProviderError(f"Rate limit exceeded: {e}")
            elif "timeout" in error_msg.lower():
                raise ProviderError(f"Request timeout: {e}")
            elif "api key" in error_msg.lower() or "401" in error_msg:
                raise ProviderError(f"Invalid API key: {e}")
            else:
                raise ProviderError(f"Anthropic API error: {e}")

    @retry_with_backoff(max_attempts=3, initial_delay=2.0)
    def extract(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str,
        timeout: float = 120.0,
        max_tokens: int = 4096,
        **kwargs
    ) -> BaseModel:
        """
        Extract structured data using instructor.

        Args:
            prompt: The extraction prompt
            response_model: Pydantic model class
            model: Model identifier
            timeout: Request timeout
            max_tokens: Maximum tokens

        Returns:
            Instance of response_model
        """
        client = self._get_instructor_client()

        try:
            response = client.messages.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                max_tokens=max_tokens,
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
