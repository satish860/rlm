"""
RLM Providers - LLM provider abstraction layer.

Supported providers:
- OpenRouter (default): Access multiple models via unified API
- OpenAI: Direct OpenAI API access
- Anthropic: Direct Anthropic Claude API access

Main functions:
- get_provider(): Get a provider instance by name
- register_provider(): Register a custom provider
- list_providers(): List available providers

Example:
    from rlm.providers import get_provider

    # Auto-detect provider from API keys
    provider = get_provider()

    # Get specific provider
    provider = get_provider("openai")

    # Chat completion
    response = provider.chat(
        messages=[{"role": "user", "content": "Hello!"}],
        model="gpt-4o-mini"
    )

    # Structured extraction
    from pydantic import BaseModel
    class Person(BaseModel):
        name: str
        age: int

    person = provider.extract(
        "Extract: John is 30 years old",
        response_model=Person,
        model="gpt-4o-mini"
    )
"""

from rlm.providers.base import BaseProvider, ChatResponse, retry_with_backoff
from rlm.providers.factory import (
    get_provider,
    register_provider,
    set_default_provider,
    list_providers,
    get_provider_class,
)

# Lazy imports for specific providers (avoid circular imports)
def __getattr__(name):
    if name == "OpenRouterProvider":
        from rlm.providers.openrouter import OpenRouterProvider
        return OpenRouterProvider
    elif name == "OpenAIProvider":
        from rlm.providers.openai import OpenAIProvider
        return OpenAIProvider
    elif name == "AnthropicProvider":
        from rlm.providers.anthropic import AnthropicProvider
        return AnthropicProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Base classes
    "BaseProvider",
    "ChatResponse",
    "retry_with_backoff",
    # Factory functions
    "get_provider",
    "register_provider",
    "set_default_provider",
    "list_providers",
    "get_provider_class",
    # Provider classes (lazy loaded)
    "OpenRouterProvider",
    "OpenAIProvider",
    "AnthropicProvider",
]
