"""
RLM Provider Factory - Create and manage LLM provider instances.

This module provides a unified way to get provider instances by name,
with support for custom provider registration.

Example:
    from rlm.providers import get_provider, register_provider

    # Get default provider (OpenRouter)
    provider = get_provider()

    # Get specific provider
    provider = get_provider("openai")
    provider = get_provider("anthropic")

    # Register custom provider
    register_provider("custom", MyCustomProvider)
"""

import os
from typing import Dict, Type, Optional

from rlm.exceptions import ProviderError
from rlm.providers.base import BaseProvider


# Registry of available providers
_providers: Dict[str, Type[BaseProvider]] = {}

# Default provider name
_default_provider: str = "openrouter"


def _ensure_providers_registered():
    """Lazy register built-in providers to avoid circular imports."""
    if not _providers:
        from rlm.providers.openrouter import OpenRouterProvider
        from rlm.providers.openai import OpenAIProvider
        from rlm.providers.anthropic import AnthropicProvider

        _providers["openrouter"] = OpenRouterProvider
        _providers["openai"] = OpenAIProvider
        _providers["anthropic"] = AnthropicProvider


def register_provider(name: str, provider_class: Type[BaseProvider]) -> None:
    """
    Register a custom provider.

    Args:
        name: Provider name (e.g., "ollama", "azure")
        provider_class: Provider class implementing BaseProvider

    Example:
        class OllamaProvider(BaseProvider):
            ...

        register_provider("ollama", OllamaProvider)
    """
    _providers[name] = provider_class


def get_provider(name: str = None, **kwargs) -> BaseProvider:
    """
    Get a provider instance by name.

    Args:
        name: Provider name. If None, uses default or auto-detects
              from available API keys.
        **kwargs: Arguments passed to provider constructor

    Returns:
        Provider instance

    Raises:
        ProviderError: If provider not found or no API key available

    Example:
        # Get default provider
        provider = get_provider()

        # Get specific provider
        provider = get_provider("openai")

        # With explicit API key
        provider = get_provider("openai", api_key="sk-...")
    """
    _ensure_providers_registered()

    # Auto-detect provider if not specified
    if name is None:
        name = _detect_provider()

    name = name.lower()

    if name not in _providers:
        available = ", ".join(_providers.keys())
        raise ProviderError(
            f"Unknown provider: {name}. Available providers: {available}"
        )

    return _providers[name](**kwargs)


def _detect_provider() -> str:
    """
    Auto-detect provider based on available API keys.

    Returns provider name based on which API key is set,
    in order of preference.
    """
    # Check environment variable for explicit default
    explicit = os.getenv("RLM_PROVIDER")
    if explicit:
        return explicit.lower()

    # Check for API keys in preference order
    if os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"

    # Fall back to default (will error if no key)
    return _default_provider


def set_default_provider(name: str) -> None:
    """
    Set the default provider name.

    Args:
        name: Provider name to use as default
    """
    global _default_provider
    _ensure_providers_registered()
    if name not in _providers:
        raise ProviderError(f"Unknown provider: {name}")
    _default_provider = name


def list_providers() -> list:
    """
    List all registered provider names.

    Returns:
        List of provider names
    """
    _ensure_providers_registered()
    return list(_providers.keys())


def get_provider_class(name: str) -> Type[BaseProvider]:
    """
    Get provider class by name (without instantiating).

    Args:
        name: Provider name

    Returns:
        Provider class
    """
    _ensure_providers_registered()
    if name not in _providers:
        raise ProviderError(f"Unknown provider: {name}")
    return _providers[name]
