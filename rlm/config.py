"""
RLM Configuration - Global settings with environment variable overrides.

Configuration precedence (highest to lowest):
1. Explicit arguments to functions/classes
2. Environment variables (RLM_ROOT_MODEL, etc.)
3. Default values defined here
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class RLMConfig:
    """
    Global configuration with environment variable overrides.

    Example:
        # Load from environment
        config = RLMConfig.from_env()

        # Or create with explicit values
        config = RLMConfig(
            root_model="anthropic/claude-sonnet-4.5",
            sub_model="openai/gpt-4o-mini",
            provider="openrouter"
        )
    """

    # Models
    root_model: str = "anthropic/claude-sonnet-4.5"
    sub_model: str = "openai/gpt-5-mini"

    # Provider
    provider: str = "openrouter"

    # API Keys (loaded from environment)
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Directories
    results_dir: str = "results"
    sessions_dir: str = "sessions"

    # Extraction settings
    max_iterations: int = 40
    parallel_workers: int = 5
    page_chunk_size: int = 5

    # Timeouts (seconds)
    api_timeout: int = 120
    retry_attempts: int = 3

    @classmethod
    def from_env(cls) -> "RLMConfig":
        """
        Load configuration from environment variables.

        Environment variables:
        - RLM_ROOT_MODEL: Override root model
        - RLM_SUB_MODEL: Override sub model
        - RLM_PROVIDER: Override provider (openrouter, openai, anthropic, ollama)
        - RLM_MAX_ITERATIONS: Override max iterations
        - RLM_PARALLEL_WORKERS: Override parallel workers
        - OPENROUTER_API_KEY: OpenRouter API key
        - OPENAI_API_KEY: OpenAI API key
        - ANTHROPIC_API_KEY: Anthropic API key
        """
        return cls(
            root_model=os.getenv("RLM_ROOT_MODEL", cls.root_model),
            sub_model=os.getenv("RLM_SUB_MODEL", cls.sub_model),
            provider=os.getenv("RLM_PROVIDER", cls.provider),
            max_iterations=int(os.getenv("RLM_MAX_ITERATIONS", cls.max_iterations)),
            parallel_workers=int(os.getenv("RLM_PARALLEL_WORKERS", cls.parallel_workers)),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )

    def get_api_key(self, provider: Optional[str] = None) -> Optional[str]:
        """Get API key for the specified or configured provider."""
        provider = provider or self.provider
        if provider == "openrouter":
            return self.openrouter_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        elif provider == "ollama":
            return None  # Ollama doesn't need API key
        return None

    def ensure_directories(self) -> None:
        """Create results and sessions directories if they don't exist."""
        Path(self.results_dir).mkdir(parents=True, exist_ok=True)
        Path(self.sessions_dir).mkdir(parents=True, exist_ok=True)


# Global default config instance
_default_config: Optional[RLMConfig] = None


def get_config() -> RLMConfig:
    """Get the global default configuration, loading from env if needed."""
    global _default_config
    if _default_config is None:
        _default_config = RLMConfig.from_env()
    return _default_config


def set_config(config: RLMConfig) -> None:
    """Set the global default configuration."""
    global _default_config
    _default_config = config
