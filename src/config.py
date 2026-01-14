"""Configuration for RLM.

All models are accessed via OpenRouter + litellm.
Set OPENROUTER_API_KEY environment variable before use.
"""

# Model configuration via OpenRouter
# See available models at: https://openrouter.ai/models

# Root LLM - generates code to explore documents (needs to be smart)
ROOT_MODEL = "openrouter/anthropic/claude-sonnet-4"

# Sub LLM - handles semantic tasks like summaries (can be cheaper)
SUB_MODEL = "openrouter/google/gemini-3-flash-preview"

# TOC Model - lightweight model for TOC fallback when regex fails
TOC_MODEL = "openrouter/google/gemini-2.0-flash-001"

# Alternative models (uncomment to use)
# ROOT_MODEL = "openrouter/anthropic/claude-sonnet-4"
# SUB_MODEL = "openrouter/anthropic/claude-haiku"
# TOC_MODEL = "openrouter/google/gemini-flash-1.5"

# Execution limits
MAX_REPL_ROUNDS = 10  # Max iterations in query loop
CODE_EXECUTION_TIMEOUT = 30  # Seconds per code execution

# Context limits
MAX_SECTION_CHARS_FOR_SUMMARY = 10000  # Truncate sections longer than this for summarization
MAX_DOC_CHARS_FOR_CONTEXT = 100000  # Max doc chars to pass for contextual retrieval
