"""Test LLM connections via litellm + OpenRouter.

Run with: pytest tests/test_llm_connection.py -v
Or directly: python tests/test_llm_connection.py

Requires: OPENROUTER_API_KEY environment variable
"""

import os
import pytest
import litellm
from src.utils.token_counter import TokenCounter


# Models via OpenRouter
TOC_MODEL = "openrouter/google/gemini-2.0-flash-001"  # Cheap model for TOC fallback
SUB_MODEL = "openrouter/openai/gpt-4o-mini"  # Sub-LLM for summaries
ROOT_MODEL = "openrouter/openai/gpt-4o"  # Root LLM for code generation


def has_openrouter_key():
    """Check if OpenRouter API key is set."""
    return bool(os.environ.get("OPENROUTER_API_KEY"))


def test_litellm_import():
    """Verify litellm is importable."""
    assert litellm is not None


@pytest.mark.skipif(not has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
def test_gemini_flash():
    """Test Gemini Flash model via OpenRouter (cheap TOC model)."""
    counter = TokenCounter()

    response = litellm.completion(
        model=TOC_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=10,
    )

    counter.record_response(response)
    content = response.choices[0].message.content

    print(f"Gemini response: {content}")
    print(f"Usage: {counter.summary()}")

    assert content is not None
    assert len(content) > 0


@pytest.mark.skipif(not has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
def test_gpt4o_mini():
    """Test GPT-4o-mini via OpenRouter (sub-LLM)."""
    counter = TokenCounter()

    response = litellm.completion(
        model=SUB_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=10,
    )

    counter.record_response(response)
    content = response.choices[0].message.content

    print(f"GPT-4o-mini response: {content}")
    print(f"Usage: {counter.summary()}")

    assert content is not None
    assert len(content) > 0


@pytest.mark.skipif(not has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
def test_gpt4o():
    """Test GPT-4o via OpenRouter (root LLM)."""
    counter = TokenCounter()

    response = litellm.completion(
        model=ROOT_MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else."}],
        max_tokens=10,
    )

    counter.record_response(response)
    content = response.choices[0].message.content

    print(f"GPT-4o response: {content}")
    print(f"Usage: {counter.summary()}")

    assert content is not None
    assert len(content) > 0


def quick_test():
    """Quick test function to run outside pytest."""
    print("Testing LLM connections via OpenRouter...")
    print()

    # Check environment variable
    has_key = has_openrouter_key()
    print(f"OPENROUTER_API_KEY: {'Set' if has_key else 'NOT SET'}")
    print()

    if not has_key:
        print("Please set OPENROUTER_API_KEY to test LLM connections.")
        print("Get your key at: https://openrouter.ai/keys")
        return

    counter = TokenCounter()

    # Test each model
    models = [
        (TOC_MODEL, "TOC model (Gemini Flash)"),
        (SUB_MODEL, "Sub-LLM (GPT-4o-mini)"),
    ]

    for model, description in models:
        print(f"Testing {description}...")
        print(f"  Model: {model}")
        try:
            response = litellm.completion(
                model=model,
                messages=[{"role": "user", "content": "Say 'test successful'"}],
                max_tokens=10,
            )
            counter.record_response(response)
            print(f"  Response: {response.choices[0].message.content}")
            print(f"  OK")
        except Exception as e:
            print(f"  Error: {e}")
        print()

    print(f"Total usage: {counter.summary()}")


if __name__ == "__main__":
    quick_test()
