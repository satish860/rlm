"""JSON Conversion via Instructor.

Stage 2 of the extraction pipeline: Convert free-text output from query()
into validated JSON using Instructor + Pydantic models.
"""

import os
import re
from typing import Type, TypeVar

import instructor
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from .errors import JSONConversionError
from ..config import JSON_MODEL

T = TypeVar("T", bound=BaseModel)


def _get_instructor_client():
    """Create Instructor client using OpenRouter."""
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    return instructor.from_openai(openai_client)


def _strip_provider_prefix(model: str) -> str:
    """Strip provider prefix from model name (e.g., 'openrouter/google/gemini-3-flash' -> 'google/gemini-3-flash')."""
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


def convert_to_json(
    free_text: str,
    response_model: Type[T],
    model: str = JSON_MODEL,
    max_retries: int = 3,
) -> T:
    """
    Convert free text to validated Pydantic model using Instructor.

    This is Stage 2 of the extraction pipeline. Takes the free-text
    output from query() and converts it to a structured Pydantic model.

    Args:
        free_text: Text to convert (output from query())
        response_model: Target Pydantic model class
        model: LLM model for conversion (default: JSON_MODEL for cost efficiency)
        max_retries: Number of retry attempts on validation failure

    Returns:
        Populated Pydantic model instance

    Raises:
        JSONConversionError: If conversion fails after all retries

    Example:
        text = "1. Paper A by Smith, 2024\\n2. Paper B by Jones, 2023"
        result = convert_to_json(text, PaperList)
        # result.items = [Paper(...), Paper(...)]
    """
    client = _get_instructor_client()
    model_name = _strip_provider_prefix(model)

    try:
        result = client.chat.completions.create(
            model=model_name,
            response_model=response_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a data extraction expert. Your job is to parse the given text "
                        "and extract structured data from it.\n\n"
                        "IMPORTANT RULES:\n"
                        "1. Each item must be a complete object with all required fields filled in.\n"
                        "2. Parse author names, titles, years, and URLs from the raw text.\n"
                        "3. For authors, split them into a list of individual names.\n"
                        "4. Extract the publication year as a string (e.g., '2024').\n"
                        "5. If a URL is present, include it. Otherwise set to null.\n"
                        "6. Be thorough - extract EVERY item from the text."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Parse the following text and extract all items as structured data:\n\n{free_text}",
                },
            ],
            max_retries=max_retries,
            temperature=0,
        )
        return result

    except ValidationError as e:
        raise JSONConversionError(
            f"Failed to validate JSON after {max_retries} retries: {e}"
        )
    except Exception as e:
        raise JSONConversionError(f"JSON conversion failed: {e}")


def extract_partial_from_text(
    text: str,
    item_model: Type[T],
) -> list[T]:
    """
    Try to extract partial results from text when full conversion fails.

    This is a fallback when convert_to_json fails. It attempts to parse
    individual items from the text one at a time, returning whatever
    was successfully extracted.

    Args:
        text: Free text containing items to extract
        item_model: Pydantic model for individual items (e.g., Paper, not PaperList)

    Returns:
        List of successfully parsed items (may be empty)

    Example:
        text = "1. Valid Paper by Author, 2024\\n2. Malformed entry\\n3. Another Paper, 2023"
        papers = extract_partial_from_text(text, Paper)
        # Returns [Paper(...), Paper(...)] - skips malformed entry
    """
    # Split text into potential items (by numbered list, bullets, or paragraphs)
    patterns = [
        r'\d+\.\s+',  # Numbered list: "1. ", "2. "
        r'[-*]\s+',   # Bullet list: "- ", "* "
        r'\n\n+',     # Paragraph breaks
    ]

    items = []
    chunks = [text]

    # Try each pattern to split
    for pattern in patterns:
        split_chunks = []
        for chunk in chunks:
            parts = re.split(pattern, chunk)
            split_chunks.extend([p.strip() for p in parts if p.strip()])
        if len(split_chunks) > len(chunks):
            chunks = split_chunks

    # Create instructor client for individual item extraction
    client = _get_instructor_client()
    model_name = _strip_provider_prefix(JSON_MODEL)

    for chunk in chunks:
        if len(chunk) < 10:  # Skip very short chunks
            continue

        try:
            item = client.chat.completions.create(
                model=model_name,
                response_model=item_model,
                messages=[
                    {
                        "role": "system",
                        "content": "Extract the item information from this text. If not enough information, leave fields empty.",
                    },
                    {
                        "role": "user",
                        "content": chunk,
                    },
                ],
                max_retries=1,
                temperature=0,
            )
            items.append(item)
        except Exception:
            # Skip items that fail to parse
            continue

    return items
