"""Prompt Builders for Extract and Summarize.

Build prompts for Stage 1 (query) based on the extraction/summary requirements.
"""

from typing import Type, get_origin, get_args
from pydantic import BaseModel


def build_extraction_prompt(
    what: str,
    response_model: Type[BaseModel],
) -> str:
    """
    Build a prompt for extraction that instructs the LLM to find all items.

    Creates a detailed prompt that tells the LLM what to extract and
    what fields to include for each item.

    Args:
        what: Description of what to extract (e.g., "papers", "figures")
        response_model: Pydantic model defining the expected structure

    Returns:
        Formatted prompt string for query()

    Example:
        prompt = build_extraction_prompt("papers", PaperList)
        # Returns:
        # List ALL papers in this document.
        #
        # For each item, include:
        # - title: Paper title
        # - authors: List of author names
        # ...
    """
    # Get field information from the model
    fields_desc = _extract_fields_description(response_model)

    prompt = f"""List ALL {what} in this document.

For each item, include:
{fields_desc}

IMPORTANT INSTRUCTIONS:
1. Be EXHAUSTIVE - list every single item, do not summarize or skip any.
2. Format as a numbered list with all details for each item.
3. If a field is not available, note it as "N/A" or "Not specified".
4. In your final_answer, include the COMPLETE numbered list of all items found.
   Do NOT leave the final answer empty - it must contain all the extracted items."""

    return prompt


def _extract_fields_description(model: Type[BaseModel]) -> str:
    """Extract field descriptions from a Pydantic model."""
    lines = []

    # Check if this is a list wrapper (like PaperList with items field)
    if "items" in model.model_fields:
        items_field = model.model_fields["items"]
        annotation = items_field.annotation

        # Extract the item type from list[ItemType]
        if get_origin(annotation) is list:
            args = get_args(annotation)
            if args and hasattr(args[0], "model_fields"):
                item_model = args[0]
                for name, field_info in item_model.model_fields.items():
                    desc = field_info.description or name
                    lines.append(f"- {name}: {desc}")
                return "\n".join(lines)

    # Fallback: use fields directly from the model
    for name, field_info in model.model_fields.items():
        desc = field_info.description or name
        lines.append(f"- {name}: {desc}")

    return "\n".join(lines)


# =============================================================================
# Summary Prompts
# =============================================================================

SUMMARY_STYLE_TEMPLATES = {
    "paragraph": "Write a clear, flowing paragraph summary.",
    "bullets": "Write a bullet-point summary with key points.",
    "executive": "Write an executive summary focusing on key findings and implications.",
    "abstract": "Write an academic abstract with background, methods, results, and conclusions.",
}


def build_summary_prompt(
    scope: str,
    style: str,
    max_length: int,
) -> str:
    """
    Build a prompt for summarization.

    Args:
        scope: What to summarize ("document", "section:NAME", "sections:A,B,C")
        style: Summary style ("paragraph", "bullets", "executive", "abstract")
        max_length: Target word count

    Returns:
        Formatted prompt string for query()
    """
    style_guidance = SUMMARY_STYLE_TEMPLATES.get(style, SUMMARY_STYLE_TEMPLATES["paragraph"])

    # Parse scope
    if scope == "document":
        scope_text = "the entire document"
    elif scope.startswith("section:"):
        section_name = scope[8:]
        scope_text = f"the section '{section_name}'"
    elif scope.startswith("sections:"):
        section_names = scope[9:].split(",")
        section_list = ", ".join([f"'{s.strip()}'" for s in section_names])
        scope_text = f"the following sections: {section_list}"
    else:
        scope_text = scope

    prompt = f"""Summarize {scope_text}.

Target length: approximately {max_length} words.

Style: {style_guidance}

Include:
- Main contribution or thesis
- Key methods or approach
- Important results or findings
- Conclusions or implications

Be concise but comprehensive. Do not exceed the target length significantly."""

    return prompt
