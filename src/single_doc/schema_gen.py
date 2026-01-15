"""Schema Generation from Plain English.

Stage 0 of the extraction pipeline: For users who describe what they want
in plain English instead of providing a Pydantic model, this module
generates a schema using LLM inference.
"""

import os
from typing import Type, Optional, get_origin, get_args

import instructor
from openai import OpenAI
from pydantic import BaseModel, Field, create_model

from .errors import SchemaGenerationError, UserCancelledError
from ..config import JSON_MODEL


def _get_instructor_client():
    """Create Instructor client using OpenRouter."""
    openai_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    return instructor.from_openai(openai_client)


def _strip_provider_prefix(model: str) -> str:
    """Strip provider prefix from model name."""
    if model.startswith("openrouter/"):
        return model[len("openrouter/"):]
    return model


# =============================================================================
# Internal Models for Schema Definition
# =============================================================================

class FieldDef(BaseModel):
    """Definition of a single field in the schema."""
    name: str = Field(description="Field name (snake_case)")
    type: str = Field(description="Type: str, int, float, bool, list[str], list[int]")
    required: bool = Field(description="Whether the field is required")
    description: str = Field(description="Brief description of the field")


class SchemaDefinition(BaseModel):
    """LLM-generated schema definition."""
    item_name: str = Field(description="Name for the item type (e.g., Paper, Figure)")
    fields: list[FieldDef] = Field(description="List of fields to extract")


# =============================================================================
# Type Mapping
# =============================================================================

TYPE_MAP = {
    "str": str,
    "string": str,
    "int": int,
    "integer": int,
    "float": float,
    "bool": bool,
    "boolean": bool,
    "list[str]": list[str],
    "list[string]": list[str],
    "list[int]": list[int],
    "list[integer]": list[int],
    "list[float]": list[float],
}


# =============================================================================
# Schema Generation
# =============================================================================

def generate_schema_from_description(
    what: str,
    model: str = JSON_MODEL,
) -> Type[BaseModel]:
    """
    Generate a Pydantic model from a plain English description.

    This is Stage 0 of the extraction pipeline. Users describe what they
    want to extract in plain English, and this function uses an LLM to
    infer the appropriate schema.

    Args:
        what: Plain English description (e.g., "papers with title, authors, year, url")
        model: LLM model for schema inference

    Returns:
        Dynamically created Pydantic model class (a List wrapper)

    Raises:
        SchemaGenerationError: If schema generation fails

    Example:
        schema = generate_schema_from_description("papers with title, authors, year")
        # Returns a PaperList model with Paper items containing those fields
    """
    client = _get_instructor_client()
    model_name = _strip_provider_prefix(model)

    try:
        schema_def = client.chat.completions.create(
            model=model_name,
            response_model=SchemaDefinition,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a schema generator. Given a description of what to extract, "
                        "define a schema with appropriate fields.\n\n"
                        "Rules:\n"
                        "- Use snake_case for field names\n"
                        "- Choose appropriate types: str, int, float, bool, list[str], list[int]\n"
                        "- Mark fields as required=True if they are essential\n"
                        "- Mark fields as required=False if they might not always be present\n"
                        "- Use list[str] for fields that can have multiple values (e.g., authors)\n"
                        "- Keep the item_name simple and singular (e.g., 'Paper' not 'Papers')"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Define a schema for extracting: {what}",
                },
            ],
            max_retries=2,
            temperature=0,
        )
    except Exception as e:
        raise SchemaGenerationError(f"Failed to generate schema: {e}")

    # Build the Pydantic model dynamically
    try:
        fields = {}
        for f in schema_def.fields:
            field_type = TYPE_MAP.get(f.type.lower(), str)

            if f.required:
                fields[f.name] = (field_type, Field(description=f.description))
            else:
                fields[f.name] = (
                    Optional[field_type],
                    Field(default=None, description=f.description),
                )

        # Create the item model
        ItemModel = create_model(schema_def.item_name, **fields)

        # Create the list wrapper model
        ListModel = create_model(
            f"{schema_def.item_name}List",
            items=(list[ItemModel], Field(description=f"List of {schema_def.item_name} items")),
        )

        return ListModel

    except Exception as e:
        raise SchemaGenerationError(f"Failed to create Pydantic model: {e}")


# =============================================================================
# Schema Display
# =============================================================================

def format_schema_for_display(schema: Type[BaseModel]) -> str:
    """
    Format a Pydantic model for human-readable display.

    Shows field names, types, and whether they are required/optional
    in a clean format without Pydantic internals.

    Args:
        schema: Pydantic model class to format

    Returns:
        Human-readable schema string

    Example:
        text = format_schema_for_display(PaperList)
        # Returns:
        # Paper:
        #   - title: str (required)
        #   - authors: list[str] (required)
        #   - year: str (required)
        #   - url: str (optional)
    """
    lines = []

    # Get the inner item model from the list wrapper
    if "items" in schema.model_fields:
        items_field = schema.model_fields["items"]
        annotation = items_field.annotation

        # Extract the item type from list[ItemType]
        if get_origin(annotation) is list:
            args = get_args(annotation)
            if args and hasattr(args[0], "model_fields"):
                item_model = args[0]
                lines.append(f"{item_model.__name__}:")

                for name, field_info in item_model.model_fields.items():
                    field_type = _format_type(field_info.annotation)
                    required = field_info.is_required()
                    status = "required" if required else "optional"
                    lines.append(f"  - {name}: {field_type} ({status})")

                return "\n".join(lines)

    # Fallback: display the schema directly
    lines.append(f"{schema.__name__}:")
    for name, field_info in schema.model_fields.items():
        field_type = _format_type(field_info.annotation)
        required = field_info.is_required()
        status = "required" if required else "optional"
        lines.append(f"  - {name}: {field_type} ({status})")

    return "\n".join(lines)


def _format_type(annotation) -> str:
    """Format a type annotation as a readable string."""
    if annotation is None:
        return "any"

    origin = get_origin(annotation)

    # Handle Optional types
    if origin is type(None):
        return "any"

    # Handle Union types (Optional is Union[T, None])
    if hasattr(annotation, "__origin__") and annotation.__origin__ is type(None):
        return "any"

    # Handle list types
    if origin is list:
        args = get_args(annotation)
        if args:
            inner = _format_type(args[0])
            return f"list[{inner}]"
        return "list"

    # Handle basic types
    if hasattr(annotation, "__name__"):
        return annotation.__name__

    # Fallback
    return str(annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")


# =============================================================================
# User Confirmation
# =============================================================================

def confirm_schema_with_user(
    schema: Type[BaseModel],
    what: str,
) -> bool:
    """
    Display generated schema and ask user for confirmation.

    Shows the schema in human-readable format and prompts the user
    to confirm before proceeding with extraction.

    Args:
        schema: Generated Pydantic model class
        what: Original description (for context)

    Returns:
        True if user confirms, False if user rejects

    Raises:
        UserCancelledError: If user explicitly cancels (Ctrl+C)

    Example:
        if confirm_schema_with_user(PaperList, "papers with title, authors"):
            # Proceed with extraction
        else:
            # User rejected, handle accordingly
    """
    print(f"\nGenerated schema for: {what}")
    print()
    print(format_schema_for_display(schema))
    print()

    try:
        response = input("Proceed with this schema? [Y/n]: ").strip().lower()

        if response in ("", "y", "yes"):
            return True
        else:
            return False

    except KeyboardInterrupt:
        print()
        raise UserCancelledError("User cancelled schema confirmation")
    except EOFError:
        raise UserCancelledError("User cancelled schema confirmation")
