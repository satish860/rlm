"""
RLM Tool Definitions - Tools available to the root model.

Defines the OpenAI-compatible tool schemas for:
- execute_code: Run Python in REPL
- final_answer: Return extraction results
- final_answer_file: Return results from saved file
"""

from typing import List, Dict, Any


# Tool definitions for root model
TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in REPL. Variables persist across calls. Use print() to see output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return extracted data with schema and verification. Use final_answer_file if data already saved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {
                        "type": "array",
                        "description": "Extracted records (skip if using final_answer_file)"
                    },
                    "schema": {
                        "type": "object",
                        "description": "Schema describing the extracted data structure",
                        "properties": {
                            "document_type": {"type": "string"},
                            "record_type": {"type": "string"},
                            "fields": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "verification": {
                        "type": "object",
                        "properties": {
                            "total_records": {"type": "integer"},
                            "pages_processed": {"type": "array"},
                            "categories_found": {"type": "object"},
                            "sample_verbatim_check": {"type": "boolean"},
                            "verification_passed": {"type": "boolean"},
                            "notes": {"type": "string"}
                        }
                    }
                },
                "required": ["data", "schema", "verification"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer_file",
            "description": "Return results when data is already saved via save_output(). Just pass the filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "Filename in results/ folder where data was saved"
                    },
                    "schema": {
                        "type": "object",
                        "description": "Schema describing the extracted data structure",
                        "properties": {
                            "document_type": {"type": "string"},
                            "record_type": {"type": "string"},
                            "fields": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "verification": {
                        "type": "object",
                        "properties": {
                            "total_records": {"type": "integer"},
                            "pages_processed": {"type": "array"},
                            "categories_found": {"type": "object"},
                            "verification_passed": {"type": "boolean"},
                            "notes": {"type": "string"}
                        }
                    }
                },
                "required": ["filename", "schema", "verification"]
            }
        }
    }
]


def get_tool_names() -> List[str]:
    """Get list of available tool names."""
    return [t["function"]["name"] for t in TOOLS]


def get_tool_by_name(name: str) -> Dict[str, Any]:
    """Get tool definition by name."""
    for tool in TOOLS:
        if tool["function"]["name"] == name:
            return tool
    raise ValueError(f"Unknown tool: {name}")
