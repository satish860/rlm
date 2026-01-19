"""
RLM Core - Engine, REPL, and orchestration components.

Main classes:
- RLMEngine: Main extraction engine
- REPLEnvironment: Code execution environment
- Citation, ExtractionResult, QueryResult: Result types
"""

from rlm.core.types import Citation, ExtractionResult, QueryResult
from rlm.core.engine import RLMEngine
from rlm.core.repl import REPLEnvironment
from rlm.core.tools import TOOLS, get_tool_names, get_tool_by_name
from rlm.core.prompts import build_system_prompt, build_query_prompt, build_user_message

__all__ = [
    # Types
    "Citation",
    "ExtractionResult",
    "QueryResult",
    # Engine
    "RLMEngine",
    "REPLEnvironment",
    # Tools
    "TOOLS",
    "get_tool_names",
    "get_tool_by_name",
    # Prompts
    "build_system_prompt",
    "build_query_prompt",
    "build_user_message",
]
