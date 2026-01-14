"""REPL environment for single document RLM.

Provides a sandboxed execution environment where the root LLM can
explore documents through code generation.
"""

import io
import re
from contextlib import redirect_stdout
from pathlib import Path
from typing import Optional

import litellm

from .indexer import SingleDocIndex
from ..config import SUB_MODEL, CODE_EXECUTION_TIMEOUT


# =============================================================================
# Safe Builtins - Only allow safe operations
# =============================================================================

SAFE_BUILTINS = {
    # Types
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    # Iteration
    "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter,
    # Aggregation
    "len": len, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "reversed": reversed,
    "any": any, "all": all,
    # String
    "format": format, "repr": repr,
    # Boolean
    "True": True, "False": False, "None": None,
    # Errors (for try/except)
    "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
    "TypeError": TypeError, "IndexError": IndexError,
}
# Explicitly NOT included: open, exec, eval, import, __import__, compile, globals, locals


# =============================================================================
# REPL Executor
# =============================================================================

class REPLExecutor:
    """Sandboxed execution environment for document exploration."""

    def __init__(
        self,
        index: SingleDocIndex,
        sub_model: str = SUB_MODEL,
    ):
        """
        Initialize REPL executor.

        Args:
            index: Document index with sections and summaries
            sub_model: Model for sub-LLM calls
        """
        self.index = index
        self.sub_model = sub_model
        self.final_answer: Optional[str] = None
        self._markdown_content: Optional[str] = None

    def _load_markdown(self) -> str:
        """Load markdown content lazily."""
        if self._markdown_content is None:
            path = Path(self.index.markdown_path)
            if path.exists():
                self._markdown_content = path.read_text(encoding="utf-8")
            else:
                raise FileNotFoundError(f"Markdown file not found: {self.index.markdown_path}")
        return self._markdown_content

    # -------------------------------------------------------------------------
    # Navigation Functions (FREE - read from index)
    # -------------------------------------------------------------------------

    def _get_toc(self) -> list[dict]:
        """Return full TOC with hierarchy info."""
        return self.index.get_toc()

    def _get_section_names(self) -> list[str]:
        """Return list of section names."""
        return self.index.get_section_names()

    def _get_summary(self, section_name: str) -> str:
        """Return pre-computed summary for a section."""
        if section_name not in self.index.summaries:
            return f"No summary available for: {section_name}"
        return self.index.summaries[section_name]

    def _get_all_summaries(self) -> dict[str, str]:
        """Return all pre-computed summaries."""
        return self.index.summaries.copy()

    # -------------------------------------------------------------------------
    # Reading Functions (FREE - read from file)
    # -------------------------------------------------------------------------

    def _read_section(self, section_name: str) -> str:
        """Read full section content."""
        if section_name not in self.index.sections:
            return f"Section not found: {section_name}"

        section = self.index.sections[section_name]
        content = self._load_markdown()
        return content[section.start_char:section.end_char]

    def _read_section_chunk(
        self,
        section_name: str,
        chunk_idx: int = 0,
        chunk_size: int = 10000,
    ) -> str:
        """Read a chunk of a section."""
        if section_name not in self.index.sections:
            return f"Section not found: {section_name}"

        section = self.index.sections[section_name]
        content = self._load_markdown()
        section_content = content[section.start_char:section.end_char]

        start = chunk_idx * chunk_size
        end = start + chunk_size

        if start >= len(section_content):
            return f"Chunk {chunk_idx} is beyond section end"

        return section_content[start:end]

    def _read_range(self, start: int, end: int) -> str:
        """Read raw character range from document."""
        content = self._load_markdown()
        return content[start:end]

    # -------------------------------------------------------------------------
    # Search Functions (FREE - regex/index lookup)
    # -------------------------------------------------------------------------

    def _grep_section(self, pattern: str, section_name: str) -> list[str]:
        """Regex search within a specific section."""
        if section_name not in self.index.sections:
            return [f"Section not found: {section_name}"]

        section_content = self._read_section(section_name)
        try:
            matches = re.findall(pattern, section_content, re.IGNORECASE)
            return matches if matches else []
        except re.error as e:
            return [f"Regex error: {e}"]

    def _grep_all(self, pattern: str) -> dict[str, list[str]]:
        """Regex search across all sections."""
        results = {}
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error as e:
            return {"error": [f"Regex error: {e}"]}

        for section_name in self.index.sections:
            section_content = self._read_section(section_name)
            matches = compiled.findall(section_content)
            if matches:
                results[section_name] = matches

        return results

    def _find_sections_by_keyword(self, keyword: str) -> list[str]:
        """Find sections containing a keyword (from index)."""
        keyword_lower = keyword.lower()
        matching_sections = []

        for section_name, keywords in self.index.keywords.items():
            if any(keyword_lower in kw.lower() for kw in keywords):
                matching_sections.append(section_name)

        return matching_sections

    # -------------------------------------------------------------------------
    # LLM Functions (COSTS MONEY - calls sub-LLM)
    # -------------------------------------------------------------------------

    def _llm_query(self, prompt: str) -> str:
        """Call sub-LLM with a custom prompt."""
        try:
            response = litellm.completion(
                model=self.sub_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"LLM error: {e}"

    def _ask_about_section(self, question: str, section_name: str) -> str:
        """Ask sub-LLM a question about a specific section."""
        if section_name not in self.index.sections:
            return f"Section not found: {section_name}"

        section_content = self._read_section(section_name)

        # Truncate if too long
        max_chars = 15000
        if len(section_content) > max_chars:
            section_content = section_content[:max_chars] + "\n...[truncated]"

        prompt = f"""Based on the following section content, answer the question.

Section: {section_name}
Content:
{section_content}

Question: {question}

Answer based only on the content above. If the answer is not in the content, say "Not found in this section"."""

        return self._llm_query(prompt)

    # -------------------------------------------------------------------------
    # Terminal Function
    # -------------------------------------------------------------------------

    def _final(self, answer: str) -> None:
        """Signal completion with final answer."""
        self.final_answer = str(answer)

    # -------------------------------------------------------------------------
    # Code Execution
    # -------------------------------------------------------------------------

    def execute_code(self, code: str) -> tuple[str, Optional[str]]:
        """
        Execute code in sandboxed environment.

        Args:
            code: Python code to execute

        Returns:
            Tuple of (stdout output, final answer or None)
        """
        # Reset final answer
        self.final_answer = None

        # Capture stdout
        stdout_buffer = io.StringIO()

        # Build globals with REPL functions
        repl_globals = {
            "__builtins__": SAFE_BUILTINS,
            # Navigation (FREE)
            "get_toc": self._get_toc,
            "get_section_names": self._get_section_names,
            "get_summary": self._get_summary,
            "get_all_summaries": self._get_all_summaries,
            # Reading (FREE)
            "read_section": self._read_section,
            "read_section_chunk": self._read_section_chunk,
            "read_range": self._read_range,
            # Search (FREE)
            "grep_section": self._grep_section,
            "grep_all": self._grep_all,
            "find_sections_by_keyword": self._find_sections_by_keyword,
            # LLM (COSTS MONEY)
            "llm_query": self._llm_query,
            "ask_about_section": self._ask_about_section,
            # Terminal
            "FINAL": self._final,
            # Override print to capture output
            "print": lambda *args, **kwargs: print(*args, file=stdout_buffer, **kwargs),
        }

        try:
            with redirect_stdout(stdout_buffer):
                exec(code, repl_globals)
        except Exception as e:
            stdout_buffer.write(f"\nError: {type(e).__name__}: {e}")

        output = stdout_buffer.getvalue()

        # Truncate output if too long
        max_output = 10000
        if len(output) > max_output:
            output = output[:max_output] + "\n...[output truncated]"

        return output, self.final_answer


# =============================================================================
# Tool Definitions for LLM
# =============================================================================

REPL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code to explore the document. Available functions: get_toc(), get_section_names(), get_summary(name), get_all_summaries(), read_section(name), read_section_chunk(name, idx, size), read_range(start, end), grep_section(pattern, name), grep_all(pattern), find_sections_by_keyword(kw), llm_query(prompt), ask_about_section(question, name). Use print() to see results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to output results."
                    }
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Call this when you have found the answer to the question. Only call after you have read relevant sections and verified the information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "The final answer to return to the user"
                    }
                },
                "required": ["answer"]
            }
        }
    }
]


# =============================================================================
# System Prompt Builder
# =============================================================================

def build_system_prompt(index: SingleDocIndex) -> str:
    """
    Build the system prompt for the root LLM.

    Args:
        index: Document index

    Returns:
        System prompt string
    """
    # Format TOC
    toc_lines = []
    for section in index.sections.values():
        indent = "  " * (section.level - 1)
        toc_lines.append(f"{indent}- {section.title}")
    toc_formatted = "\n".join(toc_lines)

    # Format summaries (just first 100 chars each)
    summary_lines = []
    for name, summary in index.summaries.items():
        short_summary = summary[:100] + "..." if len(summary) > 100 else summary
        summary_lines.append(f"- {name}: {short_summary}")
    summaries_formatted = "\n".join(summary_lines)

    return f"""You are a Recursive Language Model analyzing a document too large for context.
Use execute_code to write Python that explores the document.
Use final_answer when you have the answer.

## DOCUMENT: {index.source_path} ({index.total_chars:,} chars, {len(index.sections)} sections)

## SECTIONS
{toc_formatted}

## SUMMARIES
{summaries_formatted}

## FUNCTIONS

FREE (no cost):
- get_section_names() -> list[str]
- get_summary(name) -> str
- get_all_summaries() -> dict
- read_section(name) -> str
- grep_all(pattern) -> dict[str, list[str]]

RECURSIVE SUB-LLM CALLS (use for semantic understanding):
- ask_about_section(question, name) -> str  # Ask sub-LLM about a section
- llm_query(prompt) -> str                   # General sub-LLM query

## RECURSIVE STRATEGY

For SIMPLE queries (find specific facts):
1. grep_all() or read_section() to find info
2. final_answer

For COMPLEX queries (semantic understanding, aggregation):
1. Identify relevant sections from summaries
2. Use LOOPS to call ask_about_section() on EACH relevant section
3. Aggregate sub-LLM responses in your code
4. final_answer with aggregated result

EXAMPLE - Complex aggregation:
```python
results = []
for section in ["Methods", "Results", "Discussion"]:
    answer = ask_about_section("What metrics are reported?", section)
    results.append(f"{{section}}: {{answer}}")
print("\\n".join(results))
```

## RULES
- Use sub-LLM calls when you need UNDERSTANDING, not just retrieval
- Aggregate multiple sub-LLM responses programmatically
- Read before answering (no hallucination)
"""
