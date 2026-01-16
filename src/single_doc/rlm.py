"""Single Document RLM - Main API.

Provides the main interface for querying long documents using
the Recursive Language Model approach.
"""

import json
from pathlib import Path
from typing import Optional, Type, Literal

import litellm
from pydantic import BaseModel

from .indexer import StructuredDocument, build_index, build_index_from_text
from .repl import REPLExecutor, REPL_TOOLS, build_system_prompt
from .models import PaperList, FigureList, StructuredSummary
from .errors import ExtractionError, SchemaGenerationError, UserCancelledError
from .json_convert import convert_to_json, extract_partial_from_text
from .schema_gen import generate_schema_from_description, confirm_schema_with_user
from .prompt_builder import build_extraction_prompt, build_summary_prompt
from ..config import ROOT_MODEL, SUB_MODEL, MAX_REPL_ROUNDS


class SingleDocRLM:
    """
    Main API for single document RLM.

    Usage:
        rlm = SingleDocRLM("path/to/document.pdf")
        rlm.build_index()
        answer = rlm.query("What are the main findings?")
    """

    def __init__(
        self,
        doc_path: str,
        root_model: str = ROOT_MODEL,
        sub_model: str = SUB_MODEL,
    ):
        """
        Initialize RLM for a document.

        Args:
            doc_path: Path to document (PDF/DOCX/HTML/TXT/MD)
            root_model: Model for code generation
            sub_model: Model for segmentation/summaries/semantic tasks
        """
        self.doc_path = doc_path
        self.root_model = root_model
        self.sub_model = sub_model

        self.index: Optional[StructuredDocument] = None
        self._executor: Optional[REPLExecutor] = None
        self._system_prompt: Optional[str] = None

    def build_index(
        self,
        output_dir: Optional[str] = None,
        generate_summaries: bool = True,
    ) -> None:
        """
        Build document index (sections, summaries).

        Args:
            output_dir: Directory for output files (default: same as source)
            generate_summaries: Whether to generate contextual summaries
        """
        self.index = build_index(
            source_path=self.doc_path,
            output_dir=output_dir,
            model=self.sub_model,
            generate_summaries=generate_summaries,
        )
        self._executor = REPLExecutor(self.index, sub_model=self.sub_model)
        self._system_prompt = build_system_prompt(self.index)

    def save_index(self, path: str) -> None:
        """Save index to JSON file."""
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")
        self.index.save(path)
        print(f"Index saved: {path}")

    def load_index(self, path: str) -> None:
        """Load index from JSON file."""
        self.index = StructuredDocument.load(path)
        self._executor = REPLExecutor(self.index, sub_model=self.sub_model)
        self._system_prompt = build_system_prompt(self.index)
        print(f"Index loaded: {path}")

    @classmethod
    def from_text(
        cls,
        text: str,
        source_name: str = "raw_text",
        root_model: str = ROOT_MODEL,
        sub_model: str = SUB_MODEL,
        generate_summaries: bool = True,
    ) -> "SingleDocRLM":
        """
        Create RLM from raw text string.

        Useful for benchmarks or when document is already in memory.

        Args:
            text: Raw text content
            source_name: Name to identify this document
            root_model: Model for code generation
            sub_model: Model for segmentation/summaries
            generate_summaries: Whether to generate contextual summaries

        Returns:
            SingleDocRLM instance with index already built
        """
        # Create instance with dummy path
        instance = cls(doc_path=source_name, root_model=root_model, sub_model=sub_model)

        # Build index from text directly
        instance.index = build_index_from_text(
            text=text,
            source_name=source_name,
            model=sub_model,
            generate_summaries=generate_summaries,
        )
        instance._executor = REPLExecutor(instance.index, sub_model=sub_model)
        instance._system_prompt = build_system_prompt(instance.index)

        return instance

    def query(
        self,
        question: str,
        max_rounds: int = MAX_REPL_ROUNDS,
        verbose: bool = False,
    ) -> str:
        """
        Answer a question about the document.

        Args:
            question: Natural language question
            max_rounds: Maximum REPL execution rounds
            verbose: Whether to print execution trace

        Returns:
            Answer string
        """
        if self.index is None or self._executor is None:
            raise ValueError("No index built. Call build_index() or load_index() first.")

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": question},
        ]

        # Accumulate outputs from execute_code calls (fallback if final_answer is empty)
        accumulated_outputs = []

        for round_num in range(max_rounds):
            if verbose:
                print(f"\n--- Round {round_num + 1} ---")

            # 1. Call LLM with tools
            response = litellm.completion(
                model=self.root_model,
                messages=messages,
                tools=REPL_TOOLS,
                tool_choice="auto",
                max_tokens=2000,
                temperature=0,
            )

            assistant_message = response.choices[0].message

            # 2. Check if LLM made tool calls
            if assistant_message.tool_calls:
                # Add assistant message to history
                messages.append(assistant_message)

                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name

                    # Parse tool arguments (handle empty/malformed)
                    try:
                        raw_args = tool_call.function.arguments
                        tool_args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
                    except json.JSONDecodeError:
                        tool_args = {}

                    if verbose:
                        print(f"Tool: {tool_name}")

                    if tool_name == "execute_code":
                        code = tool_args.get("code", "")
                        if verbose:
                            print(f"Code:\n{code}")

                        # Execute code
                        output, _ = self._executor.execute_code(code)

                        # Accumulate non-empty outputs
                        if output and output.strip():
                            accumulated_outputs.append(output)

                        if verbose:
                            # Safe print for Windows
                            try:
                                print(f"Output:\n{output}")
                            except UnicodeEncodeError:
                                print(f"Output:\n{output.encode('ascii', 'replace').decode()}")

                        # Add tool result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": output if output else "(no output)"
                        })

                    elif tool_name == "final_answer":
                        answer = tool_args.get("answer", "")
                        if verbose:
                            print(f"Final answer: {answer}")

                        # If final_answer is empty but we have accumulated outputs, use those
                        if not answer or not answer.strip():
                            if accumulated_outputs:
                                if verbose:
                                    print("(Using accumulated outputs as fallback)")
                                return "\n\n".join(accumulated_outputs)

                        return answer

            else:
                # No tool calls - LLM responded with text
                content = assistant_message.content or ""
                if verbose:
                    print(f"Text response: {content[:200]}...")

                # Add to messages and prompt to use tools
                messages.append({"role": "assistant", "content": content})
                messages.append({
                    "role": "user",
                    "content": "Please use the execute_code tool to explore the document, then use final_answer when done."
                })

        return "Max rounds exceeded. Could not complete the query."

    def extract(
        self,
        what: str,
        response_model: Optional[Type[BaseModel]] = None,
        on_error: Literal["raise", "partial"] = "raise",
        verbose: bool = False,
    ) -> list[dict]:
        """
        Extract structured data from document.

        Three-stage pipeline:
        - Stage 0: Generate schema from plain English (if response_model=None)
        - Stage 1: Query document with extraction prompt
        - Stage 2: Convert to JSON via Instructor

        Args:
            what: Description of what to extract (e.g., "papers", "all figures")
            response_model: Pydantic model for output. If None, auto-generates from 'what'.
            on_error: "raise" throws on failure, "partial" returns what was extracted
            verbose: Print execution details

        Returns:
            List of extracted items as dictionaries

        Raises:
            ExtractionError: If extraction fails and on_error="raise"
            SchemaGenerationError: If auto-schema generation fails
            UserCancelledError: If user rejects generated schema

        Example:
            # Plain English (auto-generates schema)
            papers = rlm.extract("all papers with title, authors, year, url")

            # With Pydantic model (skips schema generation)
            papers = rlm.extract("papers", response_model=PaperList)
        """
        if self.index is None or self._executor is None:
            raise ValueError("No index built. Call build_index() or load_index() first.")

        # Stage 0: Schema Generation (if needed)
        if response_model is None:
            if verbose:
                print(f"Stage 0: Generating schema for '{what}'...")

            response_model = generate_schema_from_description(what)

            # Ask user to confirm the schema
            if not confirm_schema_with_user(response_model, what):
                raise UserCancelledError("User rejected the generated schema")

        # Stage 1: Query document
        if verbose:
            print("Stage 1: Querying document...")

        prompt = build_extraction_prompt(what, response_model)
        free_text = self.query(prompt, verbose=verbose)

        if verbose:
            print(f"Stage 1 result: {len(free_text)} chars")

        # Handle empty query result
        if not free_text or not free_text.strip():
            raise ExtractionError(
                f"Query returned empty result for '{what}'. "
                "The LLM may have found the content but failed to include it in the final answer. "
                "Try running the query directly to debug."
            )

        # Stage 2: Convert to JSON
        if verbose:
            print("Stage 2: Converting to JSON...")

        try:
            result = convert_to_json(free_text, response_model)

            # Extract items from the list wrapper
            if hasattr(result, "items"):
                return [item.model_dump() for item in result.items]
            else:
                return [result.model_dump()]

        except Exception as e:
            if on_error == "raise":
                raise ExtractionError(f"Failed to extract {what}: {e}")
            else:
                # Try partial extraction
                if verbose:
                    print(f"Full extraction failed, trying partial: {e}")

                # Get the item model from the list wrapper
                item_model = self._get_item_model(response_model)
                if item_model:
                    partial = extract_partial_from_text(free_text, item_model)
                    if verbose:
                        print(f"Partial extraction got {len(partial)} items")
                    return [item.model_dump() for item in partial]
                return []

    def _get_item_model(self, list_model: Type[BaseModel]) -> Optional[Type[BaseModel]]:
        """Get the item model from a list wrapper model."""
        from typing import get_origin, get_args

        if "items" in list_model.model_fields:
            items_field = list_model.model_fields["items"]
            annotation = items_field.annotation
            if get_origin(annotation) is list:
                args = get_args(annotation)
                if args:
                    return args[0]
        return None

    def summarize(
        self,
        scope: str = "document",
        style: Literal["paragraph", "bullets", "executive", "abstract"] = "paragraph",
        max_length: int = 500,
        structured: bool = False,
        verbose: bool = False,
    ) -> str | dict:
        """
        Summarize document or sections.

        Args:
            scope: What to summarize:
                   - "document": entire document
                   - "section:NAME": single section
                   - "sections:A,B,C": multiple sections
            style: Summary style ("paragraph", "bullets", "executive", "abstract")
            max_length: Target word count
            structured: If True, return dict instead of markdown
            verbose: Print execution details

        Returns:
            Markdown string (structured=False) or dict (structured=True)

        Example:
            # Markdown summary
            summary = rlm.summarize(scope="document", style="executive")

            # Structured summary
            data = rlm.summarize(scope="document", structured=True)
        """
        if self.index is None or self._executor is None:
            raise ValueError("No index built. Call build_index() or load_index() first.")

        # Stage 1: Query for summary
        if verbose:
            print(f"Summarizing {scope} in {style} style...")

        prompt = build_summary_prompt(scope, style, max_length)
        summary_text = self.query(prompt, verbose=verbose)

        if not structured:
            return summary_text

        # Stage 2: Convert to structured format
        if verbose:
            print("Converting to structured format...")

        result = convert_to_json(summary_text, StructuredSummary)
        return result.model_dump()

    def extract_papers(self, verbose: bool = False) -> list[dict]:
        """Convenience method: Extract all referenced papers."""
        return self.extract("referenced papers", response_model=PaperList, verbose=verbose)

    def extract_figures(self, verbose: bool = False) -> list[dict]:
        """Convenience method: Extract all figures with captions."""
        return self.extract("figures with captions", response_model=FigureList, verbose=verbose)

    def interactive_repl(self) -> None:
        """
        Start interactive REPL for document exploration.

        Allows manual code execution for debugging/exploration.
        """
        if self.index is None or self._executor is None:
            raise ValueError("No index built. Call build_index() or load_index() first.")

        print("Interactive REPL for document exploration")
        print(f"Document: {self.index.source_path}")
        print(f"Sections: {len(self.index.sections)}")
        print()
        print("Available functions:")
        print("  Navigation: get_toc(), get_section_names(), get_summary(name), get_all_summaries()")
        print("  Reading: read_section(name), read_section_chunk(name, idx, size), read_range(start, end)")
        print("  Search: grep_section(pattern, name), grep_all(pattern), find_sections_by_keyword(kw)")
        print("  LLM: llm_query(prompt), ask_about_section(question, name)")
        print("  Terminal: FINAL(answer)")
        print()
        print("Type 'exit' or 'quit' to exit. Enter multi-line code ending with a blank line.")
        print()

        while True:
            try:
                # Read multi-line input
                lines = []
                prompt = ">>> "
                while True:
                    line = input(prompt)
                    if line.lower() in ('exit', 'quit'):
                        print("Goodbye!")
                        return
                    if line == "" and lines:
                        break
                    lines.append(line)
                    prompt = "... "

                code = "\n".join(lines)
                if not code.strip():
                    continue

                # Execute
                output, final_answer = self._executor.execute_code(code)

                if output:
                    print(output)

                if final_answer:
                    print(f"[FINAL]: {final_answer}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                return
            except EOFError:
                print("\nGoodbye!")
                return


# =============================================================================
# Convenience Functions
# =============================================================================

def query_document(
    doc_path: str,
    question: str,
    index_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Quick function to query a document.

    Args:
        doc_path: Path to document
        question: Question to answer
        index_path: Path to pre-built index (optional)
        verbose: Print execution trace

    Returns:
        Answer string
    """
    rlm = SingleDocRLM(doc_path)

    if index_path and Path(index_path).exists():
        rlm.load_index(index_path)
    else:
        rlm.build_index()
        if index_path:
            rlm.save_index(index_path)

    return rlm.query(question, verbose=verbose)
