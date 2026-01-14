"""Single Document RLM - Main API.

Provides the main interface for querying long documents using
the Recursive Language Model approach.
"""

import json
from pathlib import Path
from typing import Optional

import litellm

from .indexer import SingleDocIndex, build_index
from .repl import REPLExecutor, REPL_TOOLS, build_system_prompt
from ..config import ROOT_MODEL, SUB_MODEL, TOC_MODEL, MAX_REPL_ROUNDS


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
        toc_model: str = TOC_MODEL,
    ):
        """
        Initialize RLM for a document.

        Args:
            doc_path: Path to document (PDF/DOCX/HTML/TXT/MD)
            root_model: Model for code generation (e.g., gpt-4o)
            sub_model: Model for summaries/semantic tasks
            toc_model: Lightweight model for TOC fallback
        """
        self.doc_path = doc_path
        self.root_model = root_model
        self.sub_model = sub_model
        self.toc_model = toc_model

        self.index: Optional[SingleDocIndex] = None
        self._executor: Optional[REPLExecutor] = None
        self._system_prompt: Optional[str] = None

    def build_index(
        self,
        output_dir: Optional[str] = None,
        generate_summaries: bool = True,
    ) -> None:
        """
        Build document index (TOC, sections, summaries).

        Args:
            output_dir: Directory for output files (default: same as source)
            generate_summaries: Whether to generate contextual summaries
        """
        self.index = build_index(
            source_path=self.doc_path,
            output_dir=output_dir,
            generate_summaries=generate_summaries,
            use_llm_toc_fallback=True,
            toc_model=self.toc_model,
            summary_model=self.sub_model,
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
        self.index = SingleDocIndex.load(path)
        self._executor = REPLExecutor(self.index, sub_model=self.sub_model)
        self._system_prompt = build_system_prompt(self.index)
        print(f"Index loaded: {path}")

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
                    tool_args = json.loads(tool_call.function.arguments)

                    if verbose:
                        print(f"Tool: {tool_name}")

                    if tool_name == "execute_code":
                        code = tool_args.get("code", "")
                        if verbose:
                            print(f"Code:\n{code}")

                        # Execute code
                        output, _ = self._executor.execute_code(code)

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
