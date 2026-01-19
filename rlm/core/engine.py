"""
RLM Engine - Main extraction engine with REPL-based tool use.

The engine orchestrates:
- Document reading and segmentation
- Root model reasoning loop
- REPL code execution
- Sub-LLM extraction calls
- Result collection and verification
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Type, Optional, Callable

from pydantic import BaseModel

from rlm.config import RLMConfig
from rlm.exceptions import ExtractionError, DocumentError
from rlm.core.types import ExtractionResult, QueryResult, Citation
from rlm.core.repl import REPLEnvironment
from rlm.core.tools import TOOLS
from rlm.core.prompts import build_system_prompt, build_query_prompt, build_user_message
from rlm.document.reader import read_document
from rlm.document.segmenter import split_into_pages, segment_document
from rlm.extraction.structured import (
    llm_extract, llm_extract_parallel, get_section, ask_about_section
)
from rlm.providers import get_provider
from rlm.providers.base import BaseProvider
from rlm.reasoning.tracer import ReasoningTracer
from rlm.reasoning.session import SessionManager


class RLMEngine:
    """
    Recursive Language Model extraction engine.

    Implements the RLM architecture:
    - Root model (e.g., Claude Sonnet) for reasoning and orchestration
    - Sub-model (e.g., GPT-4o-mini) for parallel extraction
    - REPL environment for code execution with persistent state

    Example:
        engine = RLMEngine()
        result = engine.extract("document.pdf", schema=Contact)
        print(result.data)
        print(result.citations)
    """

    def __init__(
        self,
        root_model: str = None,
        sub_model: str = None,
        provider: str = None,
        config: RLMConfig = None
    ):
        """
        Initialize RLM engine.

        Args:
            root_model: Model for root reasoning (default: from config)
            sub_model: Model for sub-extractions (default: from config)
            provider: Provider name (default: auto-detect)
            config: Full configuration object
        """
        self.config = config or RLMConfig.from_env()

        # Override config with explicit args
        self.root_model = root_model or self.config.root_model
        self.sub_model = sub_model or self.config.sub_model

        # Get provider
        self._provider = get_provider(provider or self.config.provider)

        # Output directories
        self.results_dir = Path("results")
        self.sessions_dir = Path("sessions")

    def extract(
        self,
        document: str,
        schema: Type[BaseModel] = None,
        max_iterations: int = 40,
        verbose: bool = False,
        progress_callback: Callable[[str], None] = None
    ) -> ExtractionResult:
        """
        Extract structured data from a document.

        Args:
            document: Path to document file
            schema: Optional Pydantic model for extraction
            max_iterations: Maximum root model iterations
            verbose: Print detailed progress
            progress_callback: Function called with progress messages

        Returns:
            ExtractionResult with extracted data, citations, and metadata

        Example:
            class Contact(BaseModel):
                name: str
                phone: str = None
                email: str = None
                page: int

            result = engine.extract("directory.pdf", schema=Contact)
            for contact in result.data:
                print(contact["name"], contact["phone"])
        """
        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)
            elif verbose:
                print(f"  >> {msg}")

        # Read document
        _progress(f"Reading document: {document}")
        try:
            text = read_document(document)
        except Exception as e:
            raise DocumentError(f"Failed to read document: {e}")

        _progress(f"Document length: {len(text)} chars")

        # Segment document
        _progress("Segmenting document...")
        pages = split_into_pages(text, lines_per_page=50)
        segments = segment_document(text, lines_per_page=50, chunk_size=10, max_workers=3)

        # Build TOC
        toc_text = "\n".join([
            f"[{s['page_range']['start']}-{s['page_range']['end']}] {s['heading']}"
            for s in segments
        ])

        _progress(f"Created {len(pages)} pages, {len(segments)} segments")

        # Setup REPL
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        repl = REPLEnvironment(
            output_dir=self.results_dir,
            progress_callback=_progress
        )

        # Add document data to namespace
        repl.update({
            "pages": pages,
            "segments": segments,
            "total_pages": len(pages),
            # Reasoning state
            "thinking_log": [],
            "citations": [],
            "confidence_history": [],
        })

        # Add extraction functions
        self._setup_extraction_functions(repl, pages, _progress)

        # Add reasoning functions
        self._setup_reasoning_functions(repl, _progress)

        # Build system prompt
        system = build_system_prompt(toc_text, len(pages), schema)
        user_msg = build_user_message("extract")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ]

        # Run extraction loop
        _progress(f"Starting extraction with {self.root_model}...")
        result = self._run_loop(repl, messages, max_iterations, verbose, _progress)

        # Build ExtractionResult
        return ExtractionResult(
            data=result.get("data", []),
            schema_info=result.get("schema", {}),
            verification=result.get("verification", {}),
            citations=[
                Citation(
                    snippet=c.get("snippet", ""),
                    page=c.get("page", 0),
                    note=c.get("note", "")
                )
                for c in result.get("citations", [])
            ],
            thinking_log=result.get("thinking_log", []),
            confidence_history=result.get("confidence_history", []),
            iterations=result.get("iterations", 0),
            document_path=document
        )

    def query(
        self,
        document: str,
        question: str,
        max_iterations: int = 20,
        verbose: bool = False,
        progress_callback: Callable[[str], None] = None
    ) -> QueryResult:
        """
        Answer a question about a document.

        Args:
            document: Path to document file
            question: Question to answer
            max_iterations: Maximum root model iterations
            verbose: Print detailed progress
            progress_callback: Function called with progress messages

        Returns:
            QueryResult with answer, citations, and confidence

        Example:
            result = engine.query("report.pdf", "What was Q3 revenue?")
            print(result.answer)
            print(result.citations)
        """
        def _progress(msg: str):
            if progress_callback:
                progress_callback(msg)
            elif verbose:
                print(f"  >> {msg}")

        # Read document
        _progress(f"Reading document: {document}")
        try:
            text = read_document(document)
        except Exception as e:
            raise DocumentError(f"Failed to read document: {e}")

        # Segment
        pages = split_into_pages(text, lines_per_page=50)
        segments = segment_document(text, lines_per_page=50, chunk_size=10, max_workers=3)

        toc_text = "\n".join([
            f"[{s['page_range']['start']}-{s['page_range']['end']}] {s['heading']}"
            for s in segments
        ])

        # Setup REPL
        repl = REPLEnvironment(progress_callback=_progress)
        repl.update({
            "pages": pages,
            "segments": segments,
            "total_pages": len(pages),
            "thinking_log": [],
            "citations": [],
        })

        self._setup_extraction_functions(repl, pages, _progress)
        self._setup_reasoning_functions(repl, _progress)

        # Build prompt
        system = build_query_prompt(toc_text, len(pages), question)
        user_msg = build_user_message("query")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg}
        ]

        # Run loop
        _progress(f"Answering question with {self.root_model}...")
        result = self._run_loop(repl, messages, max_iterations, verbose, _progress)

        # Extract answer from result
        data = result.get("data", [])
        answer = data[0].get("answer", "") if data else ""
        confidence = result.get("confidence_history", [{}])[-1].get("confidence", 0.0)

        return QueryResult(
            answer=answer,
            citations=[
                Citation(
                    snippet=c.get("snippet", ""),
                    page=c.get("page", 0),
                    note=c.get("note", "")
                )
                for c in result.get("citations", [])
            ],
            confidence=confidence,
            question=question,
            document_path=document
        )

    def _setup_extraction_functions(
        self,
        repl: REPLEnvironment,
        pages: List[str],
        progress_callback: Callable[[str], None]
    ):
        """Add extraction functions to REPL namespace."""
        provider = self._provider
        sub_model = self.sub_model

        def _llm_query(prompt: str, model: str = None) -> str:
            """Raw LLM query."""
            m = model or sub_model
            progress_callback(f"llm_query: {prompt[:50]}...")
            response = provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=m
            )
            return response["choices"][0]["message"]["content"]

        def _get_section(start_page: int, end_page: int, padding: int = 1) -> str:
            """Get content for pages."""
            return get_section(pages, start_page, end_page, padding)

        def _ask_about_section(question: str, start_page: int, end_page: int) -> str:
            """Ask sub-LLM about a section."""
            return ask_about_section(
                provider, question, pages, start_page, end_page,
                sub_model, progress_callback
            )

        def _llm_extract(
            prompt: str,
            response_model: Type[BaseModel],
            start_page: int = None,
            end_page: int = None
        ):
            """Structured extraction."""
            return llm_extract(
                provider, prompt, response_model, sub_model,
                pages=pages, start_page=start_page, end_page=end_page,
                progress_callback=progress_callback
            )

        def _llm_extract_parallel(
            sections: List[tuple],
            prompt_template: str,
            response_model: Type[BaseModel],
            max_workers: int = 5
        ):
            """Parallel extraction."""
            return llm_extract_parallel(
                provider, sections, prompt_template, response_model,
                sub_model, pages, max_workers=max_workers,
                progress_callback=progress_callback
            )

        repl.update({
            "llm_query": _llm_query,
            "get_section": _get_section,
            "ask_about_section": _ask_about_section,
            "llm_extract": _llm_extract,
            "llm_extract_parallel": _llm_extract_parallel,
        })

    def _setup_reasoning_functions(
        self,
        repl: REPLEnvironment,
        progress_callback: Callable[[str], None]
    ) -> ReasoningTracer:
        """
        Add reasoning functions to REPL namespace.

        Uses ReasoningTracer and SessionManager for clean separation.

        Returns:
            ReasoningTracer instance for accessing collected data
        """
        namespace = repl.namespace
        total_pages = namespace.get("total_pages", 1)

        # Create tracer and session manager
        tracer = ReasoningTracer(
            total_pages=total_pages,
            progress_callback=progress_callback
        )
        session_manager = SessionManager(sessions_dir=str(self.sessions_dir))

        # Link tracer state to namespace for compatibility
        namespace["thinking_log"] = tracer.thinking_log
        namespace["citations"] = tracer.citations
        namespace["confidence_history"] = tracer.confidence_history

        # Add tracer functions
        repl.update(tracer.get_functions())

        # Add session functions (linked to namespace)
        repl.update(session_manager.get_functions(namespace))

        return tracer

    def _run_loop(
        self,
        repl: REPLEnvironment,
        messages: List[Dict[str, Any]],
        max_iterations: int,
        verbose: bool,
        progress_callback: Callable[[str], None]
    ) -> Dict[str, Any]:
        """Run the root model reasoning loop."""
        for iteration in range(max_iterations):
            if verbose:
                print(f"\n=== Iteration {iteration + 1} ===")

            # Call root model
            response = self._provider.chat(
                messages=messages,
                model=self.root_model,
                tools=TOOLS
            )

            msg = response["choices"][0]["message"]

            if msg.get("content") and verbose:
                print(f"LLM: {msg['content'][:400]}...")

            tool_calls = msg.get("tool_calls", [])
            if not tool_calls:
                return {
                    "error": "No final_answer called",
                    "last_response": msg.get("content"),
                    "iterations": iteration + 1
                }

            messages.append(msg)

            # Process tool calls
            for tool_call in tool_calls:
                name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"].get("arguments") else {}
                except json.JSONDecodeError:
                    args = {}

                if verbose:
                    print(f"Tool: {name}")

                if name == "execute_code":
                    code = args.get("code", "")

                    if verbose:
                        print("\n" + "-" * 50)
                        print("CODE EXECUTED:")
                        print("-" * 50)
                        for i, line in enumerate(code.split("\n"), 1):
                            clean_line = line.encode('ascii', 'replace').decode('ascii')
                            print(f"  {i:3d} | {clean_line}")
                        print("-" * 50)

                    result = repl.execute(code)

                    if verbose:
                        output = result[:1000] + "\n... (truncated)" if len(result) > 1000 else result
                        clean_output = output.encode('ascii', 'replace').decode('ascii')
                        print(f"OUTPUT:\n{clean_output}")
                        print("-" * 50)

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": result
                    })

                elif name == "final_answer":
                    return {
                        "data": args.get("data", []),
                        "schema": args.get("schema", {}),
                        "verification": args.get("verification", {}),
                        "iterations": iteration + 1,
                        "citations": repl.get("citations", []),
                        "thinking_log": repl.get("thinking_log", []),
                        "confidence_history": repl.get("confidence_history", []),
                    }

                elif name == "final_answer_file":
                    filename = args.get("filename", "")
                    filepath = self.results_dir / filename
                    data = []
                    if filepath.exists():
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                    return {
                        "data": data,
                        "data_file": str(filepath),
                        "schema": args.get("schema", {}),
                        "verification": args.get("verification", {}),
                        "iterations": iteration + 1,
                        "citations": repl.get("citations", []),
                        "thinking_log": repl.get("thinking_log", []),
                        "confidence_history": repl.get("confidence_history", []),
                    }

        return {"error": "Max iterations reached", "iterations": max_iterations}
