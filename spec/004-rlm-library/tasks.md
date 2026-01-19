# Tasks: rlm-library

> Phase 3 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-19
> Prerequisites: [Design Approved]

## Summary

| Phase | Tasks | Estimated | Status |
|-------|-------|-----------|--------|
| Phase 1: Package Structure | 6 tasks | Foundation | Complete |
| Phase 2: Document Handling | 5 tasks | Core | Not Started |
| Phase 3: Provider Abstraction | 5 tasks | Core | Not Started |
| Phase 4: Extraction Engine | 6 tasks | Core | Not Started |
| Phase 5: Reasoning & Session | 4 tasks | Core | Not Started |
| Phase 6: Visualization | 4 tasks | Feature | Not Started |
| Phase 7: CLI | 4 tasks | Feature | Not Started |
| Phase 8: Demo Examples | 5 tasks | Polish | Not Started |
| Phase 9: Testing | 5 tasks | QA | Not Started |
| **Total** | **44 tasks** | | **14% Complete** |

---

## Phase 1: Package Structure

> Create the library skeleton and move existing code

### 1.1 Directory Setup

- [x] **T-001**: Create package directory structure
  ```
  rlm/
    __init__.py
    core/
    document/
    extraction/
    reasoning/
    providers/
    visualization/
    exceptions.py
    config.py
    cli.py
  ```
  - Create all directories with `__init__.py` files
  - Verify `import rlm` works

- [x] **T-002**: Create `rlm/exceptions.py`
  - Define: `RLMError`, `DocumentError`, `ProviderError`, `ExtractionError`, `SchemaError`, `SessionError`
  - Add docstrings for each exception

- [x] **T-003**: Create `rlm/config.py`
  - Implement `RLMConfig` dataclass from design
  - Add `from_env()` class method
  - Support env var overrides: `RLM_ROOT_MODEL`, `RLM_SUB_MODEL`, `RLM_PROVIDER`

### 1.2 Type Definitions

- [x] **T-004**: Create `rlm/core/types.py`
  - Define: `Citation`, `ThinkingEntry`, `ConfidenceEntry`
  - Define: `ExtractionResult`, `QueryResult`
  - Use Pydantic BaseModel with proper generics

- [x] **T-005**: Create `rlm/extraction/schemas.py`
  - Define built-in schemas: `Contact`, `Invoice`, `Entity`, `TableRow`
  - Each with `page: int` field for citation tracking
  - Add docstrings with usage examples

- [x] **T-006**: Create public API stub in `rlm/__init__.py`
  - Export: `extract`, `query`, `RLMEngine`, `visualize`
  - Export: `schemas` module
  - Export: `ExtractionResult`, `QueryResult`, `Citation`
  - Stub implementations that raise `NotImplementedError`

---

## Phase 2: Document Handling

> PDF, Markdown, Text readers and segmentation

### 2.1 Document Readers

- [ ] **T-007**: Create `rlm/document/formats/pdf.py`
  - Move `read_pdf_as_text()` from `evals/generic_extract.py`
  - Move `read_pdf_as_markdown()` from `evals/generic_extract.py`
  - Add `PDFReader` class with `read(path) -> str` method
  - Support both pymupdf and pdfplumber backends

- [ ] **T-008**: Create `rlm/document/formats/markdown.py`
  - Create `MarkdownReader` class
  - Simple file read with encoding handling

- [ ] **T-009**: Create `rlm/document/formats/text.py`
  - Create `TextReader` class
  - Handle various encodings (UTF-8, Latin-1 fallback)

- [ ] **T-010**: Create `rlm/document/reader.py`
  - Create `read_document(path) -> str` factory function
  - Auto-detect format from file extension
  - Raise `DocumentError` for unsupported formats

### 2.2 Segmentation

- [ ] **T-011**: Create `rlm/document/segmenter.py`
  - Move `split_into_pages()` from `segmenter.py`
  - Move `segment_document()` from `segmenter.py`
  - Move `process_chunk()` from `segmenter.py`
  - Update to use provider abstraction (not hardcoded OpenRouter)
  - Add `Segment` Pydantic model

---

## Phase 3: Provider Abstraction

> Abstract LLM calls across OpenRouter, OpenAI, Anthropic, Ollama

### 3.1 Base Provider

- [ ] **T-012**: Create `rlm/providers/base.py`
  - Define `BaseProvider` ABC with:
    - `chat(messages, model, tools) -> dict`
    - `extract(prompt, response_model, model) -> BaseModel`
  - Add retry decorator with exponential backoff
  - Handle rate limits and timeouts

### 3.2 Provider Implementations

- [ ] **T-013**: Create `rlm/providers/openrouter.py`
  - Implement `OpenRouterProvider(BaseProvider)`
  - Use existing OpenRouter code from `evals/generic_extract.py`
  - Support both chat and instructor-based extraction

- [ ] **T-014**: Create `rlm/providers/openai.py`
  - Implement `OpenAIProvider(BaseProvider)`
  - Direct OpenAI API calls
  - Instructor integration for structured output

- [ ] **T-015**: Create `rlm/providers/anthropic.py`
  - Implement `AnthropicProvider(BaseProvider)`
  - Use anthropic SDK
  - Handle tool use format differences

- [ ] **T-016**: Create `rlm/providers/factory.py`
  - Create `get_provider(name: str) -> BaseProvider` factory
  - Auto-detect from environment variables if name not specified
  - Raise `ProviderError` for unknown providers

---

## Phase 4: Extraction Engine

> Core RLM engine with REPL and tools

### 4.1 REPL Environment

- [ ] **T-017**: Create `rlm/core/repl.py`
  - Create `REPLEnvironment` class
  - Manage namespace dictionary
  - Provide `execute(code) -> str` method with stdout capture
  - Handle exceptions with line number context

- [ ] **T-018**: Create `rlm/core/tools.py`
  - Define TOOLS list (execute_code, final_answer, final_answer_file)
  - Move tool definitions from `evals/generic_extract.py`
  - Create helper functions: `get_section()`, `ask_about_section()`

### 4.2 Extraction Functions

- [ ] **T-019**: Create `rlm/extraction/structured.py`
  - Move `llm_extract()` from `evals/generic_extract.py`
  - Move `llm_extract_parallel()` from `evals/generic_extract.py`
  - Update to use provider abstraction
  - Add progress callback support

### 4.3 Engine

- [ ] **T-020**: Create `rlm/core/prompts.py`
  - Move `build_system_prompt()` from `evals/generic_extract.py`
  - Templatize with configurable sections
  - Support custom tool descriptions

- [ ] **T-021**: Create `rlm/core/engine.py`
  - Create `RLMEngine` class
  - Refactor `run_generic_extraction()` into class methods
  - Implement `extract(document, schema)` method
  - Implement `query(document, question)` method
  - Wire together: reader -> segmenter -> REPL -> provider

- [ ] **T-022**: Implement public API in `rlm/__init__.py`
  - Implement `extract()` function using RLMEngine
  - Implement `query()` function using RLMEngine
  - Use default config from environment

---

## Phase 5: Reasoning & Session

> Thinking, citations, progress evaluation, session persistence

### 5.1 Reasoning Tracer

- [ ] **T-023**: Create `rlm/reasoning/tracer.py`
  - Create `ReasoningTracer` class
  - Move `think()` from `evals/generic_extract.py`
  - Move `cite()` from `evals/generic_extract.py`
  - Move `evaluate_progress()` from `evals/generic_extract.py`
  - Store state in instance variables

### 5.2 Session Management

- [ ] **T-024**: Create `rlm/reasoning/session.py`
  - Create `SessionManager` class
  - Move `save_session()` from `evals/generic_extract.py`
  - Move `load_session()` from `evals/generic_extract.py`
  - Add session metadata (timestamp, document path, schema)

- [ ] **T-025**: Integrate reasoning into RLMEngine
  - Inject `ReasoningTracer` into REPL namespace
  - Collect reasoning data into `ExtractionResult`
  - Add `save_session()` and `load_session()` to engine

- [ ] **T-026**: Add session CLI support
  - `rlm sessions list` - list saved sessions
  - `rlm sessions resume <name>` - resume session

---

## Phase 6: Visualization

> HTML report generation with citation highlighting

### 6.1 HTML Generation

- [ ] **T-027**: Create `rlm/visualization/html.py`
  - Create `generate_html_report(result, document_text) -> str`
  - Template with:
    - Extracted data table
    - Source document with citation highlights
    - Collapsible reasoning trace
    - Confidence progress bar

- [ ] **T-028**: Create `rlm/visualization/citations.py`
  - Create `highlight_citations(text, citations) -> str`
  - Wrap citation snippets in `<mark>` tags
  - Handle overlapping citations
  - Add citation IDs for linking

### 6.2 Integration

- [ ] **T-029**: Add `visualize()` to public API
  - `rlm.visualize(result, output="report.html")`
  - Auto-open in browser option
  - Return HTML string if no output path

- [ ] **T-030**: Create standalone HTML template
  - No external dependencies (inline CSS/JS)
  - Responsive design
  - Print-friendly styles

---

## Phase 7: CLI

> Command-line interface using Click

### 7.1 CLI Commands

- [ ] **T-031**: Create `rlm/cli.py` with Click
  - Main group: `rlm`
  - Add `--version` flag
  - Add `--verbose` flag

- [ ] **T-032**: Implement `rlm extract` command
  ```
  rlm extract <document> --schema <name> --output <path>
  ```
  - Support built-in schema names: contact, invoice, entity, table
  - Support custom schema from Python file: `--schema path/to/schema.py:ClassName`
  - Output formats: json, csv

- [ ] **T-033**: Implement `rlm query` command
  ```
  rlm query <document> "<question>"
  ```
  - Print answer with citation
  - Support `--json` for structured output

- [ ] **T-034**: Implement `rlm visualize` command
  ```
  rlm visualize <result.json> --output <report.html>
  ```
  - Read extraction result JSON
  - Generate HTML report

---

## Phase 8: Demo Examples

> Compelling demos like LangExtract

### 8.1 Invoice Demo

- [ ] **T-035**: Create `examples/01_invoice_extraction/`
  - Source sample invoice PDF
  - Create `extract_invoice.py` (< 10 lines)
  - Generate `output.html` visualization
  - Write `README.md` with screenshot

### 8.2 Contacts Demo

- [ ] **T-036**: Create `examples/02_contacts_directory/`
  - Use existing `40255083-Agribusiness-Companies.pdf`
  - Create `extract_contacts.py`
  - Generate `output.html`
  - Write `README.md`

### 8.3 Literature Demo

- [ ] **T-037**: Create `examples/03_classic_literature/`
  - Create `fetch_gutenberg.py` to download Romeo & Juliet
  - Create `extract_characters.py` - extract all characters with descriptions
  - Generate `output.html`
  - Write `README.md`

### 8.4 Benchmark Demo

- [ ] **T-038**: Create `examples/benchmark/`
  - Create `run_benchmark.py` using QASPER dataset
  - Create `compare_metrics.py` for analysis
  - Generate comparison report HTML

- [ ] **T-039**: Update main `README.md`
  - Add installation instructions
  - Add quick start with code examples
  - Link to demo examples
  - Add benchmark results table

---

## Phase 9: Testing

> Unit and integration tests

### 9.1 Unit Tests

- [ ] **T-040**: Create `tests/test_document.py`
  - Test PDF reading (text + markdown modes)
  - Test Markdown reading
  - Test Text reading
  - Test format auto-detection
  - Test segmentation

- [ ] **T-041**: Create `tests/test_providers.py`
  - Test provider factory
  - Test retry logic (mock API errors)
  - Test each provider's chat/extract methods (mocked)

- [ ] **T-042**: Create `tests/test_extraction.py`
  - Test `llm_extract()` with mock provider
  - Test `llm_extract_parallel()` with mock provider
  - Test schema validation

- [ ] **T-043**: Create `tests/test_reasoning.py`
  - Test `ReasoningTracer` state management
  - Test `SessionManager` save/load round-trip
  - Test confidence calculation

### 9.2 Integration Tests

- [ ] **T-044**: Create `tests/test_integration.py`
  - End-to-end extraction test with sample PDF
  - Test CLI commands
  - Test visualization output
  - Test session resume workflow

---

## Task Dependencies

```
Phase 1 (Foundation)
T-001 -> T-002 -> T-003 -> T-004 -> T-005 -> T-006

Phase 2 (Documents) - can start after T-001
T-001 -> T-007, T-008, T-009 -> T-010 -> T-011

Phase 3 (Providers) - can start after T-003
T-003 -> T-012 -> T-013, T-014, T-015 -> T-016

Phase 4 (Engine) - needs Phase 2 + 3
T-010, T-016 -> T-017 -> T-018 -> T-019 -> T-020 -> T-021 -> T-022

Phase 5 (Reasoning) - needs T-017
T-017 -> T-023 -> T-024 -> T-025 -> T-026

Phase 6 (Visualization) - needs T-022
T-022 -> T-027 -> T-028 -> T-029 -> T-030

Phase 7 (CLI) - needs T-022
T-022 -> T-031 -> T-032, T-033, T-034

Phase 8 (Demos) - needs Phase 6 + 7
T-029, T-032 -> T-035, T-036, T-037, T-038, T-039

Phase 9 (Testing) - can run in parallel with Phase 8
T-022 -> T-040, T-041, T-042, T-043 -> T-044
```

## Parallel Work Opportunities

These task groups can be worked on simultaneously:

1. **T-007, T-008, T-009** - Document format readers (independent)
2. **T-013, T-014, T-015** - Provider implementations (after T-012)
3. **T-035, T-036, T-037** - Demo examples (after T-029)
4. **T-040, T-041, T-042, T-043** - Unit test files (after T-022)

---

## Implementation Order (Recommended)

For fastest path to working demo:

1. **MVP Path** (T-001 -> T-006 -> T-007 -> T-010 -> T-012 -> T-013 -> T-017 -> T-021 -> T-022)
   - Get `rlm.extract()` working with OpenRouter

2. **Add Visualization** (T-027 -> T-029)
   - Make results compelling

3. **Add CLI** (T-031 -> T-032)
   - Enable command-line usage

4. **Add Demos** (T-035 -> T-036)
   - Showcase capabilities

5. **Polish** (remaining tasks)
   - Additional providers, tests, docs

---

## Notes

- Each task should be completable in 1-4 hours
- Mark tasks complete with [x] as you finish them
- Test each component before moving to dependent tasks
- Existing code in `evals/generic_extract.py` provides reference implementation
- Keep Windows compatibility (no emojis, handle paths correctly)

---

> Next: Run `/spec:approve tasks` when this breakdown is complete and reviewed.
