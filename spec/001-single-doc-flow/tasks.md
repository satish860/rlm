# Tasks: Single-doc-flow

> Phase 3 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-14
> Prerequisites: [Design Approved]

## Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Foundation | 5 tasks | Not Started |
| Phase 2: Indexing | 6 tasks | Not Started |
| Phase 3: REPL | 5 tasks | Not Started |
| Phase 4: Integration | 4 tasks | Not Started |
| Phase 5: Testing & Polish | 5 tasks | Not Started |
| **Total** | **25 tasks** | **0% Complete** |

---

## Phase 1: Foundation

> Project setup and basic infrastructure

### 1.1 Project Structure

- [ ] **T-001**: Create directory structure
  - Create `src/` directory
  - Create `src/single_doc/` with `__init__.py`
  - Create `src/utils/` with `__init__.py`
  - Create `tests/` directory
  - Files: `src/__init__.py`, `src/single_doc/__init__.py`, `src/utils/__init__.py`

- [ ] **T-002**: Setup dependencies
  - Create `requirements.txt` with: `litellm`, `markitdown`
  - Verify installation works: `pip install -r requirements.txt`
  - Test imports work
  - File: `requirements.txt`

### 1.2 Document Conversion

- [ ] **T-003**: Implement document converter
  - Create markitdown wrapper
  - Support: PDF, DOCX, HTML, TXT, MD input
  - Output: markdown string + save to file
  - Handle encoding errors gracefully
  - File: `src/single_doc/converter.py`
  - Test: Convert sample PDF and DOCX

- [ ] **T-004**: Implement token counter utility
  - Wrap litellm token counting
  - Track input/output tokens per call
  - Track total cost
  - File: `src/utils/token_counter.py`

- [ ] **T-005**: Verify litellm works with target models
  - Test call to `gemini/gemini-2.0-flash` (cheap TOC model)
  - Test call to `gpt-4o-mini` (sub-LLM)
  - Test call to `gpt-4o` (root LLM)
  - Verify API keys from environment
  - File: `tests/test_llm_connection.py`

---

## Phase 2: Indexing

> TOC parsing, section mapping, and contextual summaries

### 2.1 TOC Parsing

- [ ] **T-006**: Implement regex TOC parser
  - Parse markdown `#` headings (levels 1-6)
  - Extract: title, level, start_char position
  - Calculate end_char (start of next section or EOF)
  - Determine parent for each section (hierarchy)
  - File: `src/single_doc/indexer.py` - `parse_toc_regex()`
  - Test with sample markdown files

- [ ] **T-007**: Implement LLM TOC fallback
  - When regex finds no headings, call lightweight LLM
  - Prompt: "Identify major sections in this document"
  - Parse LLM response into section list
  - Convert line numbers to char offsets
  - File: `src/single_doc/indexer.py` - `parse_toc_llm()`
  - Dependencies: T-005

- [ ] **T-008**: Implement tiered TOC parsing
  - Tier 1: Try regex parser
  - Tier 2: If no sections, try LLM fallback
  - Tier 3: If both fail, single section (entire doc)
  - File: `src/single_doc/indexer.py` - `parse_toc()`
  - Dependencies: T-006, T-007

### 2.2 Contextual Summaries

- [ ] **T-009**: Implement contextual summary generation
  - Use Anthropic's Contextual Retrieval approach
  - Pass whole document + section to sub-LLM
  - Generate 50-100 token context per section
  - Prompt: "<document>...</document><chunk>...</chunk> Please give short context..."
  - File: `src/single_doc/indexer.py` - `generate_section_context()`
  - Dependencies: T-005

- [ ] **T-010**: Implement full indexer pipeline
  - Load document, convert with markitdown
  - Parse TOC (tiered approach)
  - Generate contextual summary for each section
  - Build Section dataclass with parent info
  - Build SingleDocIndex dataclass
  - File: `src/single_doc/indexer.py` - `build_index()`
  - Dependencies: T-003, T-008, T-009

- [ ] **T-011**: Implement index persistence
  - `SingleDocIndex.to_json()` - serialize to JSON
  - `SingleDocIndex.from_json()` - deserialize from JSON
  - Save/load markdown file path
  - Validate index integrity on load
  - File: `src/single_doc/indexer.py`
  - Dependencies: T-010

---

## Phase 3: REPL

> REPL functions and code execution

### 3.1 REPL Functions

- [ ] **T-012**: Implement navigation functions (FREE)
  - `get_toc()` - return TOC list
  - `get_section_names()` - return section name list
  - `get_summary(name)` - return summary for section
  - `get_all_summaries()` - return all summaries dict
  - File: `src/single_doc/repl.py`
  - Dependencies: T-010

- [ ] **T-013**: Implement reading functions (FREE)
  - `read_section(name)` - read full section from markdown file
  - `read_section_chunk(name, idx, size)` - read chunk of section
  - `read_range(start, end)` - read raw char range
  - Handle section not found error
  - File: `src/single_doc/repl.py`
  - Dependencies: T-010

- [ ] **T-014**: Implement search functions (FREE)
  - `grep_section(pattern, name)` - regex search in one section
  - `grep_all(pattern)` - regex search all sections, return dict
  - `find_sections_by_keyword(kw)` - search keywords in index
  - Handle invalid regex gracefully
  - File: `src/single_doc/repl.py`
  - Dependencies: T-010

- [ ] **T-015**: Implement LLM functions (COSTS MONEY)
  - `llm_query(prompt)` - call sub-LLM with prompt
  - `ask_about_section(question, name)` - call sub-LLM with section content
  - Track token usage
  - File: `src/single_doc/repl.py`
  - Dependencies: T-004, T-005

### 3.2 Code Execution

- [ ] **T-016**: Implement sandboxed code executor
  - `REPLExecutor` class
  - Build restricted globals (SAFE_BUILTINS + REPL functions)
  - Execute code with `exec()`
  - Capture stdout with `io.StringIO`
  - Capture exceptions, feed back as output
  - Implement `FINAL(answer)` to signal completion
  - Add 30 second timeout per execution
  - File: `src/single_doc/repl.py` - `REPLExecutor`
  - Dependencies: T-012, T-013, T-014, T-015

---

## Phase 4: Integration

> Main class and query execution

### 4.1 System Prompt

- [ ] **T-017**: Implement system prompt generator
  - Template with: document info, TOC, summaries
  - List all REPL functions with FREE/COSTS labels
  - Include STRATEGY and ANTI-HALLUCINATION rules
  - Include example code
  - File: `src/single_doc/rlm.py` - `build_system_prompt()`
  - Dependencies: T-010

### 4.2 Query Loop

- [ ] **T-018**: Implement query execution loop
  - Build messages: system prompt + user question
  - Loop: call root LLM -> extract code -> execute -> feed output back
  - Check for `FINAL()` call to exit loop
  - Enforce max_rounds limit (default 10)
  - Return final answer or "max rounds exceeded"
  - File: `src/single_doc/rlm.py` - `execute_query()`
  - Dependencies: T-016, T-017

- [ ] **T-019**: Implement code extraction from LLM response
  - Extract code blocks from markdown (```python ... ```)
  - Handle multiple code blocks
  - Handle responses with no code block
  - File: `src/single_doc/rlm.py` - `extract_code_block()`

### 4.3 Main Class

- [ ] **T-020**: Implement SingleDocRLM main class
  - `__init__(doc_path, root_model, sub_model, toc_model)`
  - `build_index()` - calls indexer
  - `save_index(path)` - save to JSON
  - `load_index(path)` - load from JSON
  - `query(question, max_rounds)` - run query loop
  - `repl()` - interactive mode (Phase 5)
  - File: `src/single_doc/rlm.py`
  - Dependencies: T-010, T-011, T-018

---

## Phase 5: Testing & Polish

> Quality assurance and refinements

### 5.1 Testing

- [ ] **T-021**: Write unit tests for indexer
  - Test regex TOC parsing with various markdown formats
  - Test section boundary calculation
  - Test index serialization roundtrip
  - File: `tests/test_indexer.py`
  - Dependencies: T-010, T-011

- [ ] **T-022**: Write unit tests for REPL functions
  - Test each navigation/read/search function
  - Test with mock index data
  - Test error cases (section not found, invalid regex)
  - File: `tests/test_repl.py`
  - Dependencies: T-016

- [ ] **T-023**: Write integration test with real document
  - Use sample document (e.g., RLM paper PDF)
  - Build index, run sample queries
  - Verify answers are grounded in document
  - File: `tests/test_integration.py`
  - Dependencies: T-020

### 5.2 Polish

- [ ] **T-024**: Add error handling and retries
  - Retry LLM calls with exponential backoff (3 attempts)
  - Better error messages for common failures
  - Handle encoding errors in document conversion
  - File: Multiple files
  - Dependencies: T-020

- [ ] **T-025**: Implement interactive REPL mode
  - `SingleDocRLM.repl()` method
  - Python REPL with REPL functions pre-loaded
  - Print welcome message with available functions
  - File: `src/single_doc/rlm.py`
  - Dependencies: T-020

---

## Task Dependencies

```
T-001 --> T-002 --> T-003, T-004, T-005
                         |
                         v
                    T-006 --> T-007 --> T-008
                                          |
                                          v
                    T-009 -------------> T-010 --> T-011
                                          |
                         +----------------+----------------+
                         |                |                |
                         v                v                v
                       T-012           T-013           T-014
                         |                |                |
                         +----------------+----------------+
                                          |
                                          v
                       T-015 --------> T-016
                                          |
                                          v
                       T-017 --------> T-018 <-- T-019
                                          |
                                          v
                                       T-020
                                          |
                         +----------------+----------------+
                         |                |                |
                         v                v                v
                       T-021           T-022           T-023
                                          |
                                          v
                                   T-024, T-025
```

## Parallel Work Opportunities

These tasks can be done in parallel:
- T-003, T-004, T-005 (after T-002)
- T-006, T-009 (independent indexing components)
- T-012, T-013, T-014 (REPL function groups)
- T-021, T-022 (unit test groups)
- T-024, T-025 (polish tasks)

## Notes

- Each task is designed to be completable in 1-3 hours
- Test each component immediately after building (per user preference)
- Use OOLONG benchmark docs for testing when available
- No emojis in code (Windows environment)

---

> Next: Run `/spec:approve tasks` when this breakdown is complete and reviewed.
