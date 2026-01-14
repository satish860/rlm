# Requirements: Single-doc-flow

> Phase 1 of Spec-Driven Development
> Status: APPROVED
> Approved: 2026-01-14

## 1. Overview

### 1.1 Purpose
Enable querying and understanding of single long documents (1M - 100M tokens) that exceed LLM context windows (~200K tokens) by building a structured index with TOC, section boundaries, and summaries, then exposing navigation/search functions through a REPL environment.

### 1.2 Scope

**Included:**
- Document loading and pre-processing
- TOC generation (automatic structure detection)
- Section boundary mapping (character offsets)
- Section summary generation via sub-LLM
- REPL environment with navigation, reading, search, and LLM functions
- Query execution flow (root LLM writes code, executes in REPL)
- Index persistence (save/load)

**Excluded:**
- Multi-document corpus handling (separate spec)
- PDF parsing/extraction (use extracted text)
- Image/figure extraction and analysis
- Table extraction and querying
- Cross-reference resolution
- Real-time document updates

### 1.3 Success Criteria
- Can process documents up to 100M characters
- TOC generation works for: research papers, legal docs, technical manuals, books, markdown
- Query latency < 30 seconds for typical questions
- Accuracy comparable to or better than baseline RLM (flat string approach)
- Index build time < 5 minutes for 1M character document

## 2. User Stories

### US-001: Index a Long Document
**As a** developer
**I want** to build an index for a long document
**So that** I can query it efficiently without loading everything into context

**Acceptance Criteria:**
- [ ] Can specify document path (text file)
- [ ] Automatically detects document type (research_paper, legal, technical_manual, book, markdown, other)
- [ ] Generates hierarchical TOC with section titles and levels
- [ ] Maps section names to character offsets (start, end)
- [ ] Generates 2-3 sentence summary for each section
- [ ] Extracts keywords for each section
- [ ] Completes within reasonable time (< 5 min for 1M chars)

### US-002: Save and Load Index
**As a** developer
**I want** to persist the document index to disk
**So that** I don't rebuild it every time I query the document

**Acceptance Criteria:**
- [ ] Can save index to JSON file
- [ ] Can load index from JSON file
- [ ] Loaded index is functionally identical to freshly built one
- [ ] Index file is human-readable for debugging

### US-003: Query a Document
**As a** user
**I want** to ask natural language questions about a long document
**So that** I can find information without reading the entire document

**Acceptance Criteria:**
- [ ] Can ask questions like "What datasets were used?"
- [ ] System uses TOC/summaries to identify relevant sections
- [ ] System reads only necessary sections (not entire document)
- [ ] Returns accurate, well-formatted answers
- [ ] Shows which sections were consulted (transparency)

### US-004: Explore Document Interactively
**As a** developer
**I want** an interactive REPL to explore a document
**So that** I can understand its structure and debug queries

**Acceptance Criteria:**
- [ ] Can call navigation functions (get_toc, get_section_names, get_summary)
- [ ] Can call reading functions (read_section, read_section_chunk, read_range)
- [ ] Can call search functions (grep_section, grep_all, find_sections_by_keyword)
- [ ] Can call LLM functions (llm_query, ask_about_section)
- [ ] REPL shows function outputs clearly

### US-005: Use Different LLM Providers
**As a** developer
**I want** to use either OpenAI or Anthropic models
**So that** I can choose based on cost, performance, or preference

**Acceptance Criteria:**
- [ ] Can configure root LLM (e.g., gpt-4, claude-opus)
- [ ] Can configure sub-LLM (e.g., gpt-4-mini, claude-haiku)
- [ ] Both providers work with same interface
- [ ] API keys read from environment variables

## 3. Functional Requirements

### FR-001: Document Loading
**Description:** Load a text document from filesystem and determine its basic properties (total characters, encoding).
**Priority:** Must Have
**Dependencies:** None

### FR-002: Document Type Detection
**Description:** Automatically detect document type (research_paper, legal, technical_manual, book, markdown, report, other) by analyzing document preview with LLM.
**Priority:** Must Have
**Dependencies:** FR-001, LLM interface

### FR-003: TOC Generation
**Description:** Generate hierarchical table of contents based on document type:
- Markdown: Parse `#`, `##`, `###` headings
- Research paper: Parse numbered sections, LaTeX `\section{}`
- Legal: Parse numbered clauses (1.1, Article I)
- Book: Parse chapters, parts
- Other: Use LLM to identify structure
**Priority:** Must Have
**Dependencies:** FR-002

### FR-004: Section Boundary Mapping
**Description:** Map each section name to (start_char, end_char) positions. End of one section = start of next. Last section ends at document end.
**Priority:** Must Have
**Dependencies:** FR-003

### FR-005: Section Summary Generation
**Description:** Generate 2-3 sentence summary for each section using sub-LLM. Focus on: main topic, key points, important terms.
**Priority:** Must Have
**Dependencies:** FR-004, LLM interface

### FR-006: Keyword Extraction
**Description:** Extract keywords/key terms for each section to enable keyword-based section lookup.
**Priority:** Should Have
**Dependencies:** FR-004, LLM interface

### FR-007: Index Data Structure
**Description:** Combine all indexing outputs into SingleDocIndex dataclass:
- path: str
- doc_type: str
- total_chars: int
- toc: dict (hierarchical)
- section_map: dict (section_name -> (start, end))
- summaries: dict (section_name -> summary)
- keywords: dict (section_name -> [keywords])
**Priority:** Must Have
**Dependencies:** FR-003, FR-004, FR-005, FR-006

### FR-008: Index Persistence
**Description:** Save index to JSON file and load from JSON file.
**Priority:** Must Have
**Dependencies:** FR-007

### FR-009: REPL Navigation Functions
**Description:** Implement functions: get_toc(), get_section_names(), get_summary(section_name), get_all_summaries()
**Priority:** Must Have
**Dependencies:** FR-007

### FR-010: REPL Reading Functions
**Description:** Implement functions: read_section(section_name), read_section_chunk(section_name, chunk_idx, chunk_size), read_range(start, end)
**Priority:** Must Have
**Dependencies:** FR-007

### FR-011: REPL Search Functions
**Description:** Implement functions: grep_section(pattern, section_name), grep_all(pattern), find_sections_by_keyword(keyword)
**Priority:** Must Have
**Dependencies:** FR-007

### FR-012: REPL LLM Functions
**Description:** Implement functions: llm_query(prompt), ask_about_section(question, section_name)
**Priority:** Must Have
**Dependencies:** LLM interface

### FR-013: REPL Environment Setup
**Description:** Create REPL environment dict exposing all functions and metadata for code execution.
**Priority:** Must Have
**Dependencies:** FR-009, FR-010, FR-011, FR-012

### FR-014: Root LLM System Prompt
**Description:** Generate system prompt with document info, TOC, summaries, available functions, and strategy guidance.
**Priority:** Must Have
**Dependencies:** FR-007, FR-013

### FR-015: Query Execution Loop
**Description:** Execute root LLM code in REPL environment:
1. Root LLM generates Python code
2. Execute code in REPL environment
3. Capture output (stdout, return values)
4. Feed output back to root LLM
5. Repeat until FINAL(answer) called
6. Return final answer
**Priority:** Must Have
**Dependencies:** FR-013, FR-014

### FR-016: SingleDocRLM Main Class
**Description:** Main API class with methods: build_index(), save_index(path), load_index(path), query(question), repl()
**Priority:** Must Have
**Dependencies:** All above

## 4. Non-Functional Requirements

### NFR-001: Performance
- Index build: < 5 minutes for 1M character document
- Query latency: < 30 seconds for typical questions (excluding LLM API latency)
- Section read: < 100ms for any section
- Memory: Index should fit in RAM for documents up to 100M chars

### NFR-002: Scalability
- Support documents up to 100M characters (approx 25M tokens)
- Support up to 1000 sections per document
- Support section sizes from 100 chars to 10M chars

### NFR-003: Reliability
- Graceful degradation if TOC detection fails (fall back to flat document)
- Handle encoding errors in document files
- Retry LLM API calls on transient failures (with backoff)
- Validate index integrity on load

### NFR-004: Maintainability
- Type hints on all public functions
- Docstrings for all public functions
- Unit tests for each component
- Clear separation between indexer, REPL, and LLM components

### NFR-005: Observability
- Track total tokens used (root LLM + sub-LLM)
- Log each REPL function call
- Log each LLM API call with latency
- Report sections accessed during query

## 5. Technical Constraints

- Python 3.10+
- Must use existing LLM interface from `src/llm/`
- Must support both OpenAI and Anthropic providers
- No external dependencies beyond: openai, anthropic, pydantic (standard libs OK)
- Must run on Windows (no Unix-specific code)
- No emojis in code or output

## 6. Assumptions

- Input documents are plain text files (UTF-8 encoded)
- Document structure is detectable (has headings, sections, or chapters)
- Root LLM can generate valid Python code
- Sub-LLM can produce useful summaries and keyword extractions
- LLM API is available and responsive

## 7. Out of Scope

- PDF parsing (user must extract text first)
- Image/figure analysis
- Table extraction and structured querying
- Cross-reference resolution ("see Section 3.2")
- Real-time document editing/updates
- Multi-document corpus handling
- Streaming responses
- Web UI or GUI
- Authentication/authorization

## 8. Open Questions

- [ ] What should happen if document has no detectable structure? (Fallback to fixed-size chunks?)
- [ ] Should we support incremental index updates or always full rebuild?
- [ ] What's the maximum number of REPL execution rounds before giving up?
- [ ] Should sub-LLM calls be batched (multiple sections per call) for efficiency?
- [ ] How should we handle very large sections (>context window) during summarization?

---

> Next: Run `/spec:approve requirements` when this document is complete and reviewed.
