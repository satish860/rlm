# Requirements: rlm-library

> Phase 1 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-19

## 1. Overview

### 1.1 Purpose

Package RLM (Recursive Language Model) as a production-ready Python library that enables intelligent document extraction using the "root model + sub-LLM" architecture. The library should provide a simple, schema-driven API that outperforms alternatives like langextract by leveraging explicit reasoning, verbatim citations, and parallel extraction.

**Key Problems Solved:**
- Fuzzy alignment hangs in langextract (Issue #277) - RLM uses explicit page numbers + verbatim citations
- JSON parsing fragility (Issues #287, #283, #258) - RLM uses Pydantic models via Instructor
- No error recovery (Issue #240) - RLM provides REPL with try/except + session resume
- No reasoning trace - RLM provides think(), cite(), evaluate_progress()
- Single-pass stochastic - RLM root model iterates intelligently

### 1.2 Scope

**In Scope:**
- Simple extraction API: `rlm.extract(doc, schema)`
- Q&A API: `rlm.query(doc, question)`
- Provider abstraction (OpenRouter, OpenAI, Anthropic, Ollama)
- Document reading (PDF, Markdown, Text)
- Parallel extraction with progress tracking
- Reasoning tools (think, cite, evaluate_progress)
- Session persistence (save/load)
- HTML visualization of results
- CLI interface

**Out of Scope (v1):**
- Web UI
- Database storage
- Multi-document queries
- RAG / vector retrieval (pure RLM approach)
- Fine-tuning support

### 1.3 Success Criteria

1. **Simple API**: `rlm.extract("doc.pdf", schema=MyModel)` works out of the box
2. **Accuracy**: Match or beat langextract F1 scores on extraction benchmarks
3. **Reliability**: Zero JSON parsing failures (Pydantic handles validation)
4. **Transparency**: Full reasoning trace available via `result.thinking_log`
5. **Performance**: 5x+ speedup via parallel extraction (existing code shows this works)
6. **Production-ready**: Retry logic, caching, logging, error handling

---

## 2. User Stories

### US-001: Simple Document Extraction
**As a** developer
**I want** to extract structured data from a document with one function call
**So that** I can quickly integrate document processing into my application

**Acceptance Criteria:**
- [ ] `rlm.extract("invoice.pdf", schema=Invoice)` returns extracted data
- [ ] Schema is defined as a standard Pydantic model
- [ ] Result includes page numbers for each extracted record
- [ ] Works with PDF, Markdown, and plain text files

### US-002: Custom Schema Definition
**As a** developer
**I want** to define my own extraction schema using Pydantic
**So that** I can extract domain-specific data structures

**Acceptance Criteria:**
- [ ] Any valid Pydantic model works as a schema
- [ ] Nested models are supported (e.g., Invoice with list[LineItem])
- [ ] Optional fields handled correctly
- [ ] Built-in schemas available for common use cases (Contact, Invoice, Entity)

### US-003: Question Answering
**As a** data analyst
**I want** to ask questions about a document
**So that** I can extract specific information without defining a full schema

**Acceptance Criteria:**
- [ ] `rlm.query("report.pdf", "What was Q3 revenue?")` returns answer
- [ ] Answer includes source citation with page number
- [ ] Works on multi-page documents

### US-004: Provider Flexibility
**As a** enterprise developer
**I want** to use my preferred LLM provider
**So that** I can comply with data governance policies

**Acceptance Criteria:**
- [ ] OpenRouter support (current default)
- [ ] Direct OpenAI support
- [ ] Direct Anthropic support
- [ ] Local Ollama support for air-gapped environments
- [ ] Easy provider switching via configuration

### US-005: Extraction Transparency
**As a** compliance officer
**I want** to see the reasoning behind each extraction
**So that** I can verify and audit the process

**Acceptance Criteria:**
- [ ] Result includes thinking_log with model's reasoning
- [ ] Result includes citations with verbatim quotes and page numbers
- [ ] Result includes confidence_history showing progress
- [ ] HTML visualization shows citations highlighted in source

### US-006: Session Persistence
**As a** developer processing large documents
**I want** to save and resume extraction sessions
**So that** I can recover from interruptions without starting over

**Acceptance Criteria:**
- [ ] `engine.save_session("extraction_task")` saves current state
- [ ] `engine.load_session("extraction_task")` resumes from saved state
- [ ] Saves: records, citations, thinking_log, progress
- [ ] Works across process restarts

### US-007: CLI Usage
**As a** command-line user
**I want** to extract data without writing code
**So that** I can quickly process documents

**Acceptance Criteria:**
- [ ] `rlm extract invoice.pdf --schema invoice` works
- [ ] `rlm query document.pdf "What is the total?"` works
- [ ] Output formats: JSON, CSV, HTML
- [ ] Verbose mode shows progress

---

## 3. Functional Requirements

### FR-001: Simple API
**Description:** Provide a minimal, intuitive API for common extraction tasks.
**Priority:** Must Have
**Dependencies:** None

```python
import rlm

# Simple extraction with built-in schema
result = rlm.extract("doc.pdf", schema=rlm.schemas.Contact)

# Custom schema
from pydantic import BaseModel
class Invoice(BaseModel):
    vendor: str
    amount: float
    date: str
    line_items: list[dict]
    page: int

result = rlm.extract("invoice.pdf", schema=Invoice)

# Q&A mode
answer = rlm.query("doc.pdf", "What was the total revenue?")
```

### FR-002: Provider Abstraction
**Description:** Abstract LLM providers behind a common interface to support multiple backends.
**Priority:** Must Have
**Dependencies:** None

**Supported Providers:**
- OpenRouter (default) - multi-model access
- OpenAI - direct API access
- Anthropic - direct API access
- Ollama - local/self-hosted models

**Configuration:**
```python
# Via environment variables (recommended)
# OPENROUTER_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY

# Via explicit configuration
engine = rlm.RLMEngine(
    root_model="anthropic/claude-sonnet-4.5",
    sub_model="openai/gpt-4o-mini",
    provider="openrouter"  # or "openai", "anthropic", "ollama"
)
```

### FR-003: Document Reading
**Description:** Read various document formats and convert to processable text.
**Priority:** Must Have
**Dependencies:** None

**Supported Formats:**
- PDF (via pymupdf/fitz or pdfplumber)
- PDF to Markdown (via Marker for high quality)
- Markdown files
- Plain text files

**Auto-detection:** File extension determines reader.

### FR-004: Parallel Extraction
**Description:** Extract from multiple document sections concurrently for performance.
**Priority:** Must Have
**Dependencies:** FR-001

**Features:**
- `llm_extract_parallel()` processes multiple sections simultaneously
- Configurable worker count (default: 5)
- Progress tracking per section
- Error handling per section (failures don't block others)

### FR-005: Reasoning Tools
**Description:** Provide tools for structured reasoning and evidence tracking.
**Priority:** Must Have
**Dependencies:** FR-001

**Tools:**
- `think(reasoning)` - Record reasoning step (logged for transparency)
- `cite(snippet, page, note)` - Record verbatim evidence citation
- `evaluate_progress(records, pages, issues)` - Self-assess confidence (0.0-1.0)

### FR-006: Session Persistence
**Description:** Save and restore extraction state for long-running tasks.
**Priority:** Should Have
**Dependencies:** FR-001

**Saved State:**
- Extracted records
- Citations
- Thinking log
- Confidence history
- Page progress

### FR-007: Visualization
**Description:** Generate HTML visualization of extraction results.
**Priority:** Should Have
**Dependencies:** FR-001, FR-005

**Features:**
- Show extracted records in table format
- Highlight source citations in document view
- Display reasoning trace
- Export to standalone HTML file

### FR-008: CLI Interface
**Description:** Command-line interface for document extraction.
**Priority:** Should Have
**Dependencies:** FR-001

**Commands:**
```bash
# Extract with schema
rlm extract invoice.pdf --schema invoice --output result.json

# Query document
rlm query document.pdf "What is the total amount?"

# Batch extraction
rlm extract *.pdf --schema contact --output-dir results/

# Visualization
rlm visualize result.json --output report.html
```

---

## 4. Non-Functional Requirements

### NFR-001: Performance
- Extraction should complete within 2 minutes for documents under 50 pages
- Parallel extraction should provide 3-5x speedup over sequential
- Memory usage should stay under 500MB for typical documents

### NFR-002: Reliability
- Zero JSON parsing failures (Pydantic validation handles all edge cases)
- Automatic retry on transient API errors (3 attempts with exponential backoff)
- Graceful degradation when sections fail (continue with others)
- Session recovery after process interruption

### NFR-003: Usability
- Single function call for common use cases
- Sensible defaults (no configuration required for basic usage)
- Clear error messages with actionable suggestions
- Comprehensive docstrings and type hints

### NFR-004: Maintainability
- Type hints on all public APIs
- 80%+ test coverage for core functionality
- Modular architecture (providers, readers, extractors are pluggable)
- No circular dependencies between modules

### NFR-005: Compatibility
- Python 3.10+ required
- Works on Windows, macOS, Linux
- No emojis in code/output (Windows compatibility)
- UTF-8 encoding throughout

---

## 5. Technical Constraints

- **Python 3.10+** required (uses modern type hints)
- **litellm or instructor** for LLM calls with structured output
- **Pydantic v2** for schema definition
- **No emojis** in code or comments (Windows console compatibility)
- **Environment variables** for API keys (not hard-coded)

---

## 6. Assumptions

- Users have valid API keys for their chosen provider
- Documents are well-formed (not corrupted/encrypted PDFs)
- Network access available for cloud providers
- Sufficient memory for document processing (~500MB)

---

## 7. Out of Scope

- Web UI or GUI application
- Database storage / persistence backend
- Multi-document cross-reference queries
- RAG / vector retrieval (pure RLM uses segmentation, not embeddings)
- Fine-tuning support for custom models
- Real-time streaming of results
- Batch processing of thousands of documents
- OCR for scanned/image-only PDFs (rely on existing PDF text extraction)

---

## 8. Open Questions

- [x] Should we support streaming extraction results? **No - not in v1**
- [x] Should we include caching? **Yes - optional, configurable**
- [x] What schema formats to include as built-ins? **Contact, Invoice, Entity, Table**
- [x] Should CLI support interactive mode for schema definition? **No - not in v1, use code**
- [x] Rate limiting strategy for parallel extraction? **Configurable max_workers (default 5), user responsibility**

---

## 9. Existing Code to Refactor

The following functionality already exists in `evals/generic_extract.py` (964 lines):

| Function | Lines | Status |
|----------|-------|--------|
| `read_pdf_as_text()` | 50-68 | Move to `document/reader.py` |
| `read_pdf_as_markdown()` | 71-113 | Move to `document/reader.py` |
| `llm_query()` | 375-381 | Move to `core/tools.py` |
| `llm_extract()` | 403-450 | Move to `extraction/structured.py` |
| `llm_extract_parallel()` | 452-511 | Move to `extraction/structured.py` |
| `get_section()` | 383-387 | Move to `core/tools.py` |
| `ask_about_section()` | 389-398 | Move to `core/tools.py` |
| `think()` | 587-600 | Move to `reasoning/tracer.py` |
| `cite()` | 602-615 | Move to `reasoning/tracer.py` |
| `evaluate_progress()` | 617-658 | Move to `reasoning/tracer.py` |
| `save_session()` | 660-676 | Move to `reasoning/session.py` |
| `load_session()` | 678-696 | Move to `reasoning/session.py` |
| `run_generic_extraction()` | 350-846 | Refactor into `core/engine.py` |

Also in `segmenter.py` (141 lines):
| Function | Status |
|----------|--------|
| `split_into_pages()` | Move to `document/segmenter.py` |
| `segment_document()` | Move to `document/segmenter.py` |
| `process_chunk()` | Move to `document/segmenter.py` |

---

> Next: Run `/spec:approve requirements` when this document is complete and reviewed.
