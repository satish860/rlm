# Design: rlm-library

> Phase 2 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-19
> Prerequisites: [Requirements Approved]

## 1. Overview

### 1.1 Design Goals

1. **Simplicity**: One-liner extraction with sensible defaults
2. **Transparency**: Full reasoning trace for every extraction
3. **Reliability**: Pydantic validation eliminates JSON parsing failures
4. **Performance**: Parallel extraction with configurable concurrency
5. **Extensibility**: Pluggable providers and document readers

### 1.2 Design Principles

- **Convention over Configuration**: Works out of the box with environment variables
- **Composition over Inheritance**: Small, composable components
- **Fail Fast**: Validate inputs early, provide clear error messages
- **No Magic**: Explicit is better than implicit
- **Refactor, Don't Rewrite**: Leverage existing `evals/generic_extract.py` code

---

## 2. Architecture

### 2.1 High-Level Architecture

```
                                 +------------------+
                                 |   User Code      |
                                 |  rlm.extract()   |
                                 +--------+---------+
                                          |
                                          v
+----------------+              +------------------+              +----------------+
|   Document     |              |    RLMEngine     |              |   Provider     |
|   Readers      +------------->|   (Orchestrator) +------------->|   Abstraction  |
|   PDF/MD/TXT   |              |                  |              |   OpenRouter   |
+----------------+              +--------+---------+              |   OpenAI, etc  |
                                         |                        +----------------+
                                         v
                               +------------------+
                               |   REPL Env       |
                               |   Namespace +    |
                               |   Tool Functions |
                               +--------+---------+
                                        |
                    +-------------------+-------------------+
                    |                   |                   |
                    v                   v                   v
            +-------------+     +-------------+     +-------------+
            | Extraction  |     |  Reasoning  |     |   Session   |
            | llm_extract |     |  think/cite |     |  save/load  |
            | parallel    |     |  evaluate   |     |             |
            +-------------+     +-------------+     +-------------+
```

### 2.2 Component Breakdown

#### Component: Core Engine (`rlm/core/`)
- **Purpose:** Orchestrates the extraction loop - manages messages, tool calls, iteration
- **Location:** `rlm/core/engine.py`
- **Dependencies:** Provider, Document, Extraction, Reasoning
- **Dependents:** Public API (`rlm/__init__.py`)

#### Component: REPL Environment (`rlm/core/`)
- **Purpose:** Manages the namespace where extraction code executes
- **Location:** `rlm/core/repl.py`
- **Dependencies:** Extraction tools, Reasoning tools
- **Dependents:** Core Engine

#### Component: Document Handling (`rlm/document/`)
- **Purpose:** Read and segment documents into processable chunks
- **Location:** `rlm/document/`
- **Dependencies:** pymupdf, pdfplumber, marker-pdf (optional)
- **Dependents:** Core Engine

#### Component: Extraction (`rlm/extraction/`)
- **Purpose:** Structured data extraction with Pydantic models
- **Location:** `rlm/extraction/`
- **Dependencies:** Provider, instructor
- **Dependents:** REPL Environment

#### Component: Reasoning (`rlm/reasoning/`)
- **Purpose:** Thinking, citations, progress evaluation, session management
- **Location:** `rlm/reasoning/`
- **Dependencies:** None (pure Python)
- **Dependents:** REPL Environment, Core Engine

#### Component: Providers (`rlm/providers/`)
- **Purpose:** Abstract LLM API calls across different backends
- **Location:** `rlm/providers/`
- **Dependencies:** openai, anthropic, litellm
- **Dependents:** Core Engine, Extraction

#### Component: Visualization (`rlm/visualization/`)
- **Purpose:** Generate HTML reports with citations highlighted
- **Location:** `rlm/visualization/`
- **Dependencies:** None (generates standalone HTML)
- **Dependents:** CLI, Public API

---

## 3. Data Flow

### 3.1 Primary Extraction Flow

```
Document (PDF/MD/TXT)
        |
        v
+------------------+
| Document Reader  |  read_document(path) -> str
+------------------+
        |
        v
+------------------+
| Segmenter        |  segment_document(text) -> List[Segment]
+------------------+
        |
        v
+------------------+
| RLMEngine        |  Builds system prompt with TOC
+------------------+
        |
        v
+------------------+
| Root Model       |  Decides extraction strategy
| (Claude Sonnet)  |  Calls execute_code tool
+------------------+
        |
        v
+------------------+
| REPL Execution   |  Runs extraction code in namespace
| - llm_extract()  |  Sub-model extracts structured data
| - think/cite()   |  Records reasoning and evidence
+------------------+
        |
        v
+------------------+
| Final Answer     |  Returns ExtractionResult
+------------------+
```

### 3.2 State Management

**REPL Namespace** holds all state during extraction:
```python
namespace = {
    # Document state
    "pages": List[str],           # Document split into pages
    "segments": List[Segment],    # TOC-like structure
    "total_pages": int,

    # Extraction state
    "records": List[dict],        # Accumulated extracted records
    "extracted_data": dict,       # Additional structured data

    # Reasoning state
    "thinking_log": List[dict],   # think() entries
    "citations": List[dict],      # cite() entries
    "confidence_history": List[dict],  # evaluate_progress() scores

    # Tool functions
    "llm_extract": Callable,
    "llm_extract_parallel": Callable,
    "think": Callable,
    "cite": Callable,
    # ... etc
}
```

---

## 4. API Design

### 4.1 Public API

```python
# rlm/__init__.py

from typing import Type, TypeVar, Union
from pathlib import Path
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

def extract(
    document: Union[str, Path],
    schema: Type[T],
    *,
    verbose: bool = False,
    max_iterations: int = 40,
    root_model: str = None,
    sub_model: str = None,
    provider: str = None,
) -> ExtractionResult[T]:
    """
    Extract structured data from a document.

    Args:
        document: Path to PDF, Markdown, or text file
        schema: Pydantic model class defining the extraction schema
        verbose: Print progress during extraction
        max_iterations: Maximum root model iterations
        root_model: Override root model (default: claude-sonnet-4.5)
        sub_model: Override sub-model (default: gpt-4o-mini)
        provider: Override provider (default: openrouter)

    Returns:
        ExtractionResult containing extracted data, citations, and reasoning trace

    Example:
        from pydantic import BaseModel
        import rlm

        class Contact(BaseModel):
            name: str
            email: str = None
            phone: str = None
            page: int

        result = rlm.extract("contacts.pdf", schema=list[Contact])
        for contact in result.data:
            print(contact.name, contact.email)
    """
    pass


def query(
    document: Union[str, Path],
    question: str,
    *,
    verbose: bool = False,
) -> QueryResult:
    """
    Ask a question about a document.

    Args:
        document: Path to PDF, Markdown, or text file
        question: Natural language question

    Returns:
        QueryResult with answer, citations, and confidence

    Example:
        result = rlm.query("report.pdf", "What was Q3 revenue?")
        print(result.answer)
        print(f"Source: page {result.citations[0].page}")
    """
    pass


class RLMEngine:
    """
    Advanced API for full control over extraction.

    Example:
        engine = rlm.RLMEngine(
            root_model="anthropic/claude-sonnet-4.5",
            sub_model="openai/gpt-4o-mini",
            provider="openrouter"
        )
        result = engine.extract("document.pdf", MySchema)
        engine.save_session("my_extraction")
    """

    def __init__(
        self,
        root_model: str = "anthropic/claude-sonnet-4.5",
        sub_model: str = "openai/gpt-4o-mini",
        provider: str = "openrouter",
        results_dir: Path = None,
        sessions_dir: Path = None,
    ):
        """Initialize engine with model configuration."""
        pass

    def extract(
        self,
        document: Union[str, Path],
        schema: Type[T],
        *,
        verbose: bool = False,
        max_iterations: int = 40,
    ) -> ExtractionResult[T]:
        """Extract structured data from document."""
        pass

    def query(
        self,
        document: Union[str, Path],
        question: str,
        *,
        verbose: bool = False,
    ) -> QueryResult:
        """Ask question about document."""
        pass

    def save_session(self, name: str) -> Path:
        """Save current extraction state."""
        pass

    def load_session(self, name: str) -> None:
        """Load previously saved state."""
        pass
```

### 4.2 Result Types

```python
# rlm/core/types.py

from typing import Generic, TypeVar, List
from pydantic import BaseModel
from datetime import datetime

T = TypeVar('T')


class Citation(BaseModel):
    """A verbatim citation from the source document."""
    snippet: str          # Exact text from document
    page: int             # Page number
    note: str = ""        # Interpretation or context


class ThinkingEntry(BaseModel):
    """A recorded reasoning step."""
    timestamp: datetime
    thought: str


class ConfidenceEntry(BaseModel):
    """A progress evaluation snapshot."""
    records: int
    pages_covered: int
    total_pages: int
    coverage: float
    issues: str
    notes: str
    confidence: float


class ExtractionResult(BaseModel, Generic[T]):
    """Result of an extraction operation."""
    data: List[T]                           # Extracted records
    schema_info: dict                       # Schema metadata
    verification: dict                      # Verification status

    # Reasoning trace
    citations: List[Citation]               # Source evidence
    thinking_log: List[ThinkingEntry]       # Model reasoning
    confidence_history: List[ConfidenceEntry]  # Progress over time

    # Metadata
    iterations: int                         # Iterations used
    data_file: str = None                   # If saved to file


class QueryResult(BaseModel):
    """Result of a query operation."""
    answer: str
    citations: List[Citation]
    confidence: float
    thinking_log: List[ThinkingEntry]
```

### 4.3 Provider Interface

```python
# rlm/providers/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Type
from pydantic import BaseModel


class BaseProvider(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: List[Dict] = None,
    ) -> Dict[str, Any]:
        """Send chat completion request."""
        pass

    @abstractmethod
    def extract(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        model: str,
    ) -> BaseModel:
        """Extract structured data using instructor."""
        pass


class OpenRouterProvider(BaseProvider):
    """OpenRouter API provider (default)."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key
        )
        self.instructor_client = instructor.from_openai(self.client)


class OpenAIProvider(BaseProvider):
    """Direct OpenAI API provider."""
    pass


class AnthropicProvider(BaseProvider):
    """Direct Anthropic API provider."""
    pass


class OllamaProvider(BaseProvider):
    """Local Ollama provider."""
    pass
```

---

## 5. Technology Decisions

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| LLM Calls | `openai` + `instructor` | Already used in codebase, Pydantic integration |
| Schema Definition | `pydantic` v2 | Type safety, validation, JSON serialization |
| PDF Reading | `pymupdf` / `pdfplumber` | Fast, reliable, already in use |
| PDF to Markdown | `marker-pdf` (optional) | High quality conversion when needed |
| CLI | `click` | Simple, composable, widely used |
| Progress Display | `tqdm` | Already in use for segmentation |

### 5.2 Design Patterns

| Pattern | Usage | Rationale |
|---------|-------|-----------|
| Strategy | Provider abstraction | Swap LLM backends without changing core logic |
| Factory | Document readers | Auto-detect format, return appropriate reader |
| Builder | System prompt | Construct complex prompts from segments |
| Observer | Progress callbacks | Decouple progress reporting from core logic |
| Memento | Session save/load | Capture and restore extraction state |

---

## 6. File Structure

```
rlm/
  __init__.py              # Public API: extract(), query(), RLMEngine
  core/
    __init__.py
    engine.py              # RLMEngine - main orchestration loop
    repl.py                # REPLEnvironment - namespace management
    tools.py               # Tool definitions (execute_code, final_answer)
    types.py               # ExtractionResult, QueryResult, Citation, etc.
    prompts.py             # System prompt builder
  document/
    __init__.py
    reader.py              # read_document(), DocumentReader ABC
    segmenter.py           # segment_document(), split_into_pages()
    formats/
      __init__.py
      pdf.py               # PDFReader (pymupdf, pdfplumber, marker)
      markdown.py          # MarkdownReader
      text.py              # TextReader
  extraction/
    __init__.py
    structured.py          # llm_extract(), llm_extract_parallel()
    schemas.py             # Built-in schemas: Contact, Invoice, Entity, Table
  reasoning/
    __init__.py
    tracer.py              # think(), cite(), evaluate_progress()
    session.py             # save_session(), load_session()
  providers/
    __init__.py
    base.py                # BaseProvider ABC
    openrouter.py          # OpenRouterProvider (default)
    openai.py              # OpenAIProvider
    anthropic.py           # AnthropicProvider
    ollama.py              # OllamaProvider
    factory.py             # get_provider(name) factory
  visualization/
    __init__.py
    html.py                # generate_html_report()
    citations.py           # highlight_citations()
  cli.py                   # Click-based CLI: rlm extract, rlm query
  exceptions.py            # RLMError, ExtractionError, ProviderError
  config.py                # Configuration management
```

---

## 7. Error Handling

### 7.1 Error Types

```python
# rlm/exceptions.py

class RLMError(Exception):
    """Base exception for all RLM errors."""
    pass


class DocumentError(RLMError):
    """Error reading or parsing document."""
    pass


class ProviderError(RLMError):
    """Error communicating with LLM provider."""
    pass


class ExtractionError(RLMError):
    """Error during extraction process."""
    pass


class SchemaError(RLMError):
    """Invalid or incompatible schema."""
    pass


class SessionError(RLMError):
    """Error saving or loading session."""
    pass
```

### 7.2 Error Recovery

| Error Type | Recovery Strategy |
|------------|-------------------|
| API timeout | Retry 3x with exponential backoff |
| Rate limit | Wait and retry with backoff |
| JSON parse error | Never happens (Pydantic handles) |
| Section extraction fails | Continue with other sections, log error |
| Document read error | Fail fast with clear message |
| Session load error | Start fresh, warn user |

### 7.3 Retry Logic

```python
# In providers/base.py

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((APITimeoutError, RateLimitError))
)
def chat_with_retry(self, ...):
    return self.chat(...)
```

---

## 8. Performance Considerations

### 8.1 Bottlenecks

1. **LLM API calls**: Primary bottleneck, mitigated by parallel extraction
2. **PDF parsing**: Can be slow for large documents, but only happens once
3. **Large documents**: Memory usage for storing all pages

### 8.2 Optimization Strategies

| Bottleneck | Strategy |
|------------|----------|
| Sequential extraction | `llm_extract_parallel()` with configurable workers |
| Large page ranges | Auto-chunk into 5-page segments |
| Repeated extractions | Optional caching of results |
| PDF conversion | Lazy loading, cache converted text |

### 8.3 Concurrency Model

```python
# Parallel extraction uses ThreadPoolExecutor
# Default: 5 concurrent API calls
# User-configurable via max_workers parameter

def llm_extract_parallel(
    sections: List[tuple],
    prompt_template: str,
    response_model: Type[T],
    max_workers: int = 5
) -> List[tuple]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(extract_one, s): s for s in sections}
        for future in as_completed(futures):
            yield future.result()
```

---

## 9. Security Considerations

1. **API Keys**: Never logged, stored in environment variables only
2. **Code Execution**: REPL runs in isolated namespace, no file system access beyond results/sessions dirs
3. **Input Validation**: All inputs validated via Pydantic before processing
4. **No Arbitrary Code**: Root model can only call predefined tools
5. **Output Sanitization**: HTML visualization escapes user content

---

## 10. Testing Strategy

### 10.1 Unit Tests

| Component | Test Focus |
|-----------|------------|
| `document/reader.py` | Format detection, text extraction |
| `document/segmenter.py` | Page splitting, segment detection |
| `extraction/structured.py` | Schema validation, parallel execution |
| `reasoning/tracer.py` | think/cite/evaluate state management |
| `reasoning/session.py` | Save/load round-trip |
| `providers/*` | API call formatting, error handling |

### 10.2 Integration Tests

| Test | Description |
|------|-------------|
| Full extraction flow | PDF -> extract -> validate results |
| Provider switching | Same extraction with different providers |
| Session persistence | Save, restart, load, continue |
| Error recovery | Simulate API failures, verify retry |

### 10.3 Test Data

```
tests/
  fixtures/
    sample_invoice.pdf
    sample_contacts.pdf
    sample_report.md
    sample_text.txt
  test_document.py
  test_extraction.py
  test_providers.py
  test_reasoning.py
  test_cli.py
```

---

## 11. Migration/Compatibility

### 11.1 Code Migration from `evals/generic_extract.py`

| Source | Destination | Changes |
|--------|-------------|---------|
| `read_pdf_as_text()` | `document/formats/pdf.py` | Add format detection |
| `read_pdf_as_markdown()` | `document/formats/pdf.py` | Same |
| `split_into_pages()` | `document/segmenter.py` | Already exists in segmenter.py |
| `segment_document()` | `document/segmenter.py` | Already exists in segmenter.py |
| `llm_extract()` | `extraction/structured.py` | Add provider abstraction |
| `llm_extract_parallel()` | `extraction/structured.py` | Same |
| `think()`, `cite()`, `evaluate_progress()` | `reasoning/tracer.py` | Extract to class |
| `save_session()`, `load_session()` | `reasoning/session.py` | Extract to class |
| `run_generic_extraction()` | `core/engine.py` | Major refactor into RLMEngine |
| `build_system_prompt()` | `core/prompts.py` | Templatize |
| `TOOLS` | `core/tools.py` | Same |

### 11.2 Backward Compatibility

- Existing `evals/generic_extract.py` continues to work (not modified)
- New library is separate package under `rlm/`
- Can import both during transition: `from evals.generic_extract import ...`

---

## 12. Open Design Questions

- [x] Should providers use `litellm` for unified interface? **No - use openai + instructor directly for control**
- [x] Should we support async extraction? **No - ThreadPoolExecutor sufficient for v1**
- [x] Where should default models be configured? **In config.py with env var overrides**

---

## 13. Demo Examples

### 13.1 Demo Strategy

Each demo should:
1. Work in < 10 lines of code
2. Produce compelling visual output (HTML with highlighted citations)
3. Show the reasoning trace
4. Be self-contained with sample data

### 13.2 Demo Structure

```
examples/
  01_invoice_extraction/
    sample_invoice.pdf       # Simple 2-page invoice
    extract_invoice.py       # 10-line extraction script
    output.html              # Generated visualization
    README.md                # Screenshot + explanation

  02_contacts_directory/
    agribusiness_contacts.pdf  # Real PDF (already have this)
    extract_contacts.py
    output.html
    README.md

  03_classic_literature/
    fetch_gutenberg.py       # Download Romeo & Juliet
    extract_characters.py    # Extract all characters + scenes
    output.html
    README.md

  04_earnings_report/
    quarterly_report.pdf     # Sample Q3 earnings
    extract_financials.py    # Revenue, expenses, guidance
    output.html
    README.md

  05_research_paper/
    sample_paper.pdf         # Academic paper
    extract_findings.py      # Methods, results, citations
    output.html
    README.md
```

### 13.3 Demo Script Template

```python
# examples/01_invoice_extraction/extract_invoice.py
"""Invoice extraction demo - 10 lines of code"""

import rlm
from rlm.schemas import Invoice

result = rlm.extract("sample_invoice.pdf", schema=Invoice, verbose=True)

print(f"Vendor: {result.data[0].vendor}")
print(f"Total: ${result.data[0].total_amount}")
print(f"Confidence: {result.confidence_history[-1].confidence:.0%}")

rlm.visualize(result, output="output.html")
```

### 13.4 Visualization Output

HTML visualization should show:
1. **Extracted Data** - Table of records with all fields
2. **Source Document** - With citations highlighted in yellow
3. **Reasoning Trace** - Collapsible thinking log
4. **Confidence Score** - Progress bar showing extraction confidence
5. **Citation Links** - Click citation to jump to source location

### 13.5 Benchmark Comparison Demo

```
examples/benchmark/
  compare_langextract.py    # Side-by-side comparison
  romeo_juliet_benchmark/
    rlm_result.json
    langextract_result.json
    comparison_report.html
```

Metrics to compare:
- Extraction accuracy (F1)
- Citation grounding accuracy
- Processing time
- Token usage / cost

---

## 14. Configuration

```python
# rlm/config.py

import os
from dataclasses import dataclass


@dataclass
class RLMConfig:
    """Global configuration with environment variable overrides."""

    # Models
    root_model: str = "anthropic/claude-sonnet-4.5"
    sub_model: str = "openai/gpt-4o-mini"

    # Provider
    provider: str = "openrouter"

    # API Keys (from environment)
    openrouter_api_key: str = None
    openai_api_key: str = None
    anthropic_api_key: str = None

    # Directories
    results_dir: str = "results"
    sessions_dir: str = "sessions"

    # Extraction settings
    max_iterations: int = 40
    parallel_workers: int = 5
    page_chunk_size: int = 5

    @classmethod
    def from_env(cls) -> "RLMConfig":
        """Load config from environment variables."""
        return cls(
            root_model=os.getenv("RLM_ROOT_MODEL", cls.root_model),
            sub_model=os.getenv("RLM_SUB_MODEL", cls.sub_model),
            provider=os.getenv("RLM_PROVIDER", cls.provider),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
```

---

> Next: Run `/spec:approve design` when this document is complete and reviewed.
