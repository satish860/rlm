# Design: Query Types (Extract & Summary)

> Phase 2 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-15
> Prerequisites: [Requirements Approved]

## 1. Overview

### 1.1 Design Goals

1. **Minimal Invasiveness** - Add new methods without modifying existing query(), REPL, or indexer
2. **Layered Architecture** - Three-stage pipeline with clear separation of concerns
3. **Dual Audience** - Support both plain English users and developers with Pydantic models
4. **Guaranteed Output** - Use Instructor for validated JSON, never return malformed data

### 1.2 Design Principles

| Principle | Application |
|-----------|-------------|
| **Composition over Modification** | Build on query(), don't change it |
| **Fail Fast, Fail Clearly** | Validate schemas upfront, clear error messages |
| **Progressive Enhancement** | Plain English works, Pydantic adds control |
| **Cost Awareness** | Use SUB_MODEL for non-critical stages |

---

## 2. Architecture

### 2.1 High-Level Architecture

```
                                 SingleDocRLM
                                      |
            +------------+------------+------------+
            |            |            |            |
         query()    extract()   summarize()   extract_papers()
            |            |            |            |
            |     +------+------+     |            |
            |     |      |      |     |            |
            |  Stage0  Stage1  Stage2 |            |
            |  Schema  Query   JSON   |            |
            |    |       |       |    |            |
            |    v       v       v    |            |
            |  [LLM]  [query()] [Instructor]       |
            |    |       |       |    |            |
            +----+-------+-------+----+            |
                         |                         |
                    Pydantic Models <--------------+
                    (models.py)
```

### 2.2 Component Breakdown

#### Component: SingleDocRLM (Extended)
- **Purpose:** Main API class, extended with extract() and summarize() methods
- **Location:** `src/single_doc/rlm.py`
- **Dependencies:** query(), Instructor, models.py
- **Changes:** Add methods only, no modification to existing code

#### Component: Schema Generator
- **Purpose:** Convert plain English descriptions to Pydantic models
- **Location:** `src/single_doc/schema_gen.py`
- **Dependencies:** Instructor, Pydantic create_model
- **Dependents:** extract() when response_model=None

#### Component: JSON Converter
- **Purpose:** Convert free text to validated JSON via Instructor
- **Location:** `src/single_doc/json_convert.py`
- **Dependencies:** Instructor, litellm, Pydantic models
- **Dependents:** extract(), summarize(structured=True)

#### Component: Pydantic Models
- **Purpose:** Built-in extraction schemas (Paper, Figure, etc.)
- **Location:** `src/single_doc/models.py`
- **Dependencies:** Pydantic v2
- **Dependents:** extract_papers(), extract_figures(), summarize(structured=True)

---

## 3. Data Flow

### 3.1 Extract Data Flow

```
extract(what="papers with title, authors", response_model=None)
    |
    v
[Stage 0: Schema Generation] -----> User Confirmation -----> Reject? --> Error
    |                                      |
    | Generated PaperList model            | Confirmed
    v                                      v
[Stage 1: Query] <-- prompt built from model fields
    |
    | Free text: "1. Paper A by Author X, 2024..."
    v
[Stage 2: JSON Conversion] <-- Instructor with PaperList model
    |
    | Validated JSON or ValidationError
    v
[on_error handling]
    |
    +--> "raise": raise ExtractionError
    +--> "partial": return what was extracted
    |
    v
list[dict]
```

### 3.2 Summarize Data Flow

```
summarize(scope="document", style="executive", structured=False)
    |
    v
[Parse Scope] --> "document" | "section:NAME" | "sections:A,B,C"
    |
    v
[Build Prompt] <-- style templates
    |
    v
[Stage 1: Query] --> Free text summary
    |
    v
[structured=True?]
    |
    +--> No:  Return markdown string
    +--> Yes: [Stage 2: JSON Conversion] --> dict
```

### 3.3 State Management

- No new state added to SingleDocRLM
- Stateless functions for schema_gen and json_convert
- All state in existing index, executor, system_prompt

---

## 4. API Design

### 4.1 Public API

```python
class SingleDocRLM:
    # ... existing methods ...

    def extract(
        self,
        what: str,
        response_model: type[BaseModel] | None = None,
        on_error: Literal["raise", "partial"] = "raise",
        verbose: bool = False,
    ) -> list[dict]:
        """
        Extract structured data from document.

        Args:
            what: Description of what to extract (e.g., "papers with title, authors, year")
            response_model: Pydantic model for output. If None, auto-generates from 'what'.
            on_error: "raise" throws on failure, "partial" returns extracted items
            verbose: Print execution details

        Returns:
            List of extracted items as dictionaries

        Raises:
            ExtractionError: If extraction fails and on_error="raise"
            SchemaGenerationError: If auto-schema generation fails
            UserCancelledError: If user rejects generated schema

        Example:
            # Plain English
            papers = rlm.extract("all papers with title, authors, year, url")

            # With Pydantic model
            papers = rlm.extract("papers", response_model=PaperList)
        """
        pass

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
            style: Summary style
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
        pass

    def extract_papers(self, verbose: bool = False) -> list[dict]:
        """Convenience method: Extract all referenced papers."""
        return self.extract("referenced papers", response_model=PaperList, verbose=verbose)

    def extract_figures(self, verbose: bool = False) -> list[dict]:
        """Convenience method: Extract all figures with captions."""
        return self.extract("figures with captions", response_model=FigureList, verbose=verbose)
```

### 4.2 Internal APIs

```python
# src/single_doc/schema_gen.py

def generate_schema_from_description(
    what: str,
    model: str = SUB_MODEL,
) -> type[BaseModel]:
    """
    Generate Pydantic model from plain English description.

    Args:
        what: Plain English description (e.g., "papers with title, authors")
        model: LLM model for schema inference

    Returns:
        Dynamically created Pydantic model class
    """
    pass

def confirm_schema_with_user(
    schema: type[BaseModel],
    what: str,
) -> bool:
    """
    Display generated schema and ask user for confirmation.

    Returns:
        True if confirmed, False if rejected
    """
    pass


# src/single_doc/json_convert.py

def convert_to_json(
    free_text: str,
    response_model: type[BaseModel],
    model: str = SUB_MODEL,
    max_retries: int = 3,
) -> BaseModel:
    """
    Convert free text to validated Pydantic model using Instructor.

    Args:
        free_text: Text to convert
        response_model: Target Pydantic model
        model: LLM for conversion
        max_retries: Retry attempts on validation failure

    Returns:
        Populated Pydantic model instance

    Raises:
        JSONConversionError: If conversion fails after retries
    """
    pass
```

---

## 5. Technology Decisions

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| JSON Extraction | Instructor | Automatic validation, retries, type coercion |
| Schema Definition | Pydantic v2 | Industry standard, self-documenting |
| Dynamic Models | pydantic.create_model | Generate models from LLM output |
| LLM Calls | litellm | Already used, unified interface |
| User Input | Python input() | Simple, no new dependencies |

### 5.2 Design Patterns

| Pattern | Usage | Rationale |
|---------|-------|-----------|
| Pipeline | Three-stage extraction | Clear separation, testable stages |
| Strategy | on_error parameter | User controls failure behavior |
| Factory | create_model() | Dynamic schema generation |
| Facade | extract_papers() | Simple interface for common cases |

---

## 6. File Structure

```
src/single_doc/
    __init__.py          # Export new classes/functions
    rlm.py               # Add extract(), summarize() methods
    models.py            # NEW: Pydantic models (Paper, Figure, etc.)
    schema_gen.py        # NEW: Schema generation from plain English
    json_convert.py      # NEW: Instructor-based JSON conversion
    errors.py            # NEW: Custom exceptions
```

### 6.1 New Files Detail

**models.py**
```python
from pydantic import BaseModel, Field
from typing import Optional

class Paper(BaseModel):
    """A referenced paper/citation."""
    title: str = Field(description="Paper title")
    authors: list[str] = Field(description="Author names")
    year: str = Field(description="Publication year")
    venue: Optional[str] = Field(default=None, description="Journal/conference")
    url: Optional[str] = Field(default=None, description="URL or arXiv link")

class PaperList(BaseModel):
    """Container for extracted papers."""
    items: list[Paper]

class Figure(BaseModel):
    """A figure in the document."""
    number: str = Field(description="Figure number")
    caption: str = Field(description="Caption text")
    section: Optional[str] = Field(default=None, description="Section name")

class FigureList(BaseModel):
    """Container for extracted figures."""
    items: list[Figure]

class StructuredSummary(BaseModel):
    """Structured document summary."""
    title: str
    main_contribution: str
    problem: str
    approach: str
    key_results: list[str]
    limitations: list[str] = []
```

**errors.py**
```python
class ExtractionError(Exception):
    """Extraction failed."""
    pass

class SchemaGenerationError(Exception):
    """Schema generation failed."""
    pass

class JSONConversionError(Exception):
    """JSON conversion failed."""
    pass

class UserCancelledError(Exception):
    """User cancelled operation."""
    pass
```

---

## 7. Error Handling

### 7.1 Error Types

| Error | When | Action |
|-------|------|--------|
| SchemaGenerationError | LLM fails to generate valid schema | Raise with message |
| UserCancelledError | User rejects generated schema | Raise, allow retry |
| JSONConversionError | Instructor fails after retries | Depends on on_error |
| ExtractionError | Stage 1 query fails | Raise with details |

### 7.2 Error Recovery

```python
# on_error="raise" (default)
try:
    result = convert_to_json(text, model)
except JSONConversionError as e:
    raise ExtractionError(f"Failed to extract: {e}")

# on_error="partial"
try:
    result = convert_to_json(text, model)
except JSONConversionError as e:
    # Try to salvage partial results
    partial = extract_partial_from_text(text, model)
    print(f"Warning: Partial extraction ({len(partial)} items)")
    return partial
```

### 7.3 User Confirmation Flow

```
Generated schema for: papers with title, authors, year

  Paper:
    - title: str (required)
    - authors: list[str] (required)
    - year: str (required)

Proceed with this schema? [Y/n]:
```

---

## 8. Performance Considerations

### 8.1 Bottlenecks

| Stage | Bottleneck | Mitigation |
|-------|------------|------------|
| Stage 0 | LLM call for schema | Use fast SUB_MODEL |
| Stage 1 | Multiple REPL rounds | Existing query() optimization |
| Stage 2 | Instructor retries | Cap retries at 3 |
| Large extractions | 50+ items | Split into section-based queries |

### 8.2 Cost Optimization

```python
# Stage 0 and 2 use cheaper model
EXTRACTION_MODEL = SUB_MODEL  # e.g., gemini-3-flash

# Stage 1 uses smarter model (existing behavior)
ROOT_MODEL  # e.g., claude-sonnet-4.5
```

### 8.3 Large Extraction Strategy

```python
def _extract_large(self, what, model, sections=None):
    """Handle extractions with 50+ expected items."""
    all_items = []

    # Split by section
    target_sections = sections or self._identify_relevant_sections(what)

    for section in target_sections:
        # Query per section
        prompt = f"List all {what} in section '{section.title}'"
        text = self.query(prompt)

        # Convert and accumulate
        items = convert_to_json(text, model)
        all_items.extend(items.items)

    # Deduplicate
    return self._deduplicate(all_items)
```

---

## 9. Security Considerations

| Concern | Mitigation |
|---------|------------|
| Prompt injection in 'what' | Schema generation is sandboxed, doesn't execute code |
| Malformed user input | Pydantic validation catches invalid data |
| Cost attacks (huge extractions) | Section-based splitting limits per-query cost |

---

## 10. Testing Strategy

### 10.1 Unit Tests

```python
# test_schema_gen.py
def test_generate_schema_simple():
    """Test schema generation from simple description."""
    schema = generate_schema_from_description("papers with title, authors, year")
    assert hasattr(schema, 'model_fields')
    assert 'title' in schema.model_fields

def test_generate_schema_with_optional():
    """Test optional field detection."""
    schema = generate_schema_from_description("papers with title, optional url")
    assert schema.model_fields['url'].is_required() == False

# test_json_convert.py
def test_convert_simple():
    """Test JSON conversion of simple list."""
    text = "1. Paper A by Smith, 2024\n2. Paper B by Jones, 2023"
    result = convert_to_json(text, PaperList)
    assert len(result.items) == 2

def test_convert_retry_on_failure():
    """Test automatic retry on validation failure."""
    # Mock LLM to fail first, succeed second
    ...
```

### 10.2 Integration Tests

```python
# test_extract_integration.py
def test_extract_papers_from_rlm_paper():
    """End-to-end: Extract papers from RLM paper."""
    rlm = SingleDocRLM("recursive_language_models.pdf")
    rlm.load_index("temp/recursive_language_models.index.json")

    papers = rlm.extract_papers()

    assert len(papers) >= 30  # Paper has 30+ references
    assert all('title' in p for p in papers)
    assert all('year' in p for p in papers)

def test_summarize_document():
    """End-to-end: Summarize document."""
    rlm = SingleDocRLM("recursive_language_models.pdf")
    rlm.load_index("temp/recursive_language_models.index.json")

    summary = rlm.summarize(style="executive", max_length=300)

    words = len(summary.split())
    assert 250 <= words <= 350  # Within 10% of target
    assert "RLM" in summary or "Recursive" in summary
```

---

## 11. Migration/Compatibility

### 11.1 No Breaking Changes

- Existing query() method unchanged
- Existing SingleDocRLM API unchanged
- New methods are purely additive

### 11.2 New Dependencies

```
# requirements.txt additions
instructor>=1.0.0
pydantic>=2.0.0  # May already be present
```

### 11.3 Import Changes

```python
# Users can optionally import models
from src.single_doc.models import Paper, PaperList, Figure, FigureList
from src.single_doc.models import StructuredSummary
```

---

## 12. Open Design Questions

- [x] All design questions resolved in requirements phase

---

> Next: Run `/spec:approve design` when this document is complete and reviewed.
