# Requirements: Query Types (Extract & Summary)

> Phase 1 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-15

## 1. Overview

### 1.1 Purpose

The current `query()` method excels at Q&A-style document exploration but lacks:
- **Structured output** - Users need JSON, not free text, for programmatic use
- **Exhaustive extraction** - Q&A finds answers; extraction needs ALL items
- **Summarization** - Condensing content with configurable scope and style

This spec adds `extract()` and `summarize()` methods that build on the proven `query()` foundation using a three-stage pipeline.

### 1.2 Scope

**In Scope:**
- `extract()` method for structured JSON extraction
- `summarize()` method for document/section summarization
- Three-stage pipeline: Schema Generation -> Query -> JSON Conversion
- Two audience support: Plain English users and Developers (Pydantic)
- Integration with Instructor library for guaranteed JSON output
- Built-in Pydantic models for common extractions (papers, figures)

**Out of Scope:**
- Changes to existing `query()` method
- Changes to REPL executor or indexer
- Multi-document extraction/summarization
- Real-time streaming of results
- Custom fine-tuned models

### 1.3 Success Criteria

| Criterion | Target | Measurement |
|-----------|--------|-------------|
| Paper extraction accuracy | >95% | Extract RLM paper references, compare to manual count |
| Figure extraction completeness | 100% | All figures with correct captions |
| Summary length compliance | Within 10% | Word count vs target |
| Summary content coverage | All key points | Manual review of key observations |
| Schema generation success | >90% | Plain English descriptions produce valid schemas |
| JSON validation | 100% | All outputs pass Pydantic validation |

---

## 2. User Stories

### US-001: Extract References as JSON
**As a** researcher
**I want** to extract all cited papers from a PDF as JSON
**So that** I can build a bibliography or find related work programmatically

**Acceptance Criteria:**
- [ ] Can extract title, authors, year, URL for each reference
- [ ] Captures papers from References section
- [ ] Output is valid JSON array
- [ ] Works with plain English: `extract("all papers with title, authors, year")`
- [ ] Works with Pydantic: `extract("papers", response_model=PaperList)`

### US-002: Extract Figures with Captions
**As a** presenter
**I want** to extract all figures with their captions
**So that** I can reference them in slides or documentation

**Acceptance Criteria:**
- [ ] Extracts figure number and full caption text
- [ ] Identifies which section each figure appears in
- [ ] Output is valid JSON array
- [ ] Does not miss any figures in the document

### US-003: Quick Document Summary
**As a** busy reader
**I want** a quick executive summary of a paper
**So that** I can decide if it's worth reading in full

**Acceptance Criteria:**
- [ ] Generates summary within specified word limit
- [ ] Covers: problem, approach, key results, conclusions
- [ ] Supports multiple styles: paragraph, bullets, executive, abstract
- [ ] Returns markdown string for human reading

### US-004: Structured Summary for Database
**As a** developer building a paper database
**I want** structured summary data as JSON
**So that** I can store and query it programmatically

**Acceptance Criteria:**
- [ ] Returns dict with title, main_contribution, key_results, etc.
- [ ] All fields conform to StructuredSummary schema
- [ ] Can be serialized to JSON and stored in database

### US-005: Section-Specific Summary
**As a** reader
**I want** bullet-point summary of a specific section
**So that** I can quickly understand its content

**Acceptance Criteria:**
- [ ] Can target single section by name
- [ ] Can target multiple sections
- [ ] Generates concise bullet points
- [ ] Respects max_length parameter

### US-006: Plain English Extraction (No Code)
**As a** non-technical user
**I want** to describe what I need in plain English
**So that** I can extract data without knowing Python/Pydantic

**Acceptance Criteria:**
- [ ] System auto-generates schema from description
- [ ] Shows user the generated schema before proceeding
- [ ] Produces valid JSON output
- [ ] Works for arbitrary extraction types (not just papers/figures)

### US-007: Custom Pydantic Model Extraction
**As a** developer
**I want** to provide my own Pydantic model
**So that** I have full control over the output schema

**Acceptance Criteria:**
- [ ] Accepts any Pydantic BaseModel subclass
- [ ] Skips schema generation step
- [ ] Validates output against provided model
- [ ] Supports complex nested models

---

## 3. Functional Requirements

### FR-001: Three-Stage Pipeline
**Description:** Implement extraction as a three-stage pipeline:
- Stage 0: Schema generation from plain English (optional)
- Stage 1: Use existing query() for document exploration
- Stage 2: Convert free text to JSON via Instructor

**Priority:** Must Have
**Dependencies:** query() method, Instructor library

### FR-002: Extract Method
**Description:** Add `extract(what, response_model=None, on_error="raise", verbose=False)` method to SingleDocRLM
- If response_model is None, run Stage 0 to generate schema (with user confirmation)
- If response_model provided, skip Stage 0
- Always run Stage 1 (query) and Stage 2 (Instructor)
- For large extractions (50+ items), split into multiple queries
- on_error: "raise" (default) throws exception, "partial" returns what was extracted

**Priority:** Must Have
**Dependencies:** FR-001

### FR-003: Schema Generation
**Description:** Implement `generate_schema_from_description(what)` function
- Use LLM to parse plain English description
- Generate field names, types, and optionality
- Create Pydantic model dynamically using `create_model()`
- Display generated schema to user and ask for confirmation before proceeding

**Priority:** Must Have
**Dependencies:** Instructor library

### FR-004: Summarize Method
**Description:** Add `summarize(scope, style, max_length, structured=False, verbose=False)` method
- scope: "document", "section:NAME", "sections:A,B,C"
- style: "paragraph", "bullets", "executive", "abstract"
- max_length: target word count
- structured: if True, return dict via Instructor

**Priority:** Must Have
**Dependencies:** FR-001

### FR-005: Built-in Pydantic Models
**Description:** Provide common extraction models in `src/single_doc/models.py`:
- Paper, PaperList
- Figure, FigureList
- MethodStep, Methodology
- StructuredSummary

**Priority:** Should Have
**Dependencies:** Pydantic

### FR-006: Convenience Methods
**Description:** Add shortcut methods:
- `extract_papers()` - uses PaperList model
- `extract_figures()` - uses FigureList model

**Priority:** Should Have
**Dependencies:** FR-002, FR-005

### FR-007: Instructor Integration
**Description:** Integrate Instructor library with litellm for JSON conversion
- Use `instructor.from_litellm()` to patch completion
- Configure max_retries for validation failures
- Use SUB_MODEL for Stage 0 and Stage 2 (cheaper)

**Priority:** Must Have
**Dependencies:** Instructor library, litellm

### FR-008: Verbose Output
**Description:** When verbose=True, print execution details:
- Stage 0: Generated schema
- Stage 1: Query prompt, sections read
- Stage 2: Validation status, retries

**Priority:** Should Have
**Dependencies:** None

---

## 4. Non-Functional Requirements

### NFR-001: Performance
| Metric | Target |
|--------|--------|
| Extract 40 papers | < 90 seconds |
| Document summary | < 30 seconds |
| Schema generation | < 5 seconds |
| Stage 2 conversion | < 10 seconds |

### NFR-002: Accuracy
| Metric | Target |
|--------|--------|
| Extraction recall | > 95% (items found / items in document) |
| JSON validity | 100% (passes Pydantic validation) |
| Schema inference | > 90% (correct fields from description) |

### NFR-003: Cost Efficiency
- Stage 0 and Stage 2 should use SUB_MODEL (cheaper)
- Only Stage 1 uses ROOT_MODEL
- Total cost should be < 2x single query() cost

### NFR-004: Usability
- Plain English extraction should work without any code knowledge
- Error messages should be actionable
- Generated schema should be human-readable
- Default parameters should work for common cases

### NFR-005: Maintainability
- New code should not modify existing query() or REPL
- Pydantic models should be self-documenting
- Clear separation between stages

---

## 5. Technical Constraints

- Must use Python 3.10+ (for `type | None` syntax)
- Must integrate with existing litellm setup
- Must use Instructor library for JSON extraction
- Must use Pydantic v2 for model definitions
- Must work on Windows (no emojis in output)
- No changes to existing query(), REPL, or indexer code

---

## 6. Assumptions

- Instructor library is compatible with litellm and OpenRouter
- SUB_MODEL (Gemini Flash) supports function calling for Instructor
- Users have already built/loaded document index before calling extract/summarize
- Document sections are correctly identified by indexer

---

## 7. Out of Scope

- Modifying existing query() method behavior
- Modifying REPL executor or indexer
- Multi-document extraction (future spec)
- Real-time streaming of extraction results
- Custom model fine-tuning for extraction
- PDF image/chart extraction (visual elements)
- Caching extracted results in index file (future consideration)

---

## 8. Open Questions (Resolved)

- [x] **Large extractions**: Split queries and use multiple rounds for 50+ items
- [x] **Instructor model override**: No - internal detail, user shouldn't care about it
- [x] **Partial results on failure**: User choice via `on_error` parameter ("raise" | "partial")
- [x] **Schema confirmation**: Yes - ask user to confirm generated schema before proceeding
- [x] **Summary style presets**: No - current 4 styles (paragraph/bullets/executive/abstract) are sufficient

---

> Next: Run `/spec:approve requirements` when this document is complete and reviewed.
