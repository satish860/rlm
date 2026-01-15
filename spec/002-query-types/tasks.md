# Tasks: Query Types (Extract & Summary)

> Phase 3 of Spec-Driven Development
> Status: Draft
> Last Updated: 2026-01-15
> Prerequisites: [Design Approved]

## Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Foundation | 4 tasks | Complete |
| Phase 2: Core Components | 6 tasks | Complete |
| Phase 3: Main Methods | 4 tasks | 3/4 Complete |
| Phase 4: Integration | 3 tasks | Complete |
| Phase 5: Testing | 4 tasks | Not Started |
| **Total** | **21 tasks** | **80% Complete** |

---

## Phase 1: Foundation

> Setup dependencies and base files

### 1.1 Dependencies

- [x] **T-001**: Add Instructor dependency
  - Add `instructor>=1.0.0` to requirements.txt
  - Verify pydantic>=2.0.0 is present
  - Run `pip install -r requirements.txt`
  - Test import: `import instructor`

### 1.2 Error Handling

- [x] **T-002**: Create errors.py with custom exceptions
  - File: `src/single_doc/errors.py`
  - Classes: ExtractionError, SchemaGenerationError, JSONConversionError, UserCancelledError
  - Each with clear docstring
  - Export in `__init__.py`

### 1.3 Pydantic Models

- [x] **T-003**: Create models.py with built-in extraction schemas
  - File: `src/single_doc/models.py`
  - Models: Paper, PaperList, Figure, FigureList, StructuredSummary
  - Use Field() with descriptions for Instructor
  - Export in `__init__.py`

### 1.4 Module Exports

- [x] **T-004**: Update __init__.py with new exports
  - File: `src/single_doc/__init__.py`
  - Export: errors, models, and new methods (after implementation)
  - Dependencies: T-002, T-003

---

## Phase 2: Core Components

> Implement the three pipeline stages

### 2.1 JSON Conversion (Stage 2)

- [x] **T-005**: Create json_convert.py with Instructor integration
  - File: `src/single_doc/json_convert.py`
  - Function: `convert_to_json(free_text, response_model, model, max_retries)`
  - Use `instructor.from_litellm()` to patch litellm
  - Handle validation errors, return populated model
  - Raise JSONConversionError on failure after retries
  - Dependencies: T-001, T-002

- [x] **T-006**: Add partial extraction fallback
  - Function: `extract_partial_from_text(text, model)`
  - Try to parse whatever was extracted before failure
  - Return list of successfully parsed items
  - Dependencies: T-005

### 2.2 Schema Generation (Stage 0)

- [x] **T-007**: Create schema_gen.py with LLM-based schema inference
  - File: `src/single_doc/schema_gen.py`
  - Define SchemaDefinition and FieldDef internal models
  - Function: `generate_schema_from_description(what, model)`
  - Use Instructor to get structured schema definition
  - Use pydantic.create_model() to create dynamic model
  - Raise SchemaGenerationError on failure
  - Dependencies: T-001, T-002, T-005

- [x] **T-008**: Add user confirmation flow
  - Function: `confirm_schema_with_user(schema, what)`
  - Display generated schema in human-readable format
  - Prompt user with `[Y/n]`
  - Return True/False, raise UserCancelledError if rejected
  - Dependencies: T-007

- [x] **T-009**: Create schema display formatter
  - Function: `format_schema_for_display(schema)`
  - Show field names, types, required/optional
  - Clean output without Pydantic internals
  - Dependencies: T-007

### 2.3 Prompt Builders

- [x] **T-010**: Create extraction prompt builder
  - Function: `build_extraction_prompt(what, model_fields)`
  - Template: "List ALL {what}... For each include: {fields}..."
  - Extract field names and types from Pydantic model
  - Dependencies: T-003

---

## Phase 3: Main Methods

> Implement extract() and summarize() on SingleDocRLM

### 3.1 Extract Method

- [x] **T-011**: Implement extract() method skeleton
  - File: `src/single_doc/rlm.py`
  - Method signature per design.md
  - Parameter validation
  - Route to schema gen or direct extraction
  - Dependencies: T-005, T-007, T-010

- [x] **T-012**: Implement extract() full pipeline
  - Stage 0: Call schema gen if response_model=None
  - Stage 1: Build prompt, call self.query()
  - Stage 2: Convert result with convert_to_json()
  - Handle on_error parameter
  - Dependencies: T-011

- [ ] **T-013**: Add large extraction handling (deferred)
  - Detect when 50+ items expected (heuristic or explicit)
  - Split by relevant sections
  - Accumulate and deduplicate results
  - Dependencies: T-012

### 3.2 Summarize Method

- [x] **T-014**: Implement summarize() method
  - File: `src/single_doc/rlm.py`
  - Parse scope parameter (document/section/sections)
  - Build style-specific prompt
  - Call self.query() for free text
  - If structured=True, convert to StructuredSummary
  - Dependencies: T-003, T-005

---

## Phase 4: Integration

> Wire everything together and add convenience methods

### 4.1 Convenience Methods

- [x] **T-015**: Add extract_papers() convenience method
  - Calls extract() with PaperList model
  - Returns list[dict]
  - Dependencies: T-012

- [x] **T-016**: Add extract_figures() convenience method
  - Calls extract() with FigureList model
  - Returns list[dict]
  - Dependencies: T-012

### 4.2 Final Integration

- [x] **T-017**: Update __init__.py with all exports
  - Export extract, summarize from rlm.py
  - Export models
  - Export errors
  - Verify all imports work
  - Dependencies: T-014, T-015, T-016

---

## Phase 5: Testing

> Verify everything works

### 5.1 Unit Tests

- [ ] **T-018**: Write unit tests for json_convert.py
  - File: `tests/test_json_convert.py`
  - Test convert_to_json with valid input
  - Test retry behavior
  - Test partial extraction
  - Mock LLM calls
  - Dependencies: T-005, T-006

- [ ] **T-019**: Write unit tests for schema_gen.py
  - File: `tests/test_schema_gen.py`
  - Test generate_schema_from_description
  - Test field type inference (str, list, optional)
  - Test schema display formatting
  - Mock LLM calls
  - Dependencies: T-007, T-008, T-009

### 5.2 Integration Tests

- [ ] **T-020**: Write integration tests for extract()
  - File: `tests/test_extract_integration.py`
  - Test extract_papers() on RLM paper
  - Test plain English extraction
  - Test on_error="partial" behavior
  - Requires: RLM paper index built
  - Dependencies: T-012, T-015

- [ ] **T-021**: Write integration tests for summarize()
  - File: `tests/test_summarize_integration.py`
  - Test document summary (all styles)
  - Test section summary
  - Test structured=True output
  - Verify word count within tolerance
  - Dependencies: T-014

---

## Task Dependencies

```
Phase 1: Foundation
T-001 (Instructor) ----+
                       |
T-002 (errors.py) -----+--> T-004 (__init__.py)
                       |
T-003 (models.py) -----+

Phase 2: Core Components
T-001 + T-002 --> T-005 (json_convert) --> T-006 (partial)
                      |
                      v
              T-007 (schema_gen) --> T-008 (confirm) --> T-009 (format)
                      |
T-003 ----------------+--> T-010 (prompt builder)

Phase 3: Main Methods
T-005 + T-007 + T-010 --> T-011 (extract skeleton)
                              |
                              v
                         T-012 (extract full) --> T-013 (large)
                              |
T-003 + T-005 ----------------+--> T-014 (summarize)

Phase 4: Integration
T-012 --> T-015 (extract_papers)
T-012 --> T-016 (extract_figures)
T-014 + T-015 + T-016 --> T-017 (final __init__)

Phase 5: Testing
T-005 + T-006 --> T-018 (test json_convert)
T-007 + T-008 + T-009 --> T-019 (test schema_gen)
T-012 + T-015 --> T-020 (test extract)
T-014 --> T-021 (test summarize)
```

---

## Parallel Work Opportunities

These task groups can be worked on in parallel:

**Group A** (Core conversion):
- T-005, T-006 (json_convert)

**Group B** (Schema generation):
- T-007, T-008, T-009 (schema_gen)

**Group C** (Models):
- T-002, T-003 (errors, models)

After Groups A, B, C complete:
- T-010, T-011, T-012 (extract)
- T-014 (summarize)

---

## Estimated Effort

| Phase | Estimated Hours |
|-------|-----------------|
| Phase 1: Foundation | 1-2 hours |
| Phase 2: Core Components | 3-4 hours |
| Phase 3: Main Methods | 2-3 hours |
| Phase 4: Integration | 1 hour |
| Phase 5: Testing | 2-3 hours |
| **Total** | **9-13 hours** |

---

## Notes

- Tasks should be completed in order unless dependencies allow parallel work
- Each task should be small enough to complete in one focused session
- Mark tasks complete with [x] as you finish them
- Run tests after each phase to catch issues early
- The RLM paper index (`temp/recursive_language_models.index.json`) is needed for integration tests

---

> Next: Run `/spec:approve tasks` when this breakdown is complete and reviewed.
