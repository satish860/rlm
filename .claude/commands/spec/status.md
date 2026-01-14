# /spec:status - Show Specification Status

You are generating a status report for all specifications or the current specification.

## Arguments
- `$ARGUMENTS` - Optional: Specific spec ID to show (e.g., "001" or "001-single-doc")

## Instructions

1. **Gather specification data**:
   - List all directories in `spec/` folder
   - For each spec, check:
     - Does `.requirements-approved` exist?
     - Does `.design-approved` exist?
     - Does `.tasks-approved` exist?
   - Read `tasks.md` if it exists to count completed tasks
   - Read `spec/.current-spec` to identify active spec

2. **Generate status report**:

```
Specification Status Report
===========================

Active Specification
--------------------
{If .current-spec exists}
[ACTIVE] {spec-id} - {spec-name}
{Else}
No active specification. Use /spec:switch to select one.

All Specifications
------------------

{spec-id}: {spec-name}
  Requirements: [Approved {date} | In Progress | Not Started]
  Design:       [Approved {date} | In Progress | Blocked]
  Tasks:        [Approved {date} | In Progress | Blocked]
  Implementation: [X/Y tasks complete (Z%) | Not Started | Blocked]

{Repeat for each spec...}

{If showing specific spec in detail}
Detailed Task Progress: {spec-name}
-----------------------------------

Phase 1: Foundation
  [x] T-001: {description}
  [ ] T-002: {description}
  Progress: 1/2 (50%)

Phase 2: Core Features
  [ ] T-003: {description}
  [ ] T-004: {description}
  Progress: 0/2 (0%)

{Continue for each phase...}

Overall Progress: X/Y tasks (Z%)

Recommended Next Action
-----------------------
{Context-aware recommendation}
```

3. **Provide actionable recommendations**:

   - If no specs exist:
     ```
     No specifications found. Run /spec:new {name} to create one.
     ```

   - If requirements not started:
     ```
     Run /spec:requirements to generate requirements.
     ```

   - If requirements done but not approved:
     ```
     Review requirements.md and run /spec:approve requirements
     ```

   - If design not started:
     ```
     Run /spec:design to generate technical design.
     ```

   - If tasks not started:
     ```
     Run /spec:tasks to generate task breakdown.
     ```

   - If implementation ready:
     ```
     Run /spec:implement to begin implementation.
     Suggested first task: T-XXX - {description}
     ```

## Output Format

### Multi-Spec Overview:
```
Specification Status Report
===========================

Active: 001-single-doc-indexer

Specifications (3 total):
-------------------------

001-single-doc-indexer [ACTIVE]
    Requirements: Approved (Jan 14)
    Design:       Approved (Jan 14)
    Tasks:        Approved (Jan 14)
    Implementation: 15/42 tasks (36%)

002-multi-doc-corpus
    Requirements: In Progress
    Design:       Blocked
    Tasks:        Blocked
    Implementation: Blocked

003-benchmark-runner
    Requirements: Not Started
    Design:       Blocked
    Tasks:        Blocked
    Implementation: Blocked

Recommended Next Action
-----------------------
Continue implementation on 001-single-doc-indexer.
Next task: T-016 - Implement section boundary detection
Run: /spec:implement T-016
```

### Single Spec Detail (when ID provided):
```
Specification Detail: 001-single-doc-indexer
============================================

Phase Status
------------
Requirements: Approved (Jan 14, 2026 10:30)
Design:       Approved (Jan 14, 2026 11:45)
Tasks:        Approved (Jan 14, 2026 12:00)

Implementation Progress
-----------------------

Phase 1: Foundation (5/5 complete - 100%)
  [x] T-001: Create directory structure
  [x] T-002: Set up base classes
  [x] T-003: Configure dependencies
  [x] T-004: Create __init__.py exports
  [x] T-005: Set up type hints

Phase 2: Core Features (8/15 complete - 53%)
  [x] T-006: Implement document loader
  [x] T-007: Implement TOC extractor
  [x] T-008: Create section mapper
  [ ] T-009: Build section summarizer  <-- NEXT
  [ ] T-010: Implement REPL environment
  ...

Phase 3: Integration (0/8 complete - 0%)
  [ ] T-020: Wire up components
  ...

Phase 4: Testing (0/10 complete - 0%)
  ...

Phase 5: Polish (0/4 complete - 0%)
  ...

Overall: 15/42 tasks (36%)

Recommended Next Action
-----------------------
Continue with T-009: Build section summarizer
Run: /spec:implement T-009
```

## Example Usage

```
/spec:status              # Show all specs overview
/spec:status 001          # Show detail for spec 001
/spec:status single-doc   # Show detail (partial name match)
```
