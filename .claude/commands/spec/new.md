# /spec:new - Create a New Specification

You are creating a new specification for the RLM project. This command initializes a new specification directory with the proper structure.

## Arguments
- `$ARGUMENTS` - The name of the specification (e.g., "single-doc-indexer", "multi-doc-corpus")

## Instructions

1. **Determine the next specification ID**:
   - List existing directories in `spec/` folder
   - Find the highest numbered prefix (e.g., 001, 002)
   - Increment by 1 for the new spec (pad to 3 digits)
   - If no specs exist, start with 001

2. **Create the specification directory structure**:
   ```
   spec/
     {ID}-{name}/
       README.md           # Phase status dashboard
       requirements.md     # To be filled in Phase 1
       design.md          # To be filled in Phase 2
       tasks.md           # To be filled in Phase 3
   ```

3. **Initialize README.md** with this template:
   ```markdown
   # Specification: {name}

   **ID**: {ID}
   **Created**: {current date}
   **Status**: Requirements Phase (Not Started)

   ## Phase Status

   | Phase | Status | Approved |
   |-------|--------|----------|
   | Requirements | Not Started | - |
   | Design | Blocked | - |
   | Tasks | Blocked | - |
   | Implementation | Blocked | - |

   ## Overview

   [Brief description to be added]

   ## Quick Links

   - [Requirements](./requirements.md)
   - [Design](./design.md)
   - [Tasks](./tasks.md)
   ```

4. **Initialize requirements.md** with empty template:
   ```markdown
   # Requirements: {name}

   > Phase 1 of Spec-Driven Development
   > Run `/spec:requirements` to generate requirements, then `/spec:approve requirements` when complete.

   ## Status: Not Started

   ---

   [Requirements will be generated here]
   ```

5. **Initialize design.md** with empty template:
   ```markdown
   # Design: {name}

   > Phase 2 of Spec-Driven Development
   > Requires: Requirements approved
   > Run `/spec:design` to generate design, then `/spec:approve design` when complete.

   ## Status: Blocked (awaiting requirements approval)

   ---

   [Design will be generated here]
   ```

6. **Initialize tasks.md** with empty template:
   ```markdown
   # Tasks: {name}

   > Phase 3 of Spec-Driven Development
   > Requires: Design approved
   > Run `/spec:tasks` to generate tasks, then `/spec:approve tasks` when complete.

   ## Status: Blocked (awaiting design approval)

   ---

   [Tasks will be generated here]
   ```

7. **Set as current specification**:
   - Write the spec ID-name to `spec/.current-spec` file
   - Example: `001-single-doc-indexer`

8. **Report success**:
   ```
   Created new specification: {ID}-{name}
   Location: spec/{ID}-{name}/

   Next steps:
   1. Run /spec:requirements to generate requirements
   2. Review and refine requirements.md
   3. Run /spec:approve requirements when satisfied
   ```

## Validation

- If no name is provided in $ARGUMENTS, ask the user for a specification name
- Name should be lowercase with hyphens (kebab-case)
- Name should be descriptive but concise

## Example Usage

```
/spec:new single-doc-indexer
```

Creates:
```
spec/
  001-single-doc-indexer/
    README.md
    requirements.md
    design.md
    tasks.md
spec/.current-spec  (contains: 001-single-doc-indexer)
```
