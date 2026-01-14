# /spec:switch - Switch Active Specification

You are switching the active specification context. This changes which spec the other `/spec:*` commands operate on.

## Arguments
- `$ARGUMENTS` - The spec ID or name to switch to (e.g., "001", "single-doc", "001-single-doc-indexer")

## Instructions

1. **List available specifications**:
   - Read all directories in `spec/` folder
   - Each directory name is a spec (format: `{ID}-{name}`)

2. **Match the argument**:
   - If argument is a number (001, 002): Match by ID prefix
   - If argument is text: Match by name (partial match OK)
   - If no argument: List specs and ask user to choose

3. **Update current spec pointer**:
   - Write the full spec name to `spec/.current-spec`
   - Example: `001-single-doc-indexer`

4. **Report the switch and current status**

## Output Format

### When argument provided and matched:
```
Switched to: {spec-id} - {spec-name}

Current Status:
  Requirements: [Approved | In Progress | Not Started]
  Design:       [Approved | In Progress | Blocked]
  Tasks:        [Approved | In Progress | Blocked]
  Implementation: [X/Y tasks (Z%) | Not Started | Blocked]

{If implementation ready}
Next task: T-XXX - {description}
Run /spec:implement to continue.
{Else}
Next step: /spec:{next-phase}
```

### When no argument or no match:
```
Available Specifications:
-------------------------

1. 001-single-doc-indexer
   Status: Implementation (36% complete)

2. 002-multi-doc-corpus
   Status: Requirements (In Progress)

3. 003-benchmark-runner
   Status: Not Started

Current: {current spec or "None"}

Enter spec ID or name to switch:
```

### When argument doesn't match:
```
No specification matching "{argument}" found.

Available specifications:
- 001-single-doc-indexer
- 002-multi-doc-corpus
- 003-benchmark-runner

Use /spec:switch {id} or /spec:new {name} to create a new one.
```

## Example Usage

```
/spec:switch 001                    # Switch to spec 001
/spec:switch single-doc             # Switch by partial name
/spec:switch 002-multi-doc-corpus   # Switch by full name
/spec:switch                        # List and choose
```

## Notes

- Only one specification can be active at a time
- Switching specs does not affect any files or progress
- The active spec is stored in `spec/.current-spec`
- Other commands read this file to know which spec to operate on
