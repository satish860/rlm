# /spec:implement - Implementation Mode

You are entering implementation mode for the current specification. This is Phase 4 of Spec-Driven Development.

## Arguments
- `$ARGUMENTS` - Optional: Phase number (1-5) to focus on, or task ID (T-001)

## Prerequisites

1. Read `spec/.current-spec` to get the active specification
2. Check that `.tasks-approved` EXISTS (all planning phases must be complete)
3. If not approved, inform user to complete planning phases first

## Instructions

1. **Load specification context**:
   - Read `spec/{current-spec}/requirements.md` for what we're building
   - Read `spec/{current-spec}/design.md` for how to build it
   - Read `spec/{current-spec}/tasks.md` for the task breakdown
   - Read `CLAUDE.md` for project conventions

2. **Determine implementation scope**:
   - If `$ARGUMENTS` is a phase number (1-5): Focus on that phase's tasks
   - If `$ARGUMENTS` is a task ID (T-XXX): Focus on that specific task
   - If no arguments: Show task status and ask what to work on

3. **For each task being implemented**:

   a. **Before starting**:
      - Read the task description carefully
      - Identify dependencies (are prior tasks complete?)
      - Understand acceptance criteria

   b. **During implementation**:
      - Follow design.md specifications
      - Follow CLAUDE.md conventions
      - Write clean, documented code
      - Handle errors appropriately

   c. **After completing**:
      - Test the implementation
      - Update the task checkbox in tasks.md to [x]
      - Report what was done

4. **Track progress continuously**:
   - Use TodoWrite to track micro-tasks
   - Update tasks.md as tasks complete
   - Report progress to user

## Output Format

### Starting Implementation Session:
```
Implementation Mode: {spec-name}
Location: spec/{spec-id}-{spec-name}/

Current Progress:
- Phase 1 (Foundation): X/Y complete
- Phase 2 (Core Features): X/Y complete
- Phase 3 (Integration): X/Y complete
- Phase 4 (Testing): X/Y complete
- Phase 5 (Polish): X/Y complete
- Overall: X% complete

{If argument provided}
Focusing on: Phase {N} / Task {ID}

Ready to implement. Which task should we start with?
```

### Task Completion:
```
Completed: T-XXX - {task description}

Files modified:
- {file1}: {what was done}
- {file2}: {what was done}

Next task: T-YYY - {next task description}
Continue? (y/n)
```

### Session Summary:
```
Implementation Session Summary
------------------------------
Tasks completed: X
Files created: Y
Files modified: Z

Remaining tasks: N
Next suggested task: T-XXX

Run /spec:status for full progress report.
```

## Implementation Guidelines

1. **One task at a time**: Complete each task fully before moving on
2. **Test as you go**: Verify each task works before marking complete
3. **Update tasks.md**: Keep the task list current
4. **Follow the design**: Stick to the architecture in design.md
5. **Ask if unclear**: If a task is ambiguous, ask before implementing

## Example Usage

```
/spec:implement           # Show status, ask what to work on
/spec:implement 1         # Focus on Phase 1 tasks
/spec:implement 2         # Focus on Phase 2 tasks
/spec:implement T-005     # Work on specific task T-005
```

## Integration with TodoWrite

During implementation, use TodoWrite to track granular progress:
- Each task from tasks.md can be broken into sub-tasks
- Mark sub-tasks complete as you work
- This provides visibility into current progress

## Verification

Before marking any task complete:
1. Code compiles/runs without errors
2. Basic functionality works as expected
3. Follows project conventions
4. Has appropriate error handling
