# /spec:update-task - Update Task Status

You are updating the status of a task in the current specification's task list.

## Arguments
- `$ARGUMENTS` - Task identifier: either task ID (T-001) or task description text

## Instructions

1. **Get current specification**:
   - Read `spec/.current-spec`
   - If no current spec, ask user to run `/spec:switch` first

2. **Read tasks.md**:
   - Read `spec/{current-spec}/tasks.md`
   - Parse the task list

3. **Find the task**:
   - If argument is `T-XXX` format: Match by task ID
   - If argument is text: Match by task description (partial match OK)
   - If multiple matches: List matches and ask user to clarify

4. **Toggle task status**:
   - If task is `[ ]` (incomplete): Change to `[x]` (complete)
   - If task is `[x]` (complete): Change to `[ ]` (incomplete)
   - Update the file

5. **Recalculate progress**:
   - Count completed tasks per phase
   - Update summary table at top of tasks.md
   - Calculate overall percentage

6. **Report the update**

## Output Format

### Task Marked Complete:
```
Task Completed: T-XXX
Description: {task description}

Phase {N} Progress: X/Y tasks (Z%)
Overall Progress: A/B tasks (C%)

{If next task in same phase}
Next task: T-YYY - {description}
{Else if phase complete}
Phase {N} complete! Moving to Phase {N+1}.
Next task: T-ZZZ - {description}
{Else if all complete}
All tasks complete! Implementation finished.
Run /spec:status for final summary.
```

### Task Marked Incomplete:
```
Task Reopened: T-XXX
Description: {task description}

Phase {N} Progress: X/Y tasks (Z%)
Overall Progress: A/B tasks (C%)
```

### Task Not Found:
```
Task not found: "{argument}"

Did you mean one of these?
- T-005: {similar description 1}
- T-012: {similar description 2}

Use exact task ID (T-XXX) or more specific description.
```

### Multiple Matches:
```
Multiple tasks match "{argument}":

1. T-005: {description 1}
2. T-012: {description 2}
3. T-018: {description 3}

Please specify by task ID (e.g., /spec:update-task T-005)
```

## Summary Table Update

When updating a task, also update the summary table at the top of tasks.md:

Before:
```markdown
| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Foundation | 5 tasks | 3/5 (60%) |
| Phase 2: Core Features | 15 tasks | 0/15 (0%) |
...
| **Total** | **42 tasks** | **3/42 (7%)** |
```

After completing T-004 in Phase 1:
```markdown
| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Foundation | 5 tasks | 4/5 (80%) |
| Phase 2: Core Features | 15 tasks | 0/15 (0%) |
...
| **Total** | **42 tasks** | **4/42 (10%)** |
```

## Example Usage

```
/spec:update-task T-005                    # By task ID
/spec:update-task "Create directory"       # By description
/spec:update-task implement section        # By partial match
```

## Notes

- Task IDs (T-XXX) are the most reliable way to identify tasks
- Partial description matches work but may be ambiguous
- The summary table is automatically updated
- Use `/spec:status` to see full progress report
