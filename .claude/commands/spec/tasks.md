# /spec:tasks - Generate Task Breakdown

You are generating a detailed task breakdown for the current specification. This is Phase 3 of Spec-Driven Development.

## Prerequisites

1. Read `spec/.current-spec` to get the active specification
2. Check that `.design-approved` EXISTS (design must be approved first)
3. Check that `.tasks-approved` does NOT exist (tasks not yet approved)
4. If design not approved, inform user to run `/spec:approve design` first

## Instructions

1. **Read the specification context**:
   - Read `spec/{current-spec}/requirements.md` for what needs to be built
   - Read `spec/{current-spec}/design.md` for how it should be built
   - Read `spec/{current-spec}/README.md`
   - Read `CLAUDE.md` for project conventions

2. **Generate comprehensive tasks.md** with this structure:

```markdown
# Tasks: {spec-name}

> Phase 3 of Spec-Driven Development
> Status: Draft
> Last Updated: {date}
> Prerequisites: [Design Approved]

## Summary

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1: Foundation | X tasks | Not Started |
| Phase 2: Core Features | X tasks | Not Started |
| Phase 3: Integration | X tasks | Not Started |
| Phase 4: Testing | X tasks | Not Started |
| Phase 5: Polish | X tasks | Not Started |
| **Total** | **X tasks** | **0% Complete** |

---

## Phase 1: Foundation

> Setup and infrastructure tasks

### 1.1 Project Setup
- [ ] **T-001**: Create directory structure per design.md
  - Create `src/{module}/` directory
  - Add `__init__.py` files
  - Verify imports work

- [ ] **T-002**: Set up base classes/interfaces
  - Create abstract base classes
  - Define type hints and protocols
  - Add docstrings

### 1.2 Dependencies
- [ ] **T-003**: Configure dependencies
  - Add required packages to requirements.txt or pyproject.toml
  - Verify compatibility

---

## Phase 2: Core Features

> Main functionality implementation

### 2.1 {Feature Group 1}
- [ ] **T-004**: Implement {Component A}
  - Description of what needs to be done
  - Key acceptance criteria
  - Files to create/modify: `src/{path}/{file}.py`

- [ ] **T-005**: Implement {Component B}
  - Description
  - Acceptance criteria
  - Dependencies: T-004

### 2.2 {Feature Group 2}
- [ ] **T-006**: Implement {Feature}
  - Description
  - Acceptance criteria
  - Dependencies: T-004, T-005

---

## Phase 3: Integration

> Connecting components together

- [ ] **T-007**: Wire up {Component A} with {Component B}
  - Integration points
  - Error handling between components

- [ ] **T-008**: Create main entry point
  - CLI or API setup
  - Configuration handling

---

## Phase 4: Testing

> Quality assurance tasks

### 4.1 Unit Tests
- [ ] **T-009**: Write unit tests for {Component A}
  - Test file: `tests/test_{component_a}.py`
  - Coverage targets

- [ ] **T-010**: Write unit tests for {Component B}
  - Test file: `tests/test_{component_b}.py`

### 4.2 Integration Tests
- [ ] **T-011**: Write integration tests
  - Test end-to-end workflows
  - Test error scenarios

---

## Phase 5: Polish

> Final refinements

- [ ] **T-012**: Add error handling improvements
  - Better error messages
  - Edge case handling

- [ ] **T-013**: Performance optimization
  - Profile critical paths
  - Optimize bottlenecks

- [ ] **T-014**: Documentation
  - Update docstrings
  - Add usage examples

---

## Task Dependencies

```
T-001 --> T-002 --> T-003
              |
              v
         T-004 --> T-005 --> T-006
                        |
                        v
                   T-007 --> T-008
                        |
                        v
                   T-009, T-010 --> T-011
                                      |
                                      v
                               T-012, T-013, T-014
```

## Notes

- Tasks should be completed in order unless dependencies allow parallel work
- Each task should be small enough to complete in one focused session
- Mark tasks complete with [x] as you finish them
- Use `/spec:update-task "task description"` to update status

---

> Next: Run `/spec:approve tasks` when this breakdown is complete and reviewed.
```

3. **Update README.md** to reflect tasks status:
   - Change Tasks row to "In Progress" or "Ready for Review"

4. **Engage in dialogue**:
   - Present the task breakdown
   - Ask if granularity is appropriate
   - Identify any missing tasks
   - Iterate until the user is satisfied

## Output Format

After generating tasks:
```
Task breakdown generated for: {spec-name}
Location: spec/{spec-id}-{spec-name}/tasks.md

Summary:
- Phase 1 (Foundation): X tasks
- Phase 2 (Core Features): X tasks
- Phase 3 (Integration): X tasks
- Phase 4 (Testing): X tasks
- Phase 5 (Polish): X tasks
- Total: X tasks

Please review and:
1. Verify task granularity is appropriate
2. Check dependencies are correct
3. Add any missing tasks
4. Run /spec:approve tasks when satisfied
```

## Best Practices

- Each task should be completable in 1-4 hours
- Task IDs (T-001) make referencing easier
- Dependencies help with parallel work planning
- Group related tasks in phases
- Include testing tasks, not just implementation
- Be specific about files to create/modify
