# /spec:design - Generate Design Document

You are generating the technical design for the current specification. This is Phase 2 of Spec-Driven Development.

## Prerequisites

1. Read `spec/.current-spec` to get the active specification
2. Check that `.requirements-approved` EXISTS (requirements must be approved first)
3. Check that `.design-approved` does NOT exist (design not yet approved)
4. If requirements not approved, inform user to run `/spec:approve requirements` first

## Instructions

1. **Read the specification context**:
   - Read `spec/{current-spec}/requirements.md` thoroughly
   - Read `spec/{current-spec}/README.md`
   - Read `CLAUDE.md` for project patterns and conventions
   - Review existing codebase structure if relevant

2. **Generate comprehensive design.md** with this structure:

```markdown
# Design: {spec-name}

> Phase 2 of Spec-Driven Development
> Status: Draft
> Last Updated: {date}
> Prerequisites: [Requirements Approved]

## 1. Overview

### 1.1 Design Goals
[What are the key design objectives?]

### 1.2 Design Principles
[What principles guide this design? e.g., simplicity, extensibility, performance]

## 2. Architecture

### 2.1 High-Level Architecture
[Describe the overall architecture]

```
[ASCII diagram or description of component relationships]
```

### 2.2 Component Breakdown

#### Component: {ComponentName}
- **Purpose:** [What does this component do?]
- **Location:** `src/{path}/`
- **Dependencies:** [What does it depend on?]
- **Dependents:** [What depends on it?]

[Repeat for each component]

## 3. Data Flow

### 3.1 Primary Data Flow
[Describe how data flows through the system]

```
[Input] --> [Component A] --> [Component B] --> [Output]
```

### 3.2 State Management
[How is state managed? What patterns are used?]

## 4. API Design

### 4.1 Public API

```python
class {ClassName}:
    """
    [Class description]
    """

    def __init__(self, ...):
        """[Constructor description]"""
        pass

    def method_name(self, param: Type) -> ReturnType:
        """
        [Method description]

        Args:
            param: [Description]

        Returns:
            [Description]
        """
        pass
```

### 4.2 Internal APIs
[Document key internal interfaces]

## 5. Technology Decisions

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| [Component] | [Tech] | [Why this choice?] |

### 5.2 Design Patterns

| Pattern | Usage | Rationale |
|---------|-------|-----------|
| [Pattern Name] | [Where used] | [Why this pattern?] |

## 6. File Structure

```
src/
  {module}/
    __init__.py
    {file1}.py      # [Purpose]
    {file2}.py      # [Purpose]
```

## 7. Error Handling

### 7.1 Error Types
[What errors can occur? How are they categorized?]

### 7.2 Error Recovery
[How does the system recover from errors?]

## 8. Performance Considerations

### 8.1 Bottlenecks
[Potential performance bottlenecks]

### 8.2 Optimization Strategies
[How will performance be ensured?]

## 9. Security Considerations

[Security aspects of the design]

## 10. Testing Strategy

### 10.1 Unit Tests
[What will be unit tested?]

### 10.2 Integration Tests
[What integration tests are needed?]

## 11. Migration/Compatibility

[How does this integrate with existing code?]

## 12. Open Design Questions

- [ ] [Design question 1]
- [ ] [Design question 2]

---

> Next: Run `/spec:approve design` when this document is complete and reviewed.
```

3. **Update README.md** to reflect design status:
   - Change Design row to "In Progress" or "Ready for Review"

4. **Engage in dialogue**:
   - Present the design
   - Discuss tradeoffs and alternatives
   - Address any concerns
   - Iterate until the user is satisfied

## Output Format

After generating design:
```
Design generated for: {spec-name}
Location: spec/{spec-id}-{spec-name}/design.md

Please review the design document and:
1. Validate architectural decisions
2. Check for missing components
3. Review API signatures
4. Run /spec:approve design when satisfied

Key decisions requiring validation:
- [List key design decisions]
```

## Best Practices

- Align design with project conventions in CLAUDE.md
- Keep components loosely coupled
- Document the "why" behind decisions, not just the "what"
- Consider error cases and edge conditions
- Think about testability from the start
- Use consistent naming conventions
