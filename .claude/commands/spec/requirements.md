# /spec:requirements - Generate Requirements Document

You are generating comprehensive requirements for the current specification. This is Phase 1 of Spec-Driven Development.

## Prerequisites

1. Read `spec/.current-spec` to get the active specification
2. If no current spec, ask user to run `/spec:new` first or `/spec:switch` to select one
3. Check that `.requirements-approved` does NOT exist (requirements not yet approved)

## Instructions

1. **Read the current specification context**:
   - Read `spec/{current-spec}/README.md` for overview
   - Read `spec/{current-spec}/requirements.md` for any existing content
   - Read `CLAUDE.md` for project context

2. **Gather requirements through conversation**:
   - Ask clarifying questions about the feature/component
   - Understand the user's goals and constraints
   - Identify stakeholders and use cases

3. **Generate comprehensive requirements.md** with this structure:

```markdown
# Requirements: {spec-name}

> Phase 1 of Spec-Driven Development
> Status: Draft
> Last Updated: {date}

## 1. Overview

### 1.1 Purpose
[What problem does this solve? Why is it needed?]

### 1.2 Scope
[What is included and excluded from this specification?]

### 1.3 Success Criteria
[How do we know when this is complete and successful?]

## 2. User Stories

### US-001: [Story Title]
**As a** [user type]
**I want** [goal]
**So that** [benefit]

**Acceptance Criteria:**
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

### US-002: [Story Title]
[Continue pattern...]

## 3. Functional Requirements

### FR-001: [Requirement Title]
**Description:** [Detailed description]
**Priority:** [Must Have | Should Have | Could Have | Won't Have]
**Dependencies:** [List any dependencies]

### FR-002: [Continue pattern...]

## 4. Non-Functional Requirements

### NFR-001: Performance
[Performance requirements and benchmarks]

### NFR-002: Security
[Security requirements]

### NFR-003: Usability
[Usability requirements]

### NFR-004: Maintainability
[Code quality, documentation requirements]

## 5. Technical Constraints

- [Constraint 1: e.g., Must use Python 3.10+]
- [Constraint 2: e.g., Must integrate with existing LLM interface]
- [Constraint 3: e.g., Must support both OpenAI and Anthropic]

## 6. Assumptions

- [Assumption 1]
- [Assumption 2]

## 7. Out of Scope

- [Item 1: What we explicitly will NOT do]
- [Item 2]

## 8. Open Questions

- [ ] [Question 1 that needs resolution]
- [ ] [Question 2]

---

> Next: Run `/spec:approve requirements` when this document is complete and reviewed.
```

4. **Update README.md** to reflect requirements status:
   - Change Requirements row to "In Progress" or "Ready for Review"

5. **Engage in dialogue**:
   - Present the generated requirements
   - Ask if anything is missing or needs clarification
   - Iterate until the user is satisfied

## Output Format

After generating requirements:
```
Requirements generated for: {spec-name}
Location: spec/{spec-id}-{spec-name}/requirements.md

Please review the requirements document and:
1. Edit any sections that need refinement
2. Answer any open questions
3. Run /spec:approve requirements when satisfied

Current open questions:
- [List any open questions from section 8]
```

## Best Practices

- Be specific and measurable in acceptance criteria
- Use consistent terminology throughout
- Link requirements to user value (the "so that" clause)
- Prioritize ruthlessly - not everything is "Must Have"
- Keep scope focused - add items to "Out of Scope" liberally
