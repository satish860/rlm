# /spec:approve - Approve a Phase

You are approving a phase of the current specification. This creates a gate marker that allows progression to the next phase.

## Arguments
- `$ARGUMENTS` - The phase to approve: `requirements`, `design`, or `tasks`

## Prerequisites

1. Read `spec/.current-spec` to get the active specification
2. Validate the phase argument is one of: `requirements`, `design`, `tasks`

## Phase Validation Rules

### Approving Requirements
- **Check**: `spec/{current-spec}/requirements.md` exists and has content beyond template
- **Check**: All "Open Questions" in section 8 should be resolved (checked off)
- **Creates**: `spec/{current-spec}/.requirements-approved`

### Approving Design
- **Check**: `.requirements-approved` must exist first
- **Check**: `spec/{current-spec}/design.md` exists and has content beyond template
- **Check**: All "Open Design Questions" should be resolved
- **Creates**: `spec/{current-spec}/.design-approved`

### Approving Tasks
- **Check**: `.design-approved` must exist first
- **Check**: `spec/{current-spec}/tasks.md` exists and has content beyond template
- **Creates**: `spec/{current-spec}/.tasks-approved`

## Instructions

1. **Validate prerequisites**:
   - Check current spec exists
   - Check phase argument is valid
   - Check required prior phases are approved

2. **Validate content**:
   - Read the relevant document
   - Check it has substantive content (not just template)
   - Warn if there are unresolved open questions

3. **Create approval marker**:
   - Create the `.{phase}-approved` file with timestamp
   - Content: `Approved: {date} {time}`

4. **Update README.md**:
   - Update the Phase Status table
   - Mark the approved phase as "Approved" with date
   - Mark the next phase as "Ready" (if applicable)

5. **Report success and next steps**

## Output Format

### On Success:
```
Phase Approved: {phase}
Specification: {spec-name}

Approval marker created: spec/{spec-id}-{spec-name}/.{phase}-approved

Phase Status:
- Requirements: [Approved / Pending]
- Design: [Approved / Pending / Blocked]
- Tasks: [Approved / Pending / Blocked]
- Implementation: [Ready / Blocked]

Next Step: {guidance on what to do next}
```

### On Failure (prerequisites not met):
```
Cannot approve {phase}.

Reason: {specific reason}

Required action: {what needs to be done first}
```

### On Failure (content issues):
```
Warning: {phase} may not be ready for approval.

Issues found:
- {issue 1}
- {issue 2}

Proceed anyway? (y/n)
```

## Next Steps Guidance

After approving requirements:
```
Next Step: Run /spec:design to generate the technical design
```

After approving design:
```
Next Step: Run /spec:tasks to generate the implementation task breakdown
```

After approving tasks:
```
Next Step: Run /spec:implement to begin implementation
All phases approved - ready for implementation!
```

## Example Usage

```
/spec:approve requirements
/spec:approve design
/spec:approve tasks
```

## Rollback

To revoke an approval (for re-work):
- Manually delete the `.{phase}-approved` file
- This will block subsequent phases until re-approved
