# Specification: rlm-library

**ID**: 004
**Created**: 2026-01-19
**Status**: Implementation Phase (In Progress)

## Phase Status

| Phase | Status | Approved |
|-------|--------|----------|
| Requirements | Approved | 2026-01-19 |
| Design | Approved | 2026-01-19 |
| Tasks | Approved | 2026-01-19 |
| Implementation | In Progress | - |

## Overview

Package RLM as a production-ready Python library that beats langextract by leveraging the superior "root model + sub-LLM" architecture with explicit reasoning.

## Key Goals

1. Simple API: `rlm.extract(doc, schema)` works out of the box
2. Beat langextract on accuracy, reliability, and transparency
3. Production-ready with retry logic, caching, logging
4. Easy to extend with new providers and schemas

## Quick Links

- [Requirements](./requirements.md)
- [Design](./design.md)
- [Tasks](./tasks.md)
