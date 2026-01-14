# Specification: Single-doc-flow

**ID**: 001
**Created**: 2026-01-14
**Status**: Design Phase (Ready for Review)

## Phase Status

| Phase | Status | Approved |
|-------|--------|----------|
| Requirements | Complete | 2026-01-14 |
| Design | Complete | 2026-01-14 |
| Tasks | Complete | 2026-01-14 |
| Implementation | In Progress | - |

## Overview

Single Document RLM flow - enables querying long documents (1M-100M tokens) that exceed LLM context windows by building a structured index with TOC, section boundaries, and summaries, then exposing navigation/search functions through a REPL environment where the root LLM writes and executes code to answer queries.

## Quick Links

- [Requirements](./requirements.md)
- [Design](./design.md)
- [Tasks](./tasks.md)
