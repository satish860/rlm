# RLM Project - Claude Instructions

## Project Overview

Building an enhanced Recursive Language Model (RLM) implementation based on the paper.

**Paper**: "Recursive Language Models" (Zhang, Kraska, Khattab - Dec 2025)
**Paper PDF**: `recursive_language_models.pdf`

---

## Core Concept

```
Original RLM:  Long Doc -> REPL (flat string) -> grep/slice -> sub-LLM
```

**Key Insight**: Treat document as external environment, not part of the prompt.

---

## Reference

- **Paper**: `recursive_language_models.pdf` in this folder
- **Converted**: `recursive_language_models.converted.md`

---

## Development Guidelines

- Python 3.10+
- Type hints required
- No emojis in code or comments
- Uses `litellm` for LLM calls

---

## Environment Variables

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```
