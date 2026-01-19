# RLM - Recursive Language Model

A Python library for intelligent document extraction using the "root model + sub-LLM" architecture with explicit reasoning.

Based on the paper: *"Recursive Language Models"* (Zhang, Kraska, Khattab - Dec 2025)

## Key Features

- **Structured Extraction**: Extract typed data from PDFs, Markdown, and text files
- **Document Q&A**: Ask questions and get cited answers
- **Reasoning Traces**: Full transparency into the extraction process
- **Citation Tracking**: Every extracted fact linked to source text and page
- **Session Persistence**: Save and resume long extractions
- **HTML Visualization**: Generate interactive reports

## Installation

```bash
pip install -r requirements.txt
```

Required environment variable:
```bash
# OpenRouter (default provider)
export OPENROUTER_API_KEY=sk-or-...

# Or use OpenAI/Anthropic directly
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Quick Start

### Extract Structured Data

```python
import rlm
from pydantic import BaseModel

# Define what to extract
class Contact(BaseModel):
    name: str
    email: str = None
    phone: str = None
    page: int  # Required for citation tracking

# Extract from document
result = rlm.extract("contacts.pdf", schema=Contact)

# Use the data
for contact in result.data:
    print(contact["name"], contact.get("email"))

# Generate HTML report
rlm.visualize(result, output="report.html")
```

### Ask Questions

```python
import rlm

result = rlm.query("report.pdf", "What was Q3 revenue?")

print(result.answer)
print(f"Confidence: {result.confidence:.0%}")

for cite in result.citations:
    print(f"  Page {cite.page}: {cite.snippet[:50]}...")
```

### Use Built-in Schemas

```python
import rlm

# Available schemas
result = rlm.extract("invoice.pdf", schema=rlm.schemas.Invoice)
result = rlm.extract("directory.pdf", schema=rlm.schemas.Contact)
result = rlm.extract("document.pdf", schema=rlm.schemas.Entity)
result = rlm.extract("form.pdf", schema=rlm.schemas.KeyValuePair)
result = rlm.extract("report.pdf", schema=rlm.schemas.FinancialFigure)
```

## API Reference

### rlm.extract()

```python
result = rlm.extract(
    document="report.pdf",        # PDF, Markdown, or text file
    schema=MySchema,              # Pydantic model (optional)
    verbose=True,                 # Print progress
    max_iterations=40,            # Max reasoning iterations
    root_model="anthropic/...",   # Override root model
    sub_model="openai/...",       # Override sub model
    provider="openrouter"         # Provider: openrouter, openai, anthropic
)
```

**Returns `ExtractionResult`:**
- `result.data` - List of extracted records (dicts)
- `result.citations` - Evidence snippets with page numbers
- `result.thinking_log` - Reasoning trace
- `result.confidence_history` - Confidence progression
- `result.iterations` - Iterations used

### rlm.query()

```python
result = rlm.query(
    document="report.pdf",
    question="What was the total revenue?",
    verbose=True,
    max_iterations=20
)
```

**Returns `QueryResult`:**
- `result.answer` - The answer text
- `result.citations` - Supporting evidence
- `result.confidence` - Confidence score (0.0 to 1.0)

### rlm.visualize()

```python
# Save to file
rlm.visualize(result, output="report.html")

# Get HTML string
html = rlm.visualize(result)

# Open in browser
rlm.visualize(result, output="report.html", open_browser=True)
```

## Advanced Usage

### Custom Engine Configuration

```python
from rlm import RLMEngine

engine = RLMEngine(
    root_model="anthropic/claude-sonnet-4-20250514",
    sub_model="openai/gpt-4o-mini",
    provider="openrouter"
)

result = engine.extract(
    "document.pdf",
    schema=MySchema,
    max_iterations=50,
    verbose=True
)
```

### Access Reasoning Traces

```python
result = rlm.extract("doc.pdf", schema=Contact)

# View thinking process
for thought in result.thinking_log:
    print(f"[{thought.timestamp}] {thought.reasoning}")

# View confidence progression
for entry in result.confidence_history:
    print(f"Confidence: {entry.confidence:.0%}, Records: {entry.records_extracted}")

# View citations
for cite in result.citations:
    print(f"Page {cite.page}: \"{cite.snippet}\"")
    if cite.note:
        print(f"  Note: {cite.note}")
```

### Session Management

```python
from rlm.reasoning.session import SessionManager

manager = SessionManager()

# List saved sessions
print(manager.format_sessions_table())

# Load a session
session = manager.load_session("my_extraction")
print(f"Records: {len(session['records'])}")
```

## CLI Usage

```bash
# Extract data
rlm extract document.pdf --schema contact --output results.json

# Ask questions
rlm query document.pdf "What is the total amount?"

# Generate visualization
rlm visualize results.json --output report.html

# List sessions
rlm sessions list
```

## Architecture

RLM uses a two-model architecture:

1. **Root Model** (e.g., Claude Sonnet): Orchestrates extraction, reasons about document structure, writes extraction code
2. **Sub Model** (e.g., GPT-4o-mini): Executes parallel extraction calls, fast and cheap

The root model operates in a REPL environment with:
- `pages[]` - Document split into pages
- `get_section(start, end)` - Get page content
- `llm_extract(prompt, schema)` - Structured extraction
- `llm_extract_parallel(sections, prompt, schema)` - Parallel extraction
- `think(reasoning)` - Log reasoning steps
- `cite(snippet, page, note)` - Record evidence
- `evaluate_progress()` - Self-assess confidence

## Examples

```bash
# Quick test with sample data
python examples/quick_test.py

# Extract from Agribusiness PDF
python examples/run_extraction.py

# Q&A on RLM paper
python examples/run_query.py "What is the REPL environment?"

# View saved sessions
python examples/view_session.py
```

## Project Structure

```
rlm/
├── __init__.py           # Public API
├── cli.py                # CLI interface
├── config.py             # Configuration
├── exceptions.py         # Custom exceptions
├── core/
│   ├── engine.py         # Main RLMEngine
│   ├── repl.py           # REPL environment
│   ├── tools.py          # Tool definitions
│   └── prompts.py        # System prompts
├── document/
│   ├── reader.py         # Document reading
│   ├── segmenter.py      # Page segmentation
│   └── formats/          # PDF, Markdown, Text
├── extraction/
│   ├── schemas.py        # Built-in schemas
│   └── structured.py     # Structured extraction
├── providers/
│   ├── base.py           # Provider ABC
│   ├── openrouter.py     # OpenRouter
│   ├── openai.py         # OpenAI
│   └── anthropic.py      # Anthropic
├── reasoning/
│   ├── tracer.py         # Reasoning tracer
│   └── session.py        # Session management
└── visualization/
    ├── html.py           # HTML reports
    └── citations.py      # Citation highlighting
```

## License

MIT
