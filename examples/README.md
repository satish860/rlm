# RLM Library Examples

Practical examples using real test data.

## Quick Start

```bash
# From project root directory
cd C:\code\rlm

# Quick test with sample text file
python examples/quick_test.py

# Extract contacts from Agribusiness PDF
python examples/run_extraction.py

# Ask questions about the RLM paper
python examples/run_query.py
python examples/run_query.py "What is the REPL environment?"
```

## Test Data

| File | Description |
|------|-------------|
| `examples/sample_data.txt` | Simple text file with contacts + financials |
| `40255083-Agribusiness-Companies.pdf` | Real company directory (contact extraction) |
| `recursive_language_models.pdf` | RLM paper (Q&A testing) |

## Examples

| Script | Purpose |
|--------|---------|
| `quick_test.py` | Fast test with sample_data.txt |
| `run_extraction.py` | Extract contacts from Agribusiness PDF |
| `run_query.py` | Q&A on RLM paper |

## Library Usage

### 1. Extract Structured Data

```python
import rlm
from pydantic import BaseModel

# Use built-in schema
result = rlm.extract("document.pdf", schema=rlm.schemas.Contact)

# Or define custom schema
class Invoice(BaseModel):
    vendor: str
    amount: float
    date: str
    page: int  # Required for citation tracking

result = rlm.extract("invoice.pdf", schema=Invoice)

# Access results
for record in result.data:
    print(record["vendor"], record["amount"])
```

### 2. Ask Questions

```python
import rlm

result = rlm.query("report.pdf", "What was Q3 revenue?")
print(result.answer)
print(f"Confidence: {result.confidence:.0%}")

for cite in result.citations:
    print(f"  Page {cite.page}: {cite.snippet[:50]}...")
```

### 3. Generate HTML Report

```python
import rlm

result = rlm.extract("doc.pdf", schema=MySchema)
rlm.visualize(result, output="report.html", open_browser=True)
```

### 4. Advanced: Custom Models

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

# Access reasoning trace
for thought in result.thinking_log:
    print(thought.reasoning)

# Access confidence progression
for entry in result.confidence_history:
    print(f"Confidence: {entry.confidence:.0%}")
```

## Built-in Schemas

```python
import rlm

rlm.schemas.Contact        # Name, email, phone, company
rlm.schemas.Invoice        # Vendor, amount, line items
rlm.schemas.Entity         # Generic named entities
rlm.schemas.TableRow       # Table data
rlm.schemas.KeyValuePair   # Form fields
rlm.schemas.FinancialFigure # Financial metrics
```

## Environment Setup

```bash
# Required for OpenRouter (default)
set OPENROUTER_API_KEY=sk-or-...

# Or use OpenAI directly
set OPENAI_API_KEY=sk-...

# Or use Anthropic directly
set ANTHROPIC_API_KEY=sk-ant-...
```
