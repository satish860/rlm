# Spec 002: Query Types (Extract & Summary)

## Status: Tasks Phase

| Phase | Status |
|-------|--------|
| Requirements | Approved (2026-01-15) |
| Design | Approved (2026-01-15) |
| Tasks | Ready to Generate |
| Implementation | Blocked |

## Problem

The current `query()` method is designed for Q&A style queries. But users need:
1. **Extract** - Structured data extraction (JSON output)
2. **Summary** - Document/section summarization (markdown output)

These have different requirements than general Q&A.

---

## Key Design Decision: Three-Stage Pipeline

**Insight**: The existing `query()` method does excellent free-text extraction. Rather than replacing it, we add pre/post processing layers.

```
Stage 0: Schema Generation  -> Pydantic model (for plain English users)
Stage 1: query()            -> free text answer (KEEP AS-IS, works great)
Stage 2: Instructor/JSON    -> validated output (NEW)
```

**Why this approach:**
- Don't break what works - query() navigation is solid
- Separation of concerns - schema / exploration / structuring
- Instructor handles validation, retries, type coercion
- Can use cheaper/faster model for Stages 0 & 2 (just reformatting)
- Two audiences: developers (skip Stage 0) and plain English users (run Stage 0)

**Complete Flow:**
```
                        PLAIN ENGLISH USER                    DEVELOPER
                              |                                   |
                    "papers with title,                    PaperList model
                     authors, year, url"                          |
                              |                                   |
                              v                                   |
                    +-------------------+                         |
                    | Stage 0: Schema   |                         |
                    | Generation (LLM)  |  <-- skipped -----------+
                    +-------------------+
                              |
                    Generated PaperList
                              |
                              v
                    +-------------------+
                    | Stage 1: query()  |  <-- existing, unchanged
                    | Document nav +    |
                    | free text answer  |
                    +-------------------+
                              |
                    "1. Longbench v2 by Bai et al., 2025...
                     2. OOLONG by Bertsch et al., 2025..."
                              |
                              v
                    +-------------------+
                    | Stage 2: Instructor|
                    | JSON conversion   |
                    +-------------------+
                              |
                              v
                    [{"title": "Longbench v2", ...}, ...]
```

---

## Query Type Comparison

| Aspect | Q&A (current) | Extract | Summary |
|--------|---------------|---------|---------|
| **Goal** | Answer question | Extract structured data | Condense content |
| **Output** | Free text | JSON schema | Markdown |
| **Coverage** | Targeted sections | Exhaustive (all relevant) | Configurable scope |
| **Strategy** | Navigate to answer | Scan all, aggregate | Read then condense |
| **Validation** | Answer quality | Schema compliance | Length/coverage |

---

## Extract Query Type

### Use Cases
- Extract all references/citations as JSON
- Extract all figures/tables with captions
- Extract all equations/formulas
- Extract key terms and definitions
- Extract methodology steps
- Extract experimental results as structured data

### Design (Two-Stage)

```python
def extract(
    self,
    what: str,                           # What to extract: "papers", "figures", "methods"
    response_model: type[BaseModel],     # Pydantic model for output
    sections: list[str] = None,          # Limit to specific sections
    verbose: bool = False,
) -> list[dict]:
    """
    Extract structured data from document.

    Stage 1: Use query() to get free-text extraction
    Stage 2: Use Instructor to convert to Pydantic model

    Returns list of extracted items as dictionaries.
    """
```

### Two Audiences

| Audience | Input | Example |
|----------|-------|---------|
| **Developer** | Pydantic model | `extract("papers", response_model=PaperList)` |
| **Plain English** | Natural description | `extract("all papers with title, authors, year, url")` |

**For Plain English users, we auto-generate the schema:**

```
User: "Extract all papers with title, authors, year, and url"
                    |
                    v
        Stage 0: Schema Generation (NEW)
        Use LLM to generate Pydantic model from description
                    |
                    v
        Generated: class Paper(BaseModel):
                       title: str
                       authors: list[str]
                       year: str
                       url: str | None
                    |
                    v
        Stage 1 & 2 proceed as normal
```

### Stage 0: Schema Generation (for Plain English)

```python
from pydantic import create_model

def generate_schema_from_description(what: str) -> type[BaseModel]:
    """
    Use LLM to infer schema from plain English description.

    Example:
        what = "papers with title, authors, year, url"
        -> generates Paper model with those fields
    """
    # Use Instructor to generate the schema definition
    class SchemaDefinition(BaseModel):
        item_name: str  # e.g., "Paper"
        fields: list[FieldDef]

    class FieldDef(BaseModel):
        name: str       # e.g., "title"
        type: str       # e.g., "str", "list[str]", "int"
        optional: bool  # e.g., False

    # Ask LLM to define the schema
    schema_def = instructor_client(
        model=SUB_MODEL,
        response_model=SchemaDefinition,
        messages=[{
            "role": "user",
            "content": f"Define a schema for extracting: {what}"
        }]
    )

    # Dynamically create Pydantic model
    field_types = {
        "str": str,
        "int": int,
        "list[str]": list[str],
        # ... etc
    }

    fields = {}
    for f in schema_def.fields:
        ftype = field_types.get(f.type, str)
        if f.optional:
            fields[f.name] = (ftype | None, None)
        else:
            fields[f.name] = (ftype, ...)

    ItemModel = create_model(schema_def.item_name, **fields)
    ListModel = create_model(f"{schema_def.item_name}List", items=(list[ItemModel], ...))

    return ListModel
```

### API Design for Both Audiences

```python
def extract(
    self,
    what: str,                                    # Always required
    response_model: type[BaseModel] | None = None,  # Developer provides
    verbose: bool = False,
) -> list[dict]:
    """
    Extract structured data.

    Developer mode (response_model provided):
        - Skip Stage 0
        - Use provided Pydantic model

    Plain English mode (response_model=None):
        - Run Stage 0 to generate schema
        - Show user the generated schema
        - Proceed with generated model
    """
    if response_model is None:
        # Plain English mode - generate schema
        print(f"Generating schema for: {what}")
        response_model = generate_schema_from_description(what)
        print(f"Generated schema: {response_model.model_json_schema()}")

    # Stage 1: Query
    # Stage 2: Instructor conversion
    ...
```

### Stage 1: Query Prompt

```
List ALL {what} in this document.

For each item include: {fields from Pydantic model}

Be EXHAUSTIVE - list every single item, do not summarize or skip any.
Format as a numbered list with all details for each item.
```

### Stage 2: Instructor Conversion

```python
import instructor
from pydantic import BaseModel

class Paper(BaseModel):
    title: str
    authors: list[str]
    year: str
    url: str | None = None

class PaperList(BaseModel):
    papers: list[Paper]

# Convert free text to structured JSON
client = instructor.from_litellm(litellm.completion)
result = client(
    model=SUB_MODEL,  # Can use cheaper model - just reformatting
    response_model=PaperList,
    messages=[{"role": "user", "content": free_text_answer}],
    max_retries=3,  # Auto-retry on validation failure
)
```

### Why Instructor?

| Feature | Manual JSON | Instructor |
|---------|-------------|------------|
| Validation | Hope for the best | Pydantic enforced |
| Retries | Manual | Automatic |
| Type coercion | None | Built-in |
| Partial data | Fails | Handles gracefully |
| Schema docs | Separate | In Pydantic model |

---

## Summary Query Type

### Use Cases
- Summarize entire document (executive summary)
- Summarize specific section(s)
- Generate abstract if missing
- Create bullet-point overview
- Extract key findings/conclusions

### Design (Two-Stage for Structured Summary)

```python
def summarize(
    self,
    scope: str = "document",             # "document", "section:NAME", "sections:A,B,C"
    style: str = "paragraph",            # "paragraph", "bullets", "executive", "abstract"
    max_length: int = 500,               # Target length in words
    structured: bool = False,            # If True, return structured dict via Instructor
    verbose: bool = False,
) -> str | dict:
    """
    Summarize document or sections.

    If structured=False: Returns markdown summary (Stage 1 only)
    If structured=True:  Returns dict via Instructor (Stage 1 + Stage 2)
    """
```

### Stage 1: Query for Summary (always runs)

```
Summarize {scope} in {style} style.

Target length: ~{max_length} words.

STYLE: {style_guidance}

Include:
- Main contribution/thesis
- Key methods/approach
- Important results/findings
- Conclusions
```

### Stage 2: Structured Summary (optional, when structured=True)

```python
class StructuredSummary(BaseModel):
    title: str
    main_contribution: str
    problem: str
    approach: str
    key_results: list[str]
    limitations: list[str] = []
    future_work: list[str] = []

# Convert free text summary to structured
result = client(
    model=SUB_MODEL,
    response_model=StructuredSummary,
    messages=[{"role": "user", "content": free_text_summary}],
)
```

### When to Use Structured

| Use Case | structured= | Output |
|----------|-------------|--------|
| Human reading | False | Markdown prose |
| Feed to another LLM | True | JSON dict |
| Store in database | True | JSON dict |
| Display in UI | False | Markdown prose |

---

## Implementation Plan

### Phase 1: Add Instructor Dependency
- [ ] Add `instructor` to requirements.txt
- [ ] Verify instructor works with litellm
- [ ] Create `src/single_doc/models.py` with Pydantic models

### Phase 2: Stage 0 - Schema Generation
- [ ] Create `generate_schema_from_description()` function
- [ ] Define `SchemaDefinition` and `FieldDef` models for LLM output
- [ ] Implement dynamic Pydantic model creation with `create_model()`
- [ ] Test schema generation from plain English descriptions

### Phase 3: Extract Method
- [ ] Create `extract()` method in SingleDocRLM
- [ ] Route: response_model provided -> skip Stage 0
- [ ] Route: response_model=None -> run Stage 0 first
- [ ] Stage 1: Build query prompt from Pydantic model fields
- [ ] Stage 2: Pass query result to Instructor
- [ ] Test on RLM paper (extract references)

### Phase 4: Summarize Method
- [ ] Create `summarize()` method in SingleDocRLM
- [ ] Implement scope parsing (document/section/sections)
- [ ] Implement style prompts (paragraph/bullets/executive/abstract)
- [ ] Add optional structured=True for Instructor conversion
- [ ] Test on RLM paper (document summary)

### Phase 5: Common Models & Convenience Methods
- [ ] Paper, PaperList models
- [ ] Figure, FigureList models
- [ ] MethodStep, Methodology models
- [ ] StructuredSummary model
- [ ] Convenience methods: `extract_papers()`, `extract_figures()`

**Key Insight**: No changes to REPL or query() needed. Just add wrapper methods.

---

## API Examples

### Extract - Plain English User (No Code Knowledge)

```python
rlm = SingleDocRLM("paper.pdf")
rlm.load_index("paper.index.json")

# Just describe what you want - schema auto-generated
papers = rlm.extract("all referenced papers with title, authors, year, and arxiv url")

# Output:
# > Generating schema for: all referenced papers with title, authors, year, and arxiv url
# > Generated schema:
# >   Paper:
# >     - title: str
# >     - authors: list[str]
# >     - year: str
# >     - arxiv_url: str (optional)
# > Extracting...
# > Found 42 papers

# Another example - figures
figures = rlm.extract("all figures with their number, caption, and which section they appear in")

# Another example - custom
methods = rlm.extract("methodology steps with step number, name, and description")
```

### Extract - Developer (Pydantic Models)

```python
from src.single_doc.models import Paper, PaperList, Figure, FigureList
from pydantic import BaseModel

rlm = SingleDocRLM("paper.pdf")
rlm.load_index("paper.index.json")

# Provide your own Pydantic model - full control
papers = rlm.extract(
    what="referenced papers",
    response_model=PaperList,  # Skips schema generation
)

# Convenience methods for common extractions
papers = rlm.extract_papers()    # Uses built-in PaperList model
figures = rlm.extract_figures()  # Uses built-in FigureList model

# Custom model for specific needs
class Benchmark(BaseModel):
    name: str
    description: str
    metrics: list[str]
    results: dict[str, float] | None = None

class BenchmarkList(BaseModel):
    benchmarks: list[Benchmark]

benchmarks = rlm.extract(
    what="evaluation benchmarks",
    response_model=BenchmarkList,
)
```

### Summary

```python
# Full document summary (markdown string)
summary = rlm.summarize(
    scope="document",
    style="executive",
    max_length=300
)

# Section summary (bullets)
methods_summary = rlm.summarize(
    scope="section:2.2 METHODS AND BASELINES",
    style="bullets",
    max_length=150
)

# Multi-section summary
results_summary = rlm.summarize(
    scope="sections:3 RESULTS AND DISCUSSION,3.1 EMERGENT PATTERNS",
    style="paragraph",
    max_length=400
)

# Structured summary (returns dict for programmatic use)
summary_data = rlm.summarize(
    scope="document",
    style="executive",
    structured=True,  # Returns dict via Instructor
)
# summary_data = {
#     "title": "Recursive Language Models",
#     "main_contribution": "...",
#     "key_results": ["...", "..."],
#     ...
# }
```

---

## Open Questions

1. **Large extractions**: For 50+ items, should Stage 1 query be split into multiple calls, or let the LLM handle it in one pass with more rounds?

2. **Instructor model**: Should Stage 2 always use SUB_MODEL (cheaper), or allow user to specify?

3. **Retry strategy**: If Instructor validation fails after max_retries, should we return partial results or raise error?

4. **Caching**: Should extracted/summarized results be cached in the index file?

5. **Streaming**: Should Stage 1 stream results as they're found, or wait for complete answer?

---

## Success Criteria

- [ ] Extract 40+ references from RLM paper with >95% accuracy
- [ ] Extract all 4 figures with correct captions
- [ ] Generate document summary within 10% of target length
- [ ] Summary captures all 5 key observations from results section
