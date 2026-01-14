# Design: Single-doc-flow

> Phase 2 of Spec-Driven Development
> Status: APPROVED
> Approved: 2026-01-14
> Prerequisites: [Requirements Approved]

## 1. Overview

### 1.1 Design Goals
1. **Simplicity** - Start with markdown TOC parsing, add complexity only when needed
2. **Testability** - Each component testable in isolation with mock LLM
3. **Incremental** - Build and test one layer at a time

### 1.2 Design Principles
- Prefer regex parsing over LLM calls where possible (faster, cheaper)
- Use lightweight LLM (gemini-flash, gpt-4o-mini) as fallback for TOC when regex fails
- Lazy loading - don't read sections until needed
- Fail gracefully - if both regex and LLM fail, treat entire doc as one section

## 2. Architecture

### 2.1 High-Level Architecture

```
+------------------+
|   SingleDocRLM   |  <-- Main API class
+------------------+
         |
    +----+----+
    |         |
    v         v
+-------+  +------+
| Index |  | REPL |
+-------+  +------+
    |         |
    v         v
+-------+  +------+
|Indexer|  |Execut|
+-------+  +------+
    |         |
    +----+----+
         v
    +--------+
    |  LLM   |
    +--------+
```

### 2.2 Component Breakdown

#### Component: LLM Interface
- **Purpose:** Unified interface for any LLM provider
- **Location:** Uses `litellm` library directly (no custom wrapper needed)
- **Dependencies:** litellm
- **Dependents:** Indexer, REPL

#### Component: Indexer
- **Purpose:** Build document index (TOC, sections, summaries)
- **Location:** `src/single_doc/indexer.py`
- **Dependencies:** LLM Interface
- **Dependents:** SingleDocRLM

#### Component: REPL
- **Purpose:** Execute code with document access functions
- **Location:** `src/single_doc/repl.py`
- **Dependencies:** Index data, LLM Interface
- **Dependents:** SingleDocRLM

#### Component: SingleDocRLM
- **Purpose:** Main orchestrator - builds index, runs queries
- **Location:** `src/single_doc/rlm.py`
- **Dependencies:** Indexer, REPL, LLM Interface
- **Dependents:** User code

## 3. Data Flow

### 3.1 Index Building Flow

```
doc_path --> markitdown.convert() --> markdown_text --> parse_toc() --> map_sections()
    |                                                        |                |
    v                                                        v                v
[PDF/DOCX/HTML/TXT]                                    [# headings]    [section_map]
                                                                              |
                                                                              v
                                                                    generate_summaries()
                                                                              |
                                                                              v
                                                                    SingleDocIndex
```

**Key insight**: markitdown normalizes all formats to markdown, so TOC parsing is always `#` headings.

### 3.2 Query Execution Flow

```
question --> build_system_prompt() --> root_llm.generate_code()
                                              |
                                              v
                                       execute_in_repl()
                                              |
                                       +------+------+
                                       |             |
                                       v             v
                                   [output]    FINAL(answer)?
                                       |             |
                                       v             v
                                  feed_back     return answer
                                  to LLM
```

### 3.3 State Management
- **Index**: Immutable after build, serializable to JSON
- **REPL**: Stateless functions, reads from doc file on demand
- **Query**: Each query is independent, no cross-query state

## 4. API Design

### 4.1 Public API

```python
class SingleDocRLM:
    """Main API for single document RLM."""

    def __init__(
        self,
        doc_path: str,
        root_model: str = "gpt-4o",
        sub_model: str = "gpt-4o-mini",
        toc_model: str = "gemini/gemini-2.0-flash"
    ):
        """
        Initialize RLM for a document.

        Args:
            doc_path: Path to document (PDF/DOCX/HTML/TXT/MD)
            root_model: Model for code generation (e.g., gpt-4o, claude-sonnet)
            sub_model: Model for summaries/semantic tasks (e.g., gpt-4o-mini, claude-haiku)
            toc_model: Lightweight model for TOC fallback (e.g., gemini-flash, gpt-4o-mini)
        """

    def build_index(self) -> None:
        """Build document index (TOC, sections, summaries)."""

    def save_index(self, path: str) -> None:
        """Save index to JSON file."""

    def load_index(self, path: str) -> None:
        """Load index from JSON file."""

    def query(self, question: str, max_rounds: int = 10) -> str:
        """
        Answer a question about the document.

        Args:
            question: Natural language question
            max_rounds: Maximum REPL execution rounds

        Returns:
            Answer string
        """

    def repl(self) -> None:
        """Start interactive REPL for document exploration."""
```

### 4.2 Index Data Structure

```python
@dataclass
class Section:
    """A document section with hierarchy info."""
    title: str              # "3.2 Model Architecture"
    level: int              # 2 (depth in hierarchy)
    parent: str | None      # "3. Methods" or None for top-level
    start_char: int         # Start offset in markdown
    end_char: int           # End offset in markdown

@dataclass
class SingleDocIndex:
    """Document index containing structure and summaries."""

    source_path: str                    # Original document path (PDF/DOCX/etc)
    markdown_path: str                  # Path to converted markdown file
    total_chars: int                    # Total markdown size
    sections: dict[str, Section]        # section_name -> Section (with parent info)
    summaries: dict[str, str]           # section_name -> summary (includes "[under: parent]")
    keywords: dict[str, list[str]]      # section_name -> [keywords]

    def get_toc(self) -> list[dict]:
        """Return hierarchical TOC."""
        return [
            {"title": s.title, "level": s.level, "parent": s.parent}
            for s in self.sections.values()
        ]

    def to_json(self) -> str:
        """Serialize to JSON string."""

    @classmethod
    def from_json(cls, json_str: str) -> "SingleDocIndex":
        """Deserialize from JSON string."""
```

**Note**: Each section stores its parent, so summaries can include `[under: parent]` context.

### 4.3 REPL Functions

**Important**: Most functions are pure data access (fast, free). Only 2 functions call sub-LLM.

| Function | What It Does | Calls Sub-LLM? | Cost |
|----------|--------------|----------------|------|
| `get_toc()` | Returns TOC from index | No | Free |
| `get_section_names()` | Returns section list from index | No | Free |
| `get_summary(name)` | Returns pre-computed summary | No | Free |
| `get_all_summaries()` | Returns all pre-computed summaries | No | Free |
| `read_section(name)` | Reads section from markdown file | No | Free |
| `read_section_chunk(...)` | Reads chunk from file | No | Free |
| `read_range(start, end)` | Reads byte range from file | No | Free |
| `grep_section(pattern, name)` | Regex search in section | No | Free |
| `grep_all(pattern)` | Regex search all sections | No | Free |
| `find_sections_by_keyword(kw)` | Keyword lookup in index | No | Free |
| **`llm_query(prompt)`** | **Calls sub-LLM with prompt** | **Yes** | ~$0.001 |
| **`ask_about_section(q, name)`** | **Calls sub-LLM with section content** | **Yes** | ~$0.001 |
| `FINAL(answer)` | Signals completion, returns answer | No | Free |

**Root LLM decides** when to call sub-LLM. Simple lookups = free. Semantic understanding = sub-LLM call.

```python
# Navigation (reads from pre-built index - FREE)
def get_toc() -> list[dict]: ...
def get_section_names() -> list[str]: ...
def get_summary(section_name: str) -> str: ...
def get_all_summaries() -> dict[str, str]: ...

# Reading (reads from markdown file - FREE)
def read_section(section_name: str) -> str: ...
def read_section_chunk(section_name: str, chunk_idx: int, chunk_size: int = 10000) -> str: ...
def read_range(start: int, end: int) -> str: ...

# Search (regex/index lookup - FREE)
def grep_section(pattern: str, section_name: str) -> list[str]: ...
def grep_all(pattern: str) -> dict[str, list[str]]: ...
def find_sections_by_keyword(keyword: str) -> list[str]: ...

# LLM (CALLS SUB-LLM - costs money)
def llm_query(prompt: str) -> str: ...                        # Calls sub_model
def ask_about_section(question: str, section_name: str) -> str: ...  # Calls sub_model

# Terminal
def FINAL(answer: str) -> None: ...
```

### 4.4 REPL Execution Design

**Core concept**: Root LLM writes Python code, we execute it, capture output, feed back.

```
+------------------+       +------------------+       +------------------+
|    Root LLM      | ----> |   Code Executor  | ----> |  Output Capture  |
|  (writes code)   |       |   (exec + globals)|      |  (stdout + vars) |
+------------------+       +------------------+       +------------------+
        ^                                                      |
        |                                                      |
        +----------------------- feedback ---------------------+
```

#### Execution Loop

```python
def execute_query(question: str, max_rounds: int = 10) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    for round in range(max_rounds):
        # 1. Root LLM generates code
        response = litellm.completion(model=root_model, messages=messages)
        code = extract_code_block(response)

        # 2. Execute in sandboxed environment
        output, final_answer = execute_code(code, repl_globals)

        # 3. Check if done
        if final_answer is not None:
            return final_answer

        # 4. Feed output back to LLM
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Output:\n{output}"})

    return "Max rounds exceeded"
```

#### Sandboxed Execution

```python
class REPLExecutor:
    def __init__(self, index: SingleDocIndex, sub_model: str):
        self.index = index
        self.sub_model = sub_model
        self.final_answer = None

    def execute_code(self, code: str) -> tuple[str, str | None]:
        # Capture stdout
        stdout_buffer = io.StringIO()

        # Build globals with REPL functions
        repl_globals = {
            "__builtins__": SAFE_BUILTINS,
            # Navigation
            "get_toc": self._get_toc,
            "get_section_names": self._get_section_names,
            "get_summary": self._get_summary,
            "get_all_summaries": self._get_all_summaries,
            # Reading
            "read_section": self._read_section,
            "read_section_chunk": self._read_section_chunk,
            "read_range": self._read_range,
            # Search
            "grep_section": self._grep_section,
            "grep_all": self._grep_all,
            "find_sections_by_keyword": self._find_sections_by_keyword,
            # LLM
            "llm_query": self._llm_query,
            "ask_about_section": self._ask_about_section,
            # Terminal
            "FINAL": self._final,
            "print": lambda *args: print(*args, file=stdout_buffer),
        }

        try:
            with redirect_stdout(stdout_buffer):
                exec(code, repl_globals)
        except Exception as e:
            stdout_buffer.write(f"Error: {type(e).__name__}: {e}")

        return stdout_buffer.getvalue(), self.final_answer

    def _final(self, answer: str):
        """Called by FINAL(answer) to signal completion."""
        self.final_answer = str(answer)
```

#### Safe Builtins

```python
SAFE_BUILTINS = {
    # Types
    "str": str, "int": int, "float": float, "bool": bool,
    "list": list, "dict": dict, "tuple": tuple, "set": set,
    # Iteration
    "range": range, "enumerate": enumerate, "zip": zip,
    "map": map, "filter": filter,
    # Aggregation
    "len": len, "sum": sum, "min": min, "max": max,
    "sorted": sorted, "reversed": reversed,
    "any": any, "all": all,
    # String
    "format": format, "repr": repr,
    # Errors (for try/except)
    "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError,
}
# Explicitly NOT included: open, exec, eval, import, __import__, compile, globals, locals
```

#### Example Execution Trace

**Question**: "What datasets were used in the experiments?"

**Round 1** - Root LLM generates:
```python
# Check summaries for relevant sections
summaries = get_all_summaries()  # <-- FREE: reads from index
for name, summary in summaries.items():
    if "dataset" in summary.lower() or "experiment" in summary.lower():
        print(f"Relevant: {name}")
```
*No sub-LLM called. Just index lookup.*

**Round 2** - Root LLM generates:
```python
# Read the datasets section
content = read_section("4.1 Datasets")  # <-- FREE: file read
print(content[:2000])
```
*No sub-LLM called. Just file read.*

**Round 3** - Root LLM generates:
```python
FINAL("""The paper uses three datasets:
1. OOLONG - 50 semantic aggregation tasks
2. S-NIAH - Needle-in-haystack benchmark from RULER
3. BrowseComp - Multi-document QA benchmark""")
```
*Done. Total sub-LLM calls: 0*

---

**Alternative trace** (when semantic understanding needed):

**Round 1** - Root LLM generates:
```python
sections = grep_all("ablation")  # <-- FREE: regex search
print(sections)
```

**Round 2** - Root LLM generates:
```python
# Need to understand what the ablations mean
result = ask_about_section(        # <-- CALLS SUB-LLM ($)
    "What ablations were run and what were the findings?",
    "5.2 Ablation Studies"
)
print(result)
```
*Sub-LLM called with section content.*

**Round 3** - Root LLM generates:
```python
FINAL(result)
```
*Done. Total sub-LLM calls: 1*

### 4.5 System Prompt for Root LLM

The system prompt tells the Root LLM what functions are available and how to explore the document.

```
You are analyzing a document that is too large to fit in context.
You have access to a pre-built index and functions to explore it.

## DOCUMENT INFO
- Source: {source_path}
- Size: {total_chars} characters
- Sections: {num_sections}

## TABLE OF CONTENTS
{toc_formatted}

## SECTION SUMMARIES
{summaries_formatted}

## AVAILABLE FUNCTIONS

### Navigation (FREE - no LLM cost)
- get_toc() -> list[dict]           # Full TOC with levels and char offsets
- get_section_names() -> list[str]  # Just section names
- get_summary(name) -> str          # Pre-computed summary for one section
- get_all_summaries() -> dict       # All summaries

### Reading (FREE - no LLM cost)
- read_section(name) -> str                    # Full section content
- read_section_chunk(name, idx, size) -> str   # Chunk of section (default 10K chars)
- read_range(start, end) -> str                # Raw character range

### Search (FREE - no LLM cost)
- grep_section(pattern, name) -> list[str]     # Regex search in one section
- grep_all(pattern) -> dict[str, list[str]]    # Regex search all sections
- find_sections_by_keyword(kw) -> list[str]    # Find sections by keyword

### LLM (COSTS MONEY - use sparingly)
- llm_query(prompt) -> str                     # Ask sub-LLM anything
- ask_about_section(question, name) -> str     # Ask sub-LLM about a section

### Terminal
- FINAL(answer)                                # Return your final answer

## STRATEGY

1. START with summaries - they're FREE and show what each section contains
2. USE grep_all() to find specific terms across the document
3. USE read_section() to get full content when you know which section
4. USE ask_about_section() ONLY when you need semantic understanding
5. CALL FINAL(answer) when you have the answer

## RULES

- Write Python code to explore the document
- Use print() to see intermediate results
- Prefer FREE functions over LLM functions
- When done, call FINAL(answer) with your answer
- If stuck after 3 attempts, call FINAL("I could not find this information in the document")

## ANTI-HALLUCINATION RULES

- NEVER make up information - only report what you READ from the document
- ALWAYS read_section() or grep_all() BEFORE calling FINAL()
- QUOTE exact text from the document when possible
- If information is NOT in the document, say "Not found in document"
- If UNCERTAIN, say "The document suggests X but is not explicit"

## EXAMPLE

Question: "What model architectures were tested?"

```python
# Step 1: Check summaries for relevant sections
summaries = get_all_summaries()
for name, summary in summaries.items():
    if "model" in summary.lower() or "architecture" in summary.lower():
        print(f"Relevant: {name}")
```

Then based on output, read the relevant section and extract the answer.
```

**Template variables:**
| Variable | Source |
|----------|--------|
| `{source_path}` | `index.source_path` |
| `{total_chars}` | `index.total_chars` |
| `{num_sections}` | `len(index.section_map)` |
| `{toc_formatted}` | Format TOC as indented list |
| `{summaries_formatted}` | Format as `Section: summary` pairs |

**TOC formatting example:**
```
1. Introduction
2. Related Work
3. Methods
   3.1 Data Collection
   3.2 Model Architecture
   3.3 Training
4. Experiments
   4.1 Datasets
   4.2 Baselines
   4.3 Results
5. Conclusion
```

**Summaries with Contextual Retrieval (Anthropic technique):**

Instead of simple `[under: parent]`, we use sub-LLM to generate rich contextual preamble for each section. This is based on [Anthropic's Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval).

**Prompt for context generation (Anthropic's exact approach):**

```
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{SECTION_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall
document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
```

**Implementation:**
```python
def generate_section_context(
    whole_document: str,
    section_content: str,
    model: str = "gemini/gemini-2.0-flash"  # cheap, fast
) -> str:
    """Generate contextual preamble for a section (50-100 tokens)."""

    # If document too large, use first 100K chars + TOC
    doc_for_context = whole_document[:100000] if len(whole_document) > 100000 else whole_document

    prompt = f"""<document>
{doc_for_context}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{section_content[:5000]}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."""

    response = litellm.completion(model=model, messages=[{"role": "user", "content": prompt}])
    return response.choices[0].message.content  # ~50-100 tokens
```

**Preprocessing flow:**
```
                                    +------------------+
                                    |  Whole Document  |
                                    |   (markdown)     |
                                    +--------+---------+
                                             |
              +------------------------------+------------------------------+
              |                              |                              |
              v                              v                              v
     +--------+--------+            +--------+--------+            +--------+--------+
     |    Section 1    |            |    Section 2    |            |    Section N    |
     +--------+--------+            +--------+--------+            +--------+--------+
              |                              |                              |
              v                              v                              v
     +--------+--------+            +--------+--------+            +--------+--------+
     | Sub-LLM: Generate|            | Sub-LLM: Generate|            | Sub-LLM: Generate|
     | context (50-100  |            | context (50-100  |            | context (50-100  |
     | tokens)          |            | tokens)          |            | tokens)          |
     +--------+--------+            +--------+--------+            +--------+--------+
              |                              |                              |
              v                              v                              v
     +--------+--------+            +--------+--------+            +--------+--------+
     | Context + Summary|            | Context + Summary|            | Context + Summary|
     | stored in index  |            | stored in index  |            | stored in index  |
     +-----------------+            +-----------------+            +-----------------+
```

**Example output:**
```
1. Introduction:
   <context>This is the introduction to "Recursive Language Models", a research
   paper proposing code-writing LLMs to process documents exceeding context windows.</context>
   Presents RLM approach where root LLM writes Python code to explore documents...

3.2 Model Architecture:
   <context>From the paper "Recursive Language Models" on long-document processing.
   This is section 3.2 under "Methods", describing the neural architecture.</context>
   Uses a 12-layer transformer with 768 hidden dimensions. The root LLM generates
   Python code while sub-LLM handles semantic tasks...

4.1 Datasets:
   <context>From "Recursive Language Models" paper. Section 4.1 under "Experiments",
   listing evaluation benchmarks for the proposed approach.</context>
   Evaluates on three benchmarks: OOLONG (semantic aggregation), S-NIAH (needle
   retrieval), and BrowseComp (multi-document QA)...
```

**Why contextual retrieval matters:**
- Root LLM understands document-level context for each section
- Better grounding: "This is from a paper about X" not just "under section Y"
- Improves relevance: LLM knows WHY a section exists, not just WHERE
- Matches Anthropic's finding: 49% reduction in retrieval failures

## 5. Technology Decisions

### 5.1 Technology Stack

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Doc Conversion | markitdown | Converts PDF/DOCX/HTML to markdown uniformly |
| LLM Calls | litellm | Unified API for OpenAI/Anthropic/others, handles retries |
| TOC Parsing | regex on markdown | Fast, no LLM cost, just parse `#` headings |
| Summaries | sub-LLM via litellm | Semantic task requires LLM |
| Code Execution | exec() + restricted globals | Simple, secure enough for generated code |
| Serialization | JSON | Human-readable, standard library |
| Type Hints | dataclasses + typing | Built-in, no extra deps |

### 5.2 TOC Parsing Strategy

**Tiered approach:**

| Tier | Method | Cost | Speed |
|------|--------|------|-------|
| 1 | Regex: `^#{1,6}\s+(.+)$` | Free | Instant |
| 2 | Lightweight LLM (gemini-flash, gpt-4o-mini) | ~$0.001 | ~2 sec |
| 3 | Single section fallback | Free | Instant |

```python
def parse_toc(markdown_text: str, llm_model: str = "gemini/gemini-2.0-flash") -> list[dict]:
    # Tier 1: Try regex
    sections = regex_parse_headings(markdown_text)
    if sections:
        return sections

    # Tier 2: Ask lightweight LLM to identify structure
    sections = llm_generate_toc(markdown_text[:50000], model=llm_model)
    if sections:
        return sections

    # Tier 3: Treat entire doc as one section
    return [{"title": "Document", "level": 1, "start_char": 0, "end_char": len(markdown_text)}]
```

**LLM prompt for Tier 2:**
```
Identify the major sections/chapters in this document.
Return JSON: [{"title": "...", "start_line": N}, ...]
```

## 6. File Structure

```
src/
  __init__.py
  single_doc/
    __init__.py
    converter.py      # markitdown wrapper, doc to markdown
    indexer.py        # TOC parsing, summary generation
    repl.py           # REPL environment and functions
    rlm.py            # SingleDocRLM main class
  utils/
    __init__.py
    token_counter.py  # Track token usage (wraps litellm callbacks)
```

**Dependencies**: `pip install litellm markitdown`

## 7. Error Handling

### 7.1 Error Types

| Error | Handling |
|-------|----------|
| File not found | Raise FileNotFoundError |
| TOC parse fails | Fallback to single section |
| Section not found | Raise KeyError with helpful message |
| LLM API error | Retry with exponential backoff (3 attempts) |
| Code execution error | Capture exception, feed back to LLM |
| Max rounds exceeded | Return partial answer with warning |

### 7.2 Fallback Behavior

```python
# If TOC detection fails:
section_map = {"document": (0, total_chars)}
summaries = {"document": "Full document content"}
```

## 8. Execution Safety

### 8.1 Restricted Globals
Code execution uses restricted globals - only REPL functions exposed:

```python
safe_globals = {
    "__builtins__": {
        "print": print,
        "len": len,
        "str": str,
        "int": int,
        "list": list,
        "dict": dict,
        "range": range,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
    },
    # REPL functions added here
}
```

### 8.2 Timeout
- Each code execution has 30 second timeout
- Total query has configurable max_rounds (default 10)

## 9. Implementation Order

### Phase 1: Foundation
1. `src/single_doc/converter.py` - markitdown wrapper
2. `src/utils/token_counter.py` - Token tracking (litellm callbacks)
3. Verify litellm + markitdown work with test files

### Phase 2: Indexing
4. `src/single_doc/indexer.py` - TOC parsing (regex on markdown)
5. Add summary generation to indexer (uses sub-LLM via litellm)
6. Index save/load (SingleDocIndex dataclass)

### Phase 3: REPL
7. `src/single_doc/repl.py` - REPL functions
8. Code execution with restricted globals
9. Output capture (stdout + return values)

### Phase 4: Integration
10. `src/single_doc/rlm.py` - SingleDocRLM class
11. Query execution loop
12. System prompt generation

### Phase 5: Polish
13. Error handling and retries
14. Interactive REPL mode
15. Logging and observability

## 10. Testing Strategy

### 10.1 Test Data
Use OOLONG benchmark documents - real structured text.

### 10.2 Unit Tests

| Component | Test Focus |
|-----------|------------|
| TOC Parser | Markdown headings, numbered sections, edge cases |
| Section Map | Boundary accuracy, overlaps, gaps |
| REPL Functions | Each function with mock data |
| Code Execution | Safe globals, timeout, error capture |

### 10.3 Integration Tests

| Test | What It Validates |
|------|-------------------|
| Build index on real doc | End-to-end indexing |
| Save/load roundtrip | Serialization correctness |
| Simple query | Full query flow |
| Multi-round query | REPL loop works |

## 11. Open Design Questions (Resolved)

| Question | Decision |
|----------|----------|
| No structure detected? | Fallback to single section |
| Incremental updates? | Full rebuild only (v1) |
| Max REPL rounds? | 10 (configurable) |
| Batch sub-LLM calls? | No batching v1, add if needed |
| Large section summarization? | Truncate to first 10K chars |

---

> Next: Run `/spec:approve design` when this document is complete and reviewed.
