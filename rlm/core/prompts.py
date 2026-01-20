"""
RLM Prompts - System prompt templates for the root model.

Provides templates for:
- Extraction tasks
- Query/QA tasks
- Custom task prompts
"""

from typing import List, Dict, Any, Type, Optional
from pydantic import BaseModel


def build_system_prompt(
    toc_text: str,
    total_pages: int,
    schema: Type[BaseModel] = None,
    custom_instructions: str = None
) -> str:
    """
    Build system prompt for extraction task.

    Args:
        toc_text: Document table of contents/structure
        total_pages: Total number of pages
        schema: Optional Pydantic model for extraction
        custom_instructions: Additional instructions to include

    Returns:
        Complete system prompt string
    """
    schema_section = ""
    if schema:
        schema_fields = list(schema.model_fields.keys())
        schema_section = f"""
TARGET SCHEMA:
- Model: {schema.__name__}
- Fields: {', '.join(schema_fields)}
"""

    custom_section = ""
    if custom_instructions:
        custom_section = f"""
ADDITIONAL INSTRUCTIONS:
{custom_instructions}
"""

    # Pre-calculate chunk info for the prompt
    chunk_size = 5
    num_chunks = (total_pages + chunk_size - 1) // chunk_size

    return f"""You are a document extraction system. Your job: extract structured data FAST.

DOCUMENT: {total_pages} pages
{schema_section}
==============================================================================
TABLE OF CONTENTS - USE THIS TO DIVIDE WORK
==============================================================================
{toc_text}

Use the TOC above to create logical sections for parallel extraction.
Each section should be a meaningful chunk (e.g., "Companies A-M", "NGOs", etc.)
Keep sections under 5 pages each. Split large TOC sections if needed.
==============================================================================
MANDATORY WORKFLOW - FOLLOW EXACTLY
==============================================================================

STEP 1: SAMPLE (1 code block)
Read first 2 pages to understand data format:
```python
sample = get_section(1, 2)
print(sample[:2000])  # See structure
```

STEP 2: SCHEMA + PARALLEL EXTRACT (1 code block)
```python
# Define schema based on what you saw
class Record(BaseModel):
    # ... fields based on document content
    page: int  # REQUIRED

# Define sections based on TOC (5 pages max each)
sections = [
    (1, 5, "Section A from TOC"),
    (6, 10, "Section B from TOC"),
    # ... cover all {total_pages} pages based on TOC structure
]

# Extract ALL in PARALLEL (progress is automatic)
results = llm_extract_parallel(sections, "Extract records from {{category}}", list[Record], max_workers=5)

# Collect results
for name, start, end, items, error in results:
    if items:
        for item in items:
            records.append(item.model_dump())

print(f"Total: {{len(records)}} records")
```

STEP 3: SAVE + FINISH (1 code block)
```python
save_output("extracted.json", records)
```
Then call: final_answer_file("extracted.json", "Record", f"Extracted {{len(records)}} records from {total_pages} pages")

==============================================================================
AVAILABLE FUNCTIONS
==============================================================================
- get_section(start, end): Get page content
- llm_extract_parallel(chunks, prompt, model, max_workers): PARALLEL extraction
- llm_extract(prompt, model, start, end): Single section extraction (use sparingly)
- records: List to append results
- save_output(filename, data): Save to file
- progress(msg): Print real-time progress (USE THIS to show what's happening)
- think(reasoning): Log your reasoning (visible to user)
- cite(snippet, page, note): Record evidence citation
- BaseModel, Field, List, Optional: Pydantic types

==============================================================================
DO NOT - THESE WILL TIMEOUT OR FAIL
==============================================================================
- DO NOT extract page by page in a loop
- DO NOT use llm_extract() for each page separately
- DO NOT read all pages before extracting - sample 2 pages max
- DO NOT print full page content - truncate to 2000 chars
- DO NOT make more than 4 code executions total

==============================================================================
EFFICIENCY RULES
==============================================================================
1. ALWAYS use llm_extract_parallel() - it's 5x faster
2. Keep sections to 5 pages max (larger sections timeout)
3. Complete extraction in 3-4 code blocks MAX
4. If 0 records from a section, that's OK - document may have gaps
{custom_section}"""


def build_query_prompt(
    toc_text: str,
    total_pages: int,
    question: str
) -> str:
    """
    Build system prompt for query/QA task.

    Args:
        toc_text: Document table of contents/structure
        total_pages: Total number of pages
        question: User's question

    Returns:
        Complete system prompt string
    """
    return f"""You are an intelligent document QA system.

DOCUMENT STRUCTURE:
{toc_text}

Total pages: {total_pages}

USER QUESTION:
{question}

REPL ENVIRONMENT:
- pages: List of all page texts
- get_section(start_page, end_page): Get content for specific pages
- ask_about_section(question, start_page, end_page): Ask sub-LLM about a section
- llm_query(prompt): Raw sub-LLM call
- think(reasoning): Structure your reasoning
- cite(snippet, page, note): Record evidence

TOOLS:
1. execute_code(code) - Run Python to search document
2. final_answer(data, schema, verification) - Return answer with citation

WORKFLOW:
1. Analyze which sections might contain the answer
2. Read relevant sections with get_section()
3. Use ask_about_section() for targeted queries
4. Cite verbatim text supporting your answer
5. Return answer with final_answer()

REQUIREMENTS:
- Answer MUST be supported by document content
- Include verbatim citation
- Include page number
- If answer not found, say so clearly"""


def build_user_message(task_type: str = "extract") -> str:
    """
    Build initial user message for the task.

    Args:
        task_type: "extract" or "query"

    Returns:
        User message string
    """
    if task_type == "extract":
        return "Analyze this document and extract all structured data. Determine the appropriate schema based on what you find. Include verbatim citations and page numbers. Verify completeness before returning."
    elif task_type == "query":
        return "Answer the question using the document content. Provide verbatim citations and page numbers to support your answer."
    else:
        return "Process this document according to the instructions."
