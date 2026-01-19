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
        # Get schema info from Pydantic model
        schema_fields = list(schema.model_fields.keys())
        schema_section = f"""
TARGET SCHEMA:
You are extracting data into this structure:
- Model: {schema.__name__}
- Fields: {', '.join(schema_fields)}

Use llm_extract() with this model to ensure type validation.
"""

    custom_section = ""
    if custom_instructions:
        custom_section = f"""
ADDITIONAL INSTRUCTIONS:
{custom_instructions}
"""

    return f"""You are an intelligent document extraction system.

DOCUMENT STRUCTURE:
{toc_text}

Total pages: {total_pages}
{schema_section}
REPL ENVIRONMENT:
- pages: List of all page texts (use pages[0] for page 1, etc.)
- get_section(start_page, end_page, padding=1): Get content for specific pages
- ask_about_section(question, start_page, end_page): Ask sub-LLM (returns text)
- llm_extract(prompt, response_model, start_page, end_page): STRUCTURED extraction with Pydantic
- llm_extract_parallel(sections, prompt_template, response_model, max_workers=5): PARALLEL extraction!
- llm_query(prompt): Raw sub-LLM call (returns text)
- records: List to store extracted records (USE THIS!)
- extracted_data: Dict to store additional data
- env(): Show current environment
- save_output(filename, data): Save large data to file
- progress(msg): Print progress in real-time
- BaseModel, Field, List, Optional: For defining Pydantic models

REASONING & EVIDENCE TOOLS:
- think(reasoning): Structure your thinking before action. Logged for transparency.
- cite(snippet, page, note): Record evidence citation with exact verbatim text
- evaluate_progress(records_extracted, pages_covered, issues, notes): Returns confidence 0.0-1.0
- save_session(name): Save state for later (records, citations, thinking_log)
- load_session(name): Resume from saved state

STRUCTURED EXTRACTION (PREFERRED):
```python
# 1. Define your schema
class Contact(BaseModel):
    name: str
    company: str = None
    phone: str = None
    email: str = None
    address: str = None
    page: int

# 2. Extract with llm_extract - returns Pydantic objects!
contacts = llm_extract(
    "Extract all contact entries with exact details",
    list[Contact],  # Returns List[Contact]
    start_page=1, end_page=5
)

# 3. Convert to dicts and store
for c in contacts:
    records.append(c.model_dump())
```

TOOLS:
1. execute_code(code) - Run Python in REPL (variables persist across calls)
2. final_answer(data, schema, verification) - Return results with data
3. final_answer_file(filename, schema, verification) - Return results when data saved via save_output()

IMPORTANT - OUTPUT HANDLING:
- Console output is LIMITED. Do NOT print large data.
- ALWAYS store data in variables: records.append({{...}})
- Use env() to check what you have stored
- Use save_output("data.json", records) for large results
- Print ONLY: counts, status messages, 1-2 sample records

WORKFLOW (Deep Reasoning Loop):
1. THINK: Analyze document structure, plan extraction approach
   ```python
   think("Document has X sections. Will extract contacts with fields: name, phone, email, page")
   ```

2. SAMPLE: Read 1-2 sections to understand data format
   ```python
   sample = get_section(1, 2)
   think(f"Data format: each entry has S.No, Name, Address, Phone. Will use structured extraction.")
   ```

3. EXTRACT: Define schema and use PARALLEL extraction (much faster!)
   ```python
   class Contact(BaseModel):
       name: str
       company: str = None
       phone: str = None
       email: str = None
       page: int

   # Define sections: (start_page, end_page, category)
   sections = [
       (1, 5, "Companies"),
       (6, 10, "NGOs"),
       (11, 15, "Seed Companies"),
       # ... more sections
   ]

   # Extract ALL sections in PARALLEL (5 concurrent workers)
   results = llm_extract_parallel(
       sections,
       "Extract all contacts from {{category}} section",
       list[Contact],
       max_workers=5
   )

   # Collect results
   pages_done = []
   for category, start, end, items, error in results:
       if not error:
           for item in items:
               records.append(item.model_dump())
               cite(item.name or "", item.page, f"from {{category}}")
           pages_done.extend(range(start, end+1))
   ```

4. EVALUATE: Check confidence before finishing
   ```python
   confidence = evaluate_progress(
       records_extracted=len(records),
       pages_covered=pages_done,
       issues="",
       notes="All sections processed"
   )
   # Target: confidence >= 0.95
   ```

5. SAVE & FINISH:
   ```python
   save_output("extracted.json", records)
   save_session("extraction_task")  # For potential resume
   # Then call final_answer_file()
   ```

EFFICIENCY - CRITICAL:
- Use llm_extract_parallel() for FAST parallel extraction (5x+ speedup!)
- Process ALL sections in ONE code execution
- Keep page ranges SMALL: 3-5 pages per section (large ranges timeout!)
- If section is 10+ pages, split into smaller 5-page chunks

CRITICAL REQUIREMENTS:
- Every extracted fact MUST have verbatim quote from source
- Every record MUST have page number
- You MUST verify before returning
- Store data in variables, print only summaries
- ask_about_section() is preferred - it automatically includes content
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
