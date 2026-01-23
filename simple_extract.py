"""
Simple RLM Extraction Script

A standalone script that:
1. Takes a PDF as input
2. Converts it to markdown using Mistral OCR
3. Interactively asks user what fields to extract
4. Runs extraction using root model + sub-LLM pattern

Usage:
    python simple_extract.py document.pdf           # Interactive mode
    python simple_extract.py document.pdf --auto    # Auto mode (no prompts)

Environment variables (or set in .env file):
    OPENROUTER_API_KEY - Required for LLM calls
    MISTRAL_API_KEY - Optional (has default)
"""

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

import os
import io
import re
import sys
import json
import base64
import argparse
import traceback
from pathlib import Path
from typing import List, Dict, Any, Type, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, Field, create_model

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ROOT_MODEL = "anthropic/claude-sonnet-4"  # Reasoning model
SUB_MODEL = "anthropic/claude-haiku-4.5"          # Fast extraction model
DISCOVERY_MODEL = "openai/gpt-4o-mini"    # Field discovery model
MAX_ITERATIONS = 40
MAX_WORKERS = 5

# -----------------------------------------------------------------------------
# 1. Mistral OCR Converter
# -----------------------------------------------------------------------------

MISTRAL_OCR_API = "https://mistral-3-ocr.vercel.app/api/mistral"

def convert_pdf_to_markdown(pdf_url: str, save_md_path: str = None) -> str:
    """
    Convert PDF URL to markdown using Mistral OCR API.

    Args:
        pdf_url: URL to PDF document
        save_md_path: Optional path to save markdown file

    Returns:
        Converted markdown text
    """
    import requests

    print(f"  Converting PDF from URL...")

    # Call the API
    response = requests.post(
        MISTRAL_OCR_API,
        json={
            "document_url": pdf_url,
            "format": "markdown"
        },
        timeout=300  # 5 min timeout for large documents
    )

    if response.status_code != 200:
        raise Exception(f"OCR API error ({response.status_code}): {response.text}")

    # API returns plain text markdown (Content-Type: text/markdown)
    markdown_text = response.text

    # Save markdown file if path provided
    if save_md_path:
        with open(save_md_path, 'w', encoding='utf-8') as f:
            f.write(markdown_text)
        print(f"  Saved markdown to: {save_md_path}")

    print(f"  Converted: {len(markdown_text)} chars")
    return markdown_text

# -----------------------------------------------------------------------------
# 2. Document Segmentation
# -----------------------------------------------------------------------------

def split_into_pages(text: str, lines_per_page: int = 50) -> List[str]:
    """
    Split text into pages using OCR page markers.

    Falls back to line-based splitting if no markers found.
    """
    # Try to split by OCR page markers first
    # Pattern matches: <!-- Page N --> or *Page number: N*
    page_pattern = r'(?=<!-- Page \d+ -->)'
    page_splits = re.split(page_pattern, text)

    # Filter out empty strings and clean up
    pages = [p.strip() for p in page_splits if p.strip()]

    if len(pages) > 1:
        # Successfully split by page markers
        # Add page number prefix for consistency
        result = []
        for i, page in enumerate(pages, 1):
            # Extract actual page number from content if present
            match = re.search(r'<!-- Page (\d+) -->', page)
            page_num = match.group(1) if match else str(i)
            result.append(f"### Page Number: [PG:{page_num}]\n{page}")
        return result

    # Fallback: split by lines if no page markers
    lines = text.split('\n')
    pages = []
    for i in range(0, len(lines), lines_per_page):
        page_lines = lines[i:i + lines_per_page]
        page_num = (i // lines_per_page) + 1
        page_text = f"### Page Number: [PG:{page_num}]\n" + '\n'.join(page_lines)
        pages.append(page_text)

    return pages

def generate_toc(pages: List[str]) -> str:
    """
    Simple TOC - just page count. Let the model explore.
    """
    return f"Document has {len(pages)} pages."

# -----------------------------------------------------------------------------
# 3. LLM Provider (OpenRouter)
# -----------------------------------------------------------------------------

class LLMProvider:
    """Simple OpenRouter-based LLM provider."""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable required")
        self._client = None
        self._instructor_client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.api_key
            )
        return self._client

    def _get_instructor_client(self):
        if self._instructor_client is None:
            import instructor
            self._instructor_client = instructor.from_openai(self._get_client())
        return self._instructor_client

    def chat(self, messages: List[Dict], model: str, tools: List[Dict] = None, timeout: float = 120.0) -> Dict:
        """Send chat completion request."""
        client = self._get_client()
        params = {"model": model, "messages": messages, "timeout": timeout}
        if tools:
            params["tools"] = tools
        response = client.chat.completions.create(**params)
        return response.model_dump()

    def extract(self, prompt: str, response_model: Type[BaseModel], model: str, timeout: float = 120.0):
        """Extract structured data using instructor."""
        client = self._get_instructor_client()
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_model=response_model,
            timeout=timeout
        )

# -----------------------------------------------------------------------------
# 4. Interactive Field Discovery
# -----------------------------------------------------------------------------

class SuggestedField(BaseModel):
    """A field suggested for extraction."""
    name: str = Field(description="Field name (snake_case)")
    field_type: str = Field(description="Type: str, int, float, bool, date, list[str]")
    description: str = Field(description="What this field contains")
    examples: List[str] = Field(description="2-3 example values from document")
    required: bool = Field(description="Whether this field is always present")

class DocumentAnalysis(BaseModel):
    """Analysis of document structure."""
    document_type: str = Field(description="Type of document (invoice, contract, report, etc.)")
    record_type: str = Field(description="What each record represents (person, item, transaction, etc.)")
    suggested_fields: List[SuggestedField] = Field(description="Fields to extract")
    notes: str = Field(description="Any important observations")

def discover_fields(provider: LLMProvider, sample_text: str) -> DocumentAnalysis:
    """Use LLM to analyze document and suggest fields."""
    prompt = f"""Analyze this document sample and suggest fields to extract.

DOCUMENT SAMPLE:
{sample_text[:4000]}

Based on this sample:
1. Identify the document type
2. Identify what each record/entry represents
3. Suggest fields to extract (use snake_case names)
4. For each field, provide type, description, and examples from the text

Focus on structured data that appears consistently throughout the document."""

    return provider.extract(prompt, DocumentAnalysis, DISCOVERY_MODEL)

def interactive_field_selection(analysis: DocumentAnalysis) -> List[Dict[str, Any]]:
    """Let user select and modify fields interactively."""
    print("\n" + "=" * 60)
    print("DOCUMENT ANALYSIS")
    print("=" * 60)
    print(f"Document Type: {analysis.document_type}")
    print(f"Record Type: {analysis.record_type}")
    print(f"Notes: {analysis.notes}")

    print("\n" + "-" * 60)
    print("SUGGESTED FIELDS")
    print("-" * 60)

    fields = []
    for i, field in enumerate(analysis.suggested_fields, 1):
        examples = ", ".join(f'"{ex}"' for ex in field.examples[:2])
        req = "*" if field.required else " "
        print(f"  [{i}]{req} {field.name} ({field.field_type})")
        print(f"       {field.description}")
        print(f"       Examples: {examples}")
        fields.append({
            "name": field.name,
            "type": field.field_type,
            "description": field.description,
            "required": field.required,
            "selected": True  # Default: all selected
        })

    print("\n" + "-" * 60)
    print("FIELD SELECTION")
    print("-" * 60)
    print("Options:")
    print("  - Press ENTER to accept all fields")
    print("  - Enter field numbers to toggle (e.g., '1 3 5' to deselect)")
    print("  - Enter 'add' to add a custom field")
    print("  - Enter 'done' when finished")

    while True:
        print("\nCurrently selected:")
        selected = [f for f in fields if f["selected"]]
        for f in selected:
            print(f"  + {f['name']} ({f['type']})")

        user_input = input("\n> ").strip().lower()

        if user_input == "" or user_input == "done":
            break
        elif user_input == "add":
            # Add custom field
            print("\nAdd custom field:")
            name = input("  Field name (snake_case): ").strip()
            if not name:
                continue
            ftype = input("  Type (str/int/float/bool/date/list[str]) [str]: ").strip() or "str"
            desc = input("  Description: ").strip() or name
            fields.append({
                "name": name,
                "type": ftype,
                "description": desc,
                "required": False,
                "selected": True
            })
            print(f"  Added: {name}")
        else:
            # Toggle fields by number
            try:
                nums = [int(x) for x in user_input.split()]
                for num in nums:
                    if 1 <= num <= len(fields):
                        fields[num - 1]["selected"] = not fields[num - 1]["selected"]
                        status = "selected" if fields[num - 1]["selected"] else "deselected"
                        print(f"  {fields[num - 1]['name']}: {status}")
            except ValueError:
                print("  Invalid input. Enter numbers separated by spaces.")

    return [f for f in fields if f["selected"]]

def create_dynamic_schema(fields: List[Dict[str, Any]], schema_name: str = "Record") -> Type[BaseModel]:
    """Create a Pydantic model from field definitions."""
    type_mapping = {
        "str": str,
        "string": str,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
        "bool": bool,
        "boolean": bool,
        "date": str,  # Store dates as strings
        "list[str]": List[str],
        "list": List[str],
    }

    field_definitions = {}
    for f in fields:
        field_type = type_mapping.get(f["type"].lower(), str)
        if f.get("required", False):
            field_definitions[f["name"]] = (field_type, Field(description=f.get("description", "")))
        else:
            field_definitions[f["name"]] = (Optional[field_type], Field(default=None, description=f.get("description", "")))

    # Always add page field
    field_definitions["page"] = (int, Field(description="Page number where this record was found"))

    return create_model(schema_name, **field_definitions)

# -----------------------------------------------------------------------------
# 5. REPL Environment
# -----------------------------------------------------------------------------

class REPLEnvironment:
    """Python REPL with persistent namespace."""

    def __init__(self, output_dir: Path, progress_callback: Callable[[str], None] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._progress_callback = progress_callback
        self._real_stdout = sys.stdout
        self.namespace: Dict[str, Any] = {}
        self._setup_builtins()

    def _setup_builtins(self):
        from typing import List, Optional
        from pydantic import BaseModel, Field

        self.namespace.update({
            "List": List,
            "Optional": Optional,
            "BaseModel": BaseModel,
            "Field": Field,
            "records": [],
            "extracted_data": {},
            "save_output": self._save_output,
            "progress": self._progress,
        })

    def _save_output(self, filename: str, data: Any) -> str:
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            if isinstance(data, (dict, list)):
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            else:
                f.write(str(data))
        return f"Saved to {filepath}"

    def _progress(self, msg: str):
        clean_msg = msg.encode('ascii', 'replace').decode('ascii')
        if self._progress_callback:
            self._progress_callback(clean_msg)
        else:
            self._real_stdout.write(f"  >> {clean_msg}\n")
            self._real_stdout.flush()

    def execute(self, code: str) -> str:
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            exec(code, self.namespace)
            result = sys.stdout.getvalue()
            return result or "(no output)"
        except Exception as e:
            tb_lines = traceback.format_exc().split('\n')
            line_no = None
            for tb_line in tb_lines:
                if 'line ' in tb_line and '<string>' in tb_line:
                    try:
                        line_no = int(tb_line.split('line ')[1].split(',')[0].split()[0])
                    except (ValueError, IndexError):
                        pass

            code_lines = code.split('\n')
            error_msg = f"Error: {type(e).__name__}: {e}\n"

            if line_no and 1 <= line_no <= len(code_lines):
                error_msg += f"\nAt line {line_no}:\n"
                start = max(0, line_no - 3)
                end = min(len(code_lines), line_no + 2)
                for i in range(start, end):
                    marker = ">>> " if i == line_no - 1 else "    "
                    error_msg += f"{marker}{i+1:3d} | {code_lines[i]}\n"

            return sys.stdout.getvalue() + error_msg
        finally:
            sys.stdout = old_stdout

    def get(self, key: str, default: Any = None) -> Any:
        return self.namespace.get(key, default)

    def update(self, values: Dict[str, Any]):
        self.namespace.update(values)

# -----------------------------------------------------------------------------
# 6. Regex Extraction Tools
# -----------------------------------------------------------------------------

import re

# Common regex patterns
PATTERNS = {
    "email": r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
    "phone": r'[\d\-\(\)\s\.]{7,}',
    "phone_labeled": r'(?:Off(?:ice)?|Resi(?:dence)?|Mob(?:ile)?|Fax|Tel|Ph)[:\.\s]*([0-9\-\(\)\s\.]+)',
    "url": r'https?://[^\s<>"{}|\\^`\[\]]+',
    "date": r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
    "currency": r'[\$\u20B9\u00A3\u20AC]\s*[\d,]+\.?\d*',
    "percentage": r'\d+\.?\d*\s*%',
}

def regex_extract_all(text: str, pattern: str, flags: int = 0) -> List[str]:
    """Extract all matches of a pattern from text."""
    return re.findall(pattern, text, flags)

def regex_extract_groups(text: str, pattern: str, flags: int = 0) -> List[Tuple]:
    """Extract all matches with capture groups."""
    return re.findall(pattern, text, flags)

def parse_html_table(text: str) -> List[Dict[str, str]]:
    """
    Parse HTML tables into list of dicts.

    Handles:
        <table><tr><th>Name</th><th>Phone</th></tr>
        <tr><td>John</td><td>123</td></tr></table>

    Returns: [{"Name": "John", "Phone": "123"}]
    """
    from html import unescape

    all_rows = []

    # Find all tables in text
    table_pattern = r'<table[^>]*>(.*?)</table>'
    tables = re.findall(table_pattern, text, re.DOTALL | re.IGNORECASE)

    for table_html in tables:
        headers = []
        rows = []

        # Find all rows
        row_pattern = r'<tr[^>]*>(.*?)</tr>'
        tr_matches = re.findall(row_pattern, table_html, re.DOTALL | re.IGNORECASE)

        for tr_html in tr_matches:
            # Check for header cells first
            header_pattern = r'<th[^>]*>(.*?)</th>'
            header_cells = re.findall(header_pattern, tr_html, re.DOTALL | re.IGNORECASE)

            if header_cells:
                # This is a header row
                headers = []
                for cell in header_cells:
                    # Clean HTML and decode entities
                    clean = re.sub(r'<[^>]+>', ' ', cell)
                    clean = unescape(clean).strip()
                    clean = ' '.join(clean.split())  # Normalize whitespace
                    headers.append(clean)
            else:
                # This is a data row
                cell_pattern = r'<td[^>]*>(.*?)</td>'
                data_cells = re.findall(cell_pattern, tr_html, re.DOTALL | re.IGNORECASE)

                if data_cells and headers:
                    row_data = {}
                    for i, cell in enumerate(data_cells):
                        if i < len(headers):
                            # Replace <br/> with newlines, strip other HTML
                            clean = re.sub(r'<br\s*/?>', '\n', cell, flags=re.IGNORECASE)
                            clean = re.sub(r'<[^>]+>', '', clean)
                            clean = unescape(clean).strip()
                            row_data[headers[i]] = clean
                    if row_data:
                        rows.append(row_data)

        all_rows.extend(rows)

    return all_rows


def parse_markdown_table(text: str) -> List[Dict[str, str]]:
    """
    Parse a markdown pipe table into list of dicts.

    Example:
        | Name | Phone |
        |------|-------|
        | John | 123   |

    Returns: [{"Name": "John", "Phone": "123"}]
    """
    lines = text.strip().split('\n')
    rows = []
    headers = []

    for line in lines:
        line = line.strip()
        if not line.startswith('|'):
            continue

        # Split by | and clean up
        cells = [c.strip() for c in line.split('|')]
        cells = [c for c in cells if c]  # Remove empty strings

        # Skip separator rows (|---|---|)
        if cells and all(set(c) <= set('- :') for c in cells):
            continue

        if not headers:
            headers = cells
        else:
            if len(cells) == len(headers):
                rows.append(dict(zip(headers, cells)))

    return rows

def parse_contact_block(text: str) -> Dict[str, Any]:
    """
    Parse a contact block with labeled phone numbers and email.

    Handles patterns like:
        Off: 040-1234567
        Resi: 040-9876543
        Mob: 98765-43210
        E-Mail: test@example.com
    """
    result = {}

    # Extract labeled phones
    phone_patterns = [
        (r'Off(?:ice)?[:\.\s]*([0-9\-\(\)\s\.\/]+)', 'office_phone'),
        (r'Resi(?:dence)?[:\.\s]*([0-9\-\(\)\s\.\/]+)', 'residential_phone'),
        (r'Mob(?:ile)?[:\.\s]*([0-9\-\(\)\s\.\/]+)', 'mobile_phone'),
        (r'Fax[:\.\s]*([0-9\-\(\)\s\.\/]+)', 'fax'),
        (r'Tel[:\.\s]*([0-9\-\(\)\s\.\/]+)', 'phone'),
    ]

    for pattern, field in phone_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            # Clean and join multiple numbers
            phones = [m.strip().rstrip('.-/') for m in matches]
            phones = [p for p in phones if p and len(p) >= 5]
            if phones:
                result[field] = phones[0] if len(phones) == 1 else phones

    # Extract email
    email_match = re.search(r'E-?Mail[:\.\s\-]*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', text, re.IGNORECASE)
    if email_match:
        result['email'] = email_match.group(1)
    else:
        # Try standalone email
        emails = re.findall(PATTERNS['email'], text)
        if emails:
            result['email'] = emails[0]

    return result

# -----------------------------------------------------------------------------
# 7. Extraction Functions (injected into REPL)
# -----------------------------------------------------------------------------

def create_extraction_functions(provider: LLMProvider, pages: List[str], progress_cb: Callable):
    """Create extraction functions to inject into REPL namespace."""

    def get_section(start_page: int, end_page: int, padding: int = 1) -> str:
        start = max(0, start_page - 1 - padding)
        end = min(len(pages), end_page + padding)
        return '\n\n'.join(pages[start:end])

    # -------------------------------------------------------------------------
    # Regex-based extraction tools (fast, no API calls)
    # -------------------------------------------------------------------------

    def regex_extract(
        pattern: str,
        start_page: int = 1,
        end_page: int = None,
        with_groups: bool = False
    ) -> List:
        """
        Extract all matches of a regex pattern from pages.

        Args:
            pattern: Regex pattern (use PATTERNS dict for common ones)
            start_page: Start page (1-indexed)
            end_page: End page (1-indexed), defaults to all pages
            with_groups: If True, return capture groups as tuples

        Returns:
            List of matches (strings or tuples if with_groups=True)

        Example:
            # Extract all emails
            emails = regex_extract(PATTERNS['email'], 1, 10)

            # Extract with custom pattern and groups
            matches = regex_extract(r'Name: (.+), Phone: (.+)', with_groups=True)
        """
        end_page = end_page or len(pages)
        start_idx = max(0, start_page - 1)
        end_idx = min(len(pages), end_page)
        text = '\n\n'.join(pages[start_idx:end_idx])

        progress_cb(f"regex_extract: {pattern[:40]}... (pages {start_page}-{end_page})")

        if with_groups:
            results = regex_extract_groups(text, pattern, re.IGNORECASE | re.MULTILINE)
        else:
            results = regex_extract_all(text, pattern, re.IGNORECASE | re.MULTILINE)

        progress_cb(f"regex_extract: Found {len(results)} matches")
        return results

    def regex_table(start_page: int = 1, end_page: int = None) -> List[Dict[str, str]]:
        """
        Parse tables from pages into list of dicts.
        Automatically detects HTML tables or markdown tables.

        Args:
            start_page: Start page (1-indexed)
            end_page: End page (1-indexed)

        Returns:
            List of dicts, one per table row

        Example:
            rows = regex_table(1, 5)
            for row in rows:
                print(row['Name'], row['Phone'])
        """
        end_page = end_page or len(pages)
        start_idx = max(0, start_page - 1)
        end_idx = min(len(pages), end_page)
        text = '\n\n'.join(pages[start_idx:end_idx])

        progress_cb(f"regex_table: Parsing tables from pages {start_page}-{end_page}")

        # Try HTML tables first (API returns HTML)
        results = parse_html_table(text)
        if results:
            progress_cb(f"regex_table: Found {len(results)} rows (HTML format)")
            return results

        # Fall back to markdown tables
        results = parse_markdown_table(text)
        progress_cb(f"regex_table: Found {len(results)} rows (markdown format)")
        return results

    def regex_contacts(start_page: int = 1, end_page: int = None) -> List[Dict[str, Any]]:
        """
        Extract contact information from table rows with labeled phones/emails.

        Handles both HTML and markdown tables like:
            | S.No | Name & Address | Phone No. |
            | 1.   | Mr. John...    | Off: 123  |

        Returns:
            List of contact dicts with parsed phone numbers and emails
        """
        end_page = end_page or len(pages)
        start_idx = max(0, start_page - 1)
        end_idx = min(len(pages), end_page)
        text = '\n\n'.join(pages[start_idx:end_idx])

        progress_cb(f"regex_contacts: Parsing contacts from pages {start_page}-{end_page}")

        # Try HTML tables first (API returns HTML)
        rows = parse_html_table(text)
        table_format = "HTML"
        if not rows:
            rows = parse_markdown_table(text)
            table_format = "markdown"

        progress_cb(f"regex_contacts: Found {len(rows)} table rows ({table_format} format)")

        contacts = []
        for i, row in enumerate(rows, 1):
            contact = {"s_no": i}

            # Find name/address column (usually has newlines from <br/> tags)
            for key, value in row.items():
                key_lower = key.lower()

                # Serial number column
                if 's.no' in key_lower or 'sno' in key_lower or 'sl' in key_lower:
                    contact['s_no'] = value.strip().rstrip('.')

                # Name & Address column
                elif 'name' in key_lower or 'address' in key_lower:
                    # Parse name & address block (lines separated by \n from <br/> tags)
                    lines = [l.strip() for l in value.split('\n') if l.strip()]
                    if lines:
                        contact['name'] = lines[0]
                        # Check for designation in parentheses
                        name_match = re.match(r'([^(]+)\s*\(([^)]+)\)', lines[0])
                        if name_match:
                            contact['name'] = name_match.group(1).strip()
                            contact['designation'] = name_match.group(2).strip()

                        if len(lines) > 1:
                            contact['company_name'] = lines[1]
                        if len(lines) > 2:
                            contact['address'] = '\n'.join(lines[2:])

                # Phone/Contact column
                elif 'phone' in key_lower or 'contact' in key_lower:
                    # Parse phone block with labeled numbers
                    parsed = parse_contact_block(value)
                    contact.update(parsed)

            # Add page number (estimate based on position in document)
            # More accurate: count which page section this row appears in
            page_num = start_page + (i // 8)  # ~8 contacts per page
            contact['page'] = min(page_num, end_page)

            if contact.get('name'):
                contacts.append(contact)

        progress_cb(f"regex_contacts: Extracted {len(contacts)} contacts")
        return contacts

    # -------------------------------------------------------------------------
    # LLM-based extraction tools (flexible, semantic understanding)
    # -------------------------------------------------------------------------

    def llm_extract(prompt: str, response_model: Type[BaseModel], start_page: int = None, end_page: int = None):
        """
        Extract structured data using LLM (use for unstructured content).

        For structured tables, prefer regex_table() or regex_contacts().
        """
        if start_page and end_page:
            start_idx = max(0, start_page - 1)
            end_idx = min(len(pages), end_page)
            content = '\n\n'.join(pages[start_idx:end_idx])
            full_prompt = f"{prompt}\n\nDOCUMENT SECTION (pages {start_page}-{end_page}):\n{content}"
        else:
            full_prompt = prompt

        progress_cb(f"llm_extract: {prompt[:50]}...")
        return provider.extract(full_prompt, response_model, SUB_MODEL)

    def llm_extract_parallel(
        sections: List[Tuple[int, int, str]],
        prompt_template: str,
        response_model: Type[BaseModel],
        max_workers: int = MAX_WORKERS
    ):
        """
        Extract from multiple sections in parallel using LLM.

        For structured tables, prefer regex_table() or regex_contacts().
        """
        def extract_one(section):
            start, end, category = section
            prompt = prompt_template.format(category=category)
            try:
                start_idx = max(0, start - 1)
                end_idx = min(len(pages), end)
                content = '\n\n'.join(pages[start_idx:end_idx])
                full_prompt = f"{prompt}\n\nDOCUMENT SECTION (pages {start}-{end}):\n{content}"
                result = provider.extract(full_prompt, response_model, SUB_MODEL)
                return (category, start, end, result, None)
            except Exception as e:
                return (category, start, end, None, str(e))

        progress_cb(f"Starting parallel extraction of {len(sections)} sections...")

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(extract_one, s): s for s in sections}
            for future in as_completed(futures):
                category, start, end, data, error = future.result()
                if error:
                    progress_cb(f"FAILED: {category} (pages {start}-{end}): {error}")
                    results.append((category, start, end, [], error))
                else:
                    count = len(data) if isinstance(data, list) else 1
                    progress_cb(f"DONE: {category} (pages {start}-{end}): {count} records")
                    results.append((category, start, end, data, None))

        return results

    # -------------------------------------------------------------------------
    # Utility functions
    # -------------------------------------------------------------------------

    def think(reasoning: str):
        progress_cb(f"THINK: {reasoning[:100]}...")

    def cite(snippet: str, page: int, note: str = ""):
        progress_cb(f"CITE: page {page} - {snippet[:50]}...")

    def clean_record(record: dict) -> dict:
        """
        Remove null values, empty strings, and 'N/A' from record.
        """
        cleaned = {}
        for key, value in record.items():
            # Skip null, empty string, and N/A values
            if value is None:
                continue
            if isinstance(value, str) and value.strip() in ("", "N/A", "n/a"):
                continue
            cleaned[key] = value
        return cleaned

    return {
        # Document access
        "get_section": get_section,
        # Regex tools (fast, no API)
        "regex_extract": regex_extract,
        "regex_table": regex_table,
        "regex_contacts": regex_contacts,
        "PATTERNS": PATTERNS,
        "re": re,  # Give access to re module
        # LLM tools (flexible, semantic)
        "llm_extract": llm_extract,
        "llm_extract_parallel": llm_extract_parallel,
        # Utilities
        "think": think,
        "cite": cite,
        "clean_record": clean_record,
    }

# -----------------------------------------------------------------------------
# 7. System Prompt
# -----------------------------------------------------------------------------

def build_system_prompt(toc_text: str, total_pages: int, schema: Type[BaseModel] = None, fields_info: str = None) -> str:
    schema_section = ""
    if schema:
        schema_fields = list(schema.model_fields.keys())
        schema_section = f"\nTARGET SCHEMA:\n- Model: {schema.__name__}\n- Fields: {', '.join(schema_fields)}\n"

    if fields_info:
        schema_section += f"\nFIELD DETAILS:\n{fields_info}\n"

    return f"""You are a document extraction system. Your job: extract structured data FAST.

DOCUMENT: {total_pages} pages
{schema_section}
==============================================================================
DOCUMENT INFO
==============================================================================
{toc_text}

==============================================================================
PHASE 1: EXPLORE
==============================================================================

Sample the document to understand its structure:
```python
print("=== BEGINNING (pages 1-2) ===")
print(get_section(1, 2)[:2000])

print("\\n=== MIDDLE ===")
print(get_section(total_pages//2, total_pages//2+1)[:2000])

print("\\n=== END ===")
print(get_section(total_pages-1, total_pages)[:2000])
```

==============================================================================
PHASE 2: ANALYZE & DECIDE
==============================================================================

After exploring, YOU MUST output your analysis:

```python
print("=== ANALYSIS ===")
print("Document type: [directory/contract/form/report/invoice/other]")
print("Content density: [dense (many records/page) / sparse (few records/page)]")
print("Section structure: [uniform throughout / different sections with headers]")
print("")
print("=== CHUNKING DECISION ===")
print("Strategy: [1-page / 2-page / 5-page / by-section-headers]")
print("Reason: [why this chunking makes sense for this document]")
print("")
print("=== SCHEMA DECISION ===")
print("Number of schemas needed: [1 / multiple]")
print("Schema fields: [list only the relevant fields for this document]")
```

CHUNKING GUIDELINES:
- Dense directory (many entries/page): 1-2 pages at a time
- Contract with sections: Extract by section/topic
- Sparse report: 5-10 pages at a time
- Single page form: Whole document at once

SCHEMA GUIDELINES:
- Only include fields that EXIST in this document
- Different sections MAY need different schemas
- Always include: record_type, page
- NEVER create a giant schema with all possible fields

==============================================================================
PHASE 3: EXTRACT
==============================================================================

Execute based on YOUR decisions from Phase 2:
```python
# Define minimal schema(s) based on your analysis
class RecordTypeA(BaseModel):
    record_type: str = "TypeA"
    # ONLY fields relevant to this document
    page: int

# Create sections based on your chunking decision
sections = [...]  # Based on your decision

# Extract
results = llm_extract_parallel(sections, "Extract all records", List[RecordTypeA])
for category, start, end, items, error in results:
    if items and not error:
        for item in items:
            records.append(clean_record(item.model_dump()))
```

==============================================================================
PHASE 4: SAVE & FINISH
==============================================================================
```python
save_output("extracted.json", records)
```
Then call: final_answer_file("extracted.json", ...)

==============================================================================
AVAILABLE FUNCTIONS
==============================================================================

LLM EXTRACTION:
- llm_extract(prompt, schema, start, end): Extract from single section
- llm_extract_parallel(sections, prompt, schema): Extract from multiple sections in parallel

DOCUMENT ACCESS:
- get_section(start, end): Get page content (1-indexed)
- pages: List of all pages
- total_pages: Number of pages

DATA MANAGEMENT:
- records: List to append results
- clean_record(dict): Remove null/empty values from a record - USE THIS!
- save_output(filename, data): Save to JSON file
- progress(msg): Print progress message

PYDANTIC (for schemas):
- BaseModel, Field, List, Optional

REGEX (optional):
- regex_extract(pattern, start, end): Extract pattern matches
"""

# -----------------------------------------------------------------------------
# 8. Tool Definitions
# -----------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute Python code in REPL. Variables persist across calls.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code to execute"}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer",
            "description": "Return extracted data with schema and verification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "data": {"type": "array", "description": "Extracted records"},
                    "schema": {"type": "object"},
                    "verification": {"type": "object"}
                },
                "required": ["data", "schema", "verification"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "final_answer_file",
            "description": "Return results from saved file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {"type": "string"},
                    "schema": {"type": "object"},
                    "verification": {"type": "object"}
                },
                "required": ["filename", "schema", "verification"]
            }
        }
    }
]

# -----------------------------------------------------------------------------
# 9. Main Extraction Loop
# -----------------------------------------------------------------------------

def run_extraction(
    pdf_url: str,
    schema: Type[BaseModel] = None,
    fields_info: str = None,
    query: str = None,
    verbose: bool = True,
    save_md_path: str = None,
    preconverted_text: str = None,
    preconverted_pages: List[str] = None
) -> Dict[str, Any]:
    """
    Main extraction function.

    Args:
        pdf_url: URL to PDF document
        schema: Optional Pydantic model for extraction
        fields_info: Human-readable field descriptions
        query: User query describing what to extract (e.g., "Extract all company names and emails")
        verbose: Print progress
        save_md_path: Optional path to save markdown file
        preconverted_text: Already converted text (skip OCR if provided)
        preconverted_pages: Already segmented pages (skip segmentation if provided)

    Returns:
        Dict with extracted data, iterations, etc.
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    def progress(msg: str):
        if verbose:
            clean = msg.encode('ascii', 'replace').decode('ascii')
            sys.__stdout__.write(f"  >> {clean}\n")
            sys.__stdout__.flush()

    # Step 1: Convert PDF (skip if pre-converted)
    if preconverted_text:
        text = preconverted_text
        print(f"\n[1/3] Using pre-converted document ({len(text)} chars)")
    else:
        print(f"\n[1/3] Converting PDF from URL...")
        text = convert_pdf_to_markdown(pdf_url, save_md_path=save_md_path)
        progress(f"Document length: {len(text)} chars")

    # Step 2: Segment (skip if pre-segmented)
    if preconverted_pages:
        pages = preconverted_pages
        print(f"[2/3] Using pre-segmented pages ({len(pages)} pages)")
    else:
        print(f"\n[2/3] Segmenting document...")
        pages = split_into_pages(text, lines_per_page=50)
        progress(f"Created {len(pages)} pages")

    toc_text = generate_toc(pages)
    print("TOC Text:", toc_text)

    # Step 3: Setup REPL
    step_prefix = "[2/3]" if preconverted_text else "[3/4]"
    print(f"\n{step_prefix} Setting up extraction environment...")
    provider = LLMProvider()
    repl = REPLEnvironment(output_dir=results_dir, progress_callback=progress)

    # Add document data
    repl.update({
        "pages": pages,
        "total_pages": len(pages),
    })

    # Add extraction functions
    extraction_funcs = create_extraction_functions(provider, pages, progress)
    repl.update(extraction_funcs)

    # Build prompt
    system_prompt = build_system_prompt(toc_text, len(pages), schema, fields_info)

    # Use query if provided, otherwise default message
    if query:
        user_msg = f"""USER REQUEST: {query}

Extract the requested data from this document. Include page numbers for each record."""
    else:
        user_msg = "Analyze this document and extract all structured data. Include page numbers."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg}
    ]

    # Step 4: Run loop
    step_prefix = "[3/3]" if preconverted_text else "[4/4]"
    print(f"\n{step_prefix} Running extraction with {ROOT_MODEL}...")

    for iteration in range(MAX_ITERATIONS):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        # Call root model
        response = provider.chat(messages, ROOT_MODEL, tools=TOOLS)
        msg = response["choices"][0]["message"]

        if msg.get("content") and verbose:
            print(f"LLM: {msg['content'][:300]}...")

        tool_calls = msg.get("tool_calls", [])
        if not tool_calls:
            return {"error": "No final_answer called", "iterations": iteration + 1}

        messages.append(msg)

        # Process tool calls
        for tool_call in tool_calls:
            name = tool_call["function"]["name"]
            try:
                args = json.loads(tool_call["function"]["arguments"]) if tool_call["function"].get("arguments") else {}
            except json.JSONDecodeError:
                args = {}

            if verbose:
                print(f"Tool: {name}")

            if name == "execute_code":
                code = args.get("code", "")

                if verbose:
                    print("-" * 50)
                    for i, line in enumerate(code.split("\n"), 1):
                        clean = line.encode('ascii', 'replace').decode('ascii')
                        print(f"  {i:3d} | {clean}")
                    print("-" * 50)

                result = repl.execute(code)

                if verbose:
                    output = result[:800] + "\n...(truncated)" if len(result) > 800 else result
                    print(f"OUTPUT:\n{output}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": result
                })

            elif name == "final_answer":
                return {
                    "data": args.get("data", []),
                    "schema": args.get("schema", {}),
                    "verification": args.get("verification", {}),
                    "iterations": iteration + 1,
                }

            elif name == "final_answer_file":
                filename = args.get("filename", "")
                filepath = results_dir / filename
                data = []
                if filepath.exists():
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                return {
                    "data": data,
                    "data_file": str(filepath),
                    "schema": args.get("schema", {}),
                    "verification": args.get("verification", {}),
                    "iterations": iteration + 1,
                }

    return {"error": "Max iterations reached", "iterations": MAX_ITERATIONS}

# -----------------------------------------------------------------------------
# 10. Interactive Mode
# -----------------------------------------------------------------------------

def run_interactive(pdf_url: str, save_md_path: str = None, verbose: bool = True) -> Dict[str, Any]:
    """
    Run extraction with interactive field selection.

    Flow:
    1. Convert PDF from URL
    2. Analyze document and suggest fields
    3. Let user select/modify fields
    4. Run extraction with selected fields
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("INTERACTIVE RLM EXTRACTION")
    print("=" * 60)

    # Step 1: Convert PDF
    print(f"\n[1/5] Converting PDF from URL...")
    text = convert_pdf_to_markdown(pdf_url, save_md_path=save_md_path)
    print(f"  Document length: {len(text)} chars")

    # Step 2: Show sample
    print("\n[2/5] Document Preview")
    print("-" * 60)
    preview = text[:1500]
    # Clean for display
    preview_clean = preview.encode('ascii', 'replace').decode('ascii')
    print(preview_clean)
    print("-" * 60)
    print("(showing first 1500 characters)")

    # Step 3: Analyze and suggest fields
    print("\n[3/5] Analyzing document structure...")
    provider = LLMProvider()

    # Get first few pages for analysis
    pages = split_into_pages(text, lines_per_page=50)
    sample_text = '\n\n'.join(pages[:3]) if len(pages) >= 3 else text[:5000]

    analysis = discover_fields(provider, sample_text)

    # Step 4: Interactive field selection
    print("\n[4/5] Field Selection")
    selected_fields = interactive_field_selection(analysis)

    if not selected_fields:
        print("\nNo fields selected. Exiting.")
        return {"error": "No fields selected", "data": []}

    # Create dynamic schema
    schema = create_dynamic_schema(selected_fields, analysis.record_type.replace(" ", ""))

    # Build fields info for prompt
    fields_info = "\n".join([
        f"- {f['name']} ({f['type']}): {f['description']}"
        for f in selected_fields
    ])

    print("\n" + "-" * 60)
    print("EXTRACTION CONFIGURATION")
    print("-" * 60)
    print(f"Document Type: {analysis.document_type}")
    print(f"Record Type: {analysis.record_type}")
    print(f"Fields to extract:")
    for f in selected_fields:
        print(f"  - {f['name']} ({f['type']})")

    # Confirm
    confirm = input("\nProceed with extraction? [Y/n]: ").strip().lower()
    if confirm == 'n':
        print("Extraction cancelled.")
        return {"error": "Cancelled by user", "data": []}

    # Step 5: Run extraction (pass pre-converted data to avoid re-processing)
    print("\n[5/5] Running extraction...")

    return run_extraction(
        pdf_url,
        schema=schema,
        fields_info=fields_info,
        verbose=verbose,
        preconverted_text=text,
        preconverted_pages=pages
    )

# -----------------------------------------------------------------------------
# 11. CLI Entry Point
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Simple RLM PDF Extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python simple_extract.py "https://example.com/doc.pdf"                    # Interactive mode
  python simple_extract.py "https://example.com/doc.pdf" --auto             # Auto mode (extract everything)
  python simple_extract.py "https://example.com/doc.pdf" --query "company names and emails"
  python simple_extract.py "https://example.com/doc.pdf" -q "phone numbers" # Short form
  python simple_extract.py "https://example.com/doc.pdf" -m doc.md          # Save markdown
        """
    )
    parser.add_argument("url", help="URL to PDF document")
    parser.add_argument("-o", "--output", help="Output JSON file", default="results/output.json")
    parser.add_argument("-m", "--save-md", help="Save markdown to file", default=None)
    parser.add_argument("-q", "--query", help="What to extract (e.g., 'company names and phone numbers')")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode (less output)")
    parser.add_argument("--auto", action="store_true", help="Auto mode - skip interactive field selection")

    args = parser.parse_args()

    if not args.url.startswith(('http://', 'https://')):
        print(f"Error: Expected URL, got: {args.url}")
        sys.exit(1)

    # Run in appropriate mode
    if args.query:
        # Query mode - extract specific data
        print("=" * 60)
        print("Simple RLM Extraction (Query Mode)")
        print("=" * 60)
        print(f"Query: {args.query}")
        result = run_extraction(
            args.url,
            query=args.query,
            save_md_path=args.save_md,
            verbose=not args.quiet
        )
    elif args.auto:
        # Auto mode - extract everything
        print("=" * 60)
        print("Simple RLM Extraction (Auto Mode)")
        print("=" * 60)
        result = run_extraction(args.url, save_md_path=args.save_md, verbose=not args.quiet)
    else:
        # Interactive mode
        result = run_interactive(args.url, save_md_path=args.save_md, verbose=not args.quiet)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

    print("\n" + "=" * 60)
    print("EXTRACTION COMPLETE")
    print("=" * 60)

    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"Records extracted: {len(result.get('data', []))}")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Output saved to: {output_path}")

        if result.get("data"):
            print(f"\nSample record:")
            sample = result["data"][0]
            # Pretty print with truncation for long values
            for k, v in sample.items():
                v_str = str(v)
                if len(v_str) > 50:
                    v_str = v_str[:50] + "..."
                print(f"  {k}: {v_str}")

if __name__ == "__main__":
    main()
