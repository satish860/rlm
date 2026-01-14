# RLM Spec: Single Document Approach

## Overview

Processing a single long document (e.g., research paper, book, legal contract, report) that exceeds LLM context window.

---

## 1. Problem

```
Input: Single document (1M - 100M tokens)
Goal: Answer queries requiring deep understanding
Constraint: LLM context window (~200K tokens)
```

**Examples:**
- 500-page legal contract: "What are the termination clauses?"
- Research paper: "What ablations did they run?"
- Technical manual: "How do I configure authentication?"
- Novel: "How does the protagonist's motivation change?"

---

## 2. Architecture

```
+------------------------+
|     Single Document    |
|     (file path)        |
+------------------------+
            |
            v
+------------------------+
|    Pre-processor       |
|    (runs once)         |
+------------------------+
            |
            v
+------------------------+
|    Document Index      |
|    - TOC               |
|    - Section Map       |
|    - Section Summaries |
+------------------------+
            |
            v
+------------------------+
|      REPL Env          |
+------------------------+
            |
            v
+------------------------+
|      Root LLM          |
+------------------------+
            |
            v
+------------------------+
|      Sub-LLM           |
+------------------------+
```

---

## 3. Pre-processing Phase

### 3.1 TOC Generation

```python
def generate_toc(doc_path: str, llm) -> dict:
    """
    Generate table of contents from document.

    Returns:
        {
            "sections": [
                {
                    "title": "1. Introduction",
                    "level": 1,
                    "start_char": 0,
                    "end_char": 5000,
                    "children": [...]
                },
                ...
            ]
        }
    """
    # Read beginning to detect structure
    with open(doc_path, 'r') as f:
        preview = f.read(50000)

    # Detect document type
    doc_type = llm.query(f"""
        What type of document is this?
        Options: research_paper, legal, technical_manual, book, report, other

        Preview:
        {preview[:5000]}
    """)

    # Generate TOC based on type
    if doc_type == "research_paper":
        return _toc_from_headings(doc_path)
    elif doc_type == "legal":
        return _toc_from_numbered_sections(doc_path)
    elif doc_type == "book":
        return _toc_from_chapters(doc_path)
    else:
        return _toc_from_llm(doc_path, llm)


def _toc_from_headings(doc_path: str) -> dict:
    """Extract TOC from markdown/latex style headings"""
    import re

    sections = []
    with open(doc_path, 'r') as f:
        content = f.read()

    # Match patterns like "## Section Name" or "1.2 Section Name"
    patterns = [
        r'^(#{1,6})\s+(.+)$',           # Markdown
        r'^(\d+\.[\d.]*)\s+(.+)$',       # Numbered
        r'^\\section\{(.+)\}',           # LaTeX
    ]

    for pattern in patterns:
        for match in re.finditer(pattern, content, re.MULTILINE):
            sections.append({
                "title": match.group(2) if len(match.groups()) > 1 else match.group(1),
                "start_char": match.start(),
                "level": _detect_level(match),
            })

    # Calculate end positions
    for i, section in enumerate(sections[:-1]):
        section["end_char"] = sections[i+1]["start_char"]
    if sections:
        sections[-1]["end_char"] = len(content)

    return {"sections": sections}
```

### 3.2 Section Summaries

```python
def generate_summaries(doc_path: str, toc: dict, llm) -> dict:
    """
    Generate summary for each section.

    Returns:
        {
            "1. Introduction": "This section covers...",
            "2. Methods": "Describes the experimental...",
            ...
        }
    """
    summaries = {}

    for section in toc["sections"]:
        content = read_section(doc_path, section["start_char"], section["end_char"])

        # Use sub-LLM for summarization (cheaper)
        summary = llm.query(f"""
            Summarize this section in 2-3 sentences.
            Focus on: main topic, key points, important terms.

            Section: {section["title"]}
            Content:
            {content[:10000]}
        """)

        summaries[section["title"]] = summary

    return summaries
```

### 3.3 Document Index Structure

```python
@dataclass
class SingleDocIndex:
    path: str
    doc_type: str
    total_chars: int
    toc: dict                    # Hierarchical TOC
    section_map: dict            # section_name -> (start, end)
    summaries: dict              # section_name -> summary
    keywords: dict               # section_name -> [keywords]

    def save(self, index_path: str):
        """Persist index to disk"""
        with open(index_path, 'w') as f:
            json.dump(asdict(self), f)

    @classmethod
    def load(cls, index_path: str) -> 'SingleDocIndex':
        """Load index from disk"""
        with open(index_path, 'r') as f:
            return cls(**json.load(f))
```

---

## 4. REPL Environment

### 4.1 Available Functions

```python
class SingleDocREPL:
    """REPL environment for single document exploration"""

    def __init__(self, doc_path: str, index: SingleDocIndex, sub_llm):
        self.doc_path = doc_path
        self.index = index
        self.sub_llm = sub_llm

    # === Navigation Functions ===

    def get_toc(self) -> dict:
        """Return full table of contents"""
        return self.index.toc

    def get_section_names(self) -> list:
        """Return list of all section names"""
        return list(self.index.section_map.keys())

    def get_summary(self, section_name: str) -> str:
        """Get summary of a specific section"""
        return self.index.summaries.get(section_name, "No summary available")

    def get_all_summaries(self) -> dict:
        """Get all section summaries"""
        return self.index.summaries

    # === Reading Functions ===

    def read_section(self, section_name: str) -> str:
        """Read full content of a section"""
        start, end = self.index.section_map[section_name]
        with open(self.doc_path, 'r') as f:
            f.seek(start)
            return f.read(end - start)

    def read_section_chunk(self, section_name: str, chunk_idx: int, chunk_size: int = 10000) -> str:
        """Read a chunk of a section"""
        start, end = self.index.section_map[section_name]
        chunk_start = start + (chunk_idx * chunk_size)
        chunk_end = min(chunk_start + chunk_size, end)

        with open(self.doc_path, 'r') as f:
            f.seek(chunk_start)
            return f.read(chunk_end - chunk_start)

    def read_range(self, start: int, end: int) -> str:
        """Read arbitrary character range"""
        with open(self.doc_path, 'r') as f:
            f.seek(start)
            return f.read(end - start)

    # === Search Functions ===

    def grep_section(self, pattern: str, section_name: str) -> list:
        """Search within a specific section"""
        content = self.read_section(section_name)
        return re.findall(pattern, content, re.IGNORECASE)

    def grep_all(self, pattern: str) -> dict:
        """Search all sections, return matches by section"""
        results = {}
        for section_name in self.index.section_map:
            matches = self.grep_section(pattern, section_name)
            if matches:
                results[section_name] = matches
        return results

    def find_sections_by_keyword(self, keyword: str) -> list:
        """Find sections containing keyword in their keywords list"""
        matching = []
        for section_name, keywords in self.index.keywords.items():
            if keyword.lower() in [k.lower() for k in keywords]:
                matching.append(section_name)
        return matching

    # === LLM Functions ===

    def llm_query(self, prompt: str) -> str:
        """Call sub-LLM for semantic tasks"""
        return self.sub_llm.query(prompt)

    def ask_about_section(self, question: str, section_name: str) -> str:
        """Ask a question about a specific section"""
        content = self.read_section(section_name)
        return self.llm_query(f"""
            Answer this question based on the section content.

            Question: {question}

            Section "{section_name}":
            {content[:50000]}
        """)
```

### 4.2 Environment Setup

```python
def create_repl_env(doc_path: str, index: SingleDocIndex, sub_llm) -> dict:
    """Create REPL environment with all functions exposed"""

    repl = SingleDocREPL(doc_path, index, sub_llm)

    return {
        # Metadata
        "doc_path": doc_path,
        "doc_type": index.doc_type,
        "total_chars": index.total_chars,

        # Index data
        "toc": index.toc,
        "section_names": list(index.section_map.keys()),
        "summaries": index.summaries,

        # Functions
        "get_toc": repl.get_toc,
        "get_section_names": repl.get_section_names,
        "get_summary": repl.get_summary,
        "get_all_summaries": repl.get_all_summaries,
        "read_section": repl.read_section,
        "read_section_chunk": repl.read_section_chunk,
        "read_range": repl.read_range,
        "grep_section": repl.grep_section,
        "grep_all": repl.grep_all,
        "find_sections_by_keyword": repl.find_sections_by_keyword,
        "llm_query": repl.llm_query,
        "ask_about_section": repl.ask_about_section,
    }
```

---

## 5. Query Flow

### 5.1 System Prompt for Root LLM

```
You are analyzing a single long document that exceeds your context window.
You have access to a pre-built index and functions to explore the document.

DOCUMENT INFO:
- Path: {doc_path}
- Type: {doc_type}
- Total size: {total_chars} characters

TABLE OF CONTENTS:
{toc}

SECTION SUMMARIES:
{summaries}

AVAILABLE FUNCTIONS:
- get_toc() -> dict                              # Full TOC structure
- get_section_names() -> list                    # List of section names
- get_summary(section_name) -> str               # Summary of one section
- read_section(section_name) -> str              # Full section content
- read_section_chunk(section_name, idx) -> str   # Chunk of section
- grep_section(pattern, section_name) -> list    # Search in section
- grep_all(pattern) -> dict                      # Search all sections
- find_sections_by_keyword(keyword) -> list      # Find relevant sections
- llm_query(prompt) -> str                       # Sub-LLM for semantic tasks
- ask_about_section(question, section) -> str    # Ask about specific section

STRATEGY:
1. First, consult TOC and summaries to identify relevant sections
2. Use grep_section() for targeted search within sections
3. Use llm_query() or ask_about_section() for semantic understanding
4. Aggregate findings and return final answer

Write Python code to answer the user's query.
When done, use FINAL(answer) to return your answer.
```

### 5.2 Example Execution

**Query**: "What datasets were used in the experiments?"

```python
# Root LLM generates:

# Step 1: Check summaries for relevant sections
print("Checking summaries...")
summaries = get_all_summaries()
for name, summary in summaries.items():
    if "data" in summary.lower() or "experiment" in summary.lower():
        print(f"  Relevant: {name}")

# Output:
# Relevant: 3. Methodology
# Relevant: 4. Experiments
# Relevant: 4.1 Datasets

# Step 2: Read the specific datasets section
datasets_content = read_section("4.1 Datasets")
print(f"Datasets section length: {len(datasets_content)} chars")

# Step 3: Extract structured info via sub-LLM
datasets = llm_query(f"""
    Extract all datasets mentioned in this section.
    Return as a list with: name, size, description.

    Content:
    {datasets_content}
""")

print(datasets)

# Step 4: Verify by checking methodology section too
methodology_datasets = grep_section(r"dataset|corpus|benchmark", "3. Methodology")
print(f"Also mentioned in methodology: {methodology_datasets}")

FINAL(datasets)
```

---

## 6. Optimizations

### 6.1 Lazy Section Loading

```python
class LazySection:
    """Load section content only when accessed"""

    def __init__(self, doc_path: str, start: int, end: int):
        self.doc_path = doc_path
        self.start = start
        self.end = end
        self._content = None

    @property
    def content(self) -> str:
        if self._content is None:
            with open(self.doc_path, 'r') as f:
                f.seek(self.start)
                self._content = f.read(self.end - self.start)
        return self._content

    def __len__(self):
        return self.end - self.start
```

### 6.2 Section Caching

```python
from functools import lru_cache

@lru_cache(maxsize=10)
def cached_read_section(doc_path: str, start: int, end: int) -> str:
    """Cache recently accessed sections"""
    with open(doc_path, 'r') as f:
        f.seek(start)
        return f.read(end - start)
```

### 6.3 Parallel Sub-LLM Calls

```python
import asyncio

async def parallel_section_analysis(sections: list, question: str, llm) -> dict:
    """Analyze multiple sections in parallel"""

    async def analyze_one(section_name: str, content: str):
        result = await llm.aquery(f"Answer '{question}' from: {content[:10000]}")
        return section_name, result

    tasks = [analyze_one(name, content) for name, content in sections]
    results = await asyncio.gather(*tasks)

    return dict(results)
```

---

## 7. API

```python
from single_doc_rlm import SingleDocRLM

# Initialize
rlm = SingleDocRLM(
    doc_path="/path/to/document.pdf",
    root_model="gpt-5",
    sub_model="gpt-5-mini"
)

# Build index (one-time, can be cached)
rlm.build_index()
rlm.save_index("/path/to/document.index.json")

# Or load existing index
rlm.load_index("/path/to/document.index.json")

# Query
answer = rlm.query("What are the main findings?")

# Interactive exploration
rlm.repl()  # Opens interactive Python REPL
```

---

## 8. Supported Document Types

| Type | TOC Detection | Section Markers |
|------|---------------|-----------------|
| Research Paper | Headings, Abstract/Methods/Results | ##, \section{} |
| Legal Contract | Numbered clauses | 1.1, 1.2, Article I |
| Technical Manual | Chapters, Procedures | Chapter X, Step N |
| Book | Chapters, Parts | Chapter, Part, Act |
| Report | Executive Summary, Sections | Heading levels |
| Markdown | Hash headings | #, ##, ### |
| PDF | Outline/Bookmarks | PDF structure |

---

## 9. Limitations

1. **Structure Detection**: Documents without clear headings harder to index
2. **Cross-References**: "See Section 3.2" requires additional linking
3. **Tables/Figures**: Need special handling for non-text content
4. **Index Staleness**: Index must be rebuilt if document changes

---

## 10. Future Enhancements

- [ ] PDF native support (extract outline, handle figures)
- [ ] Cross-reference resolution (link "see section X" mentions)
- [ ] Table extraction and querying
- [ ] Figure/image description generation
- [ ] Incremental index updates
