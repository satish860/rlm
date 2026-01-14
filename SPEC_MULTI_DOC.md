# RLM Spec: Multi-Document Approach

## Overview

Processing a corpus of multiple documents (e.g., 1000 articles, email archive, codebase) where each document may itself be large.

---

## 1. Problem

```
Input: Corpus of N documents (N = 10 to 100,000+)
       Each document: 1K - 1M tokens
       Total corpus: 10M - 1B+ tokens
Goal: Answer queries requiring cross-document reasoning
Constraint: LLM context window (~200K tokens)
```

**Examples:**
- 1000 news articles: "What companies announced layoffs in Q4?"
- Email archive: "Find all discussions about Project X budget"
- Research corpus: "Compare methodologies across ML papers from 2024"
- Legal discovery: "Find all documents mentioning the defendant"
- Codebase: "Where is authentication implemented?"

---

## 2. Key Differences from Single Document

| Aspect | Single Document | Multi-Document |
|--------|-----------------|----------------|
| Unit | Sections | Documents |
| First question | "Which section?" | "Which documents?" |
| Structure | Hierarchical TOC | Flat or folder-based |
| Relationships | Sequential/referenced | Independent or linked |
| Index | One TOC | Corpus index + per-doc index |
| Search | Within-doc grep | Cross-corpus search |
| Relevance | Section relevance | Document relevance ranking |

---

## 3. Architecture

```
+--------------------------------+
|         Document Corpus        |
|   /corpus/                     |
|      doc1.txt                  |
|      doc2.txt                  |
|      ...                       |
|      doc1000.txt               |
+--------------------------------+
              |
              v
+--------------------------------+
|      Corpus Pre-processor      |
|      (runs once per corpus)    |
+--------------------------------+
              |
              v
+--------------------------------+
|        Two-Level Index         |
|                                |
|  Level 1: Corpus Index         |
|    - Doc summaries             |
|    - Doc keywords              |
|    - Doc metadata              |
|    - Cross-doc relationships   |
|                                |
|  Level 2: Per-Doc Index        |
|    - TOC (if applicable)       |
|    - Section map               |
|    - Section summaries         |
|    (loaded on-demand)          |
+--------------------------------+
              |
              v
+--------------------------------+
|         REPL Environment       |
+--------------------------------+
              |
              v
+--------------------------------+
|          Root LLM              |
+--------------------------------+
              |
              v
+--------------------------------+
|          Sub-LLM               |
+--------------------------------+
```

---

## 4. Two-Level Index

### 4.1 Level 1: Corpus Index

```python
@dataclass
class CorpusIndex:
    """Top-level index for entire corpus"""

    corpus_path: str
    total_docs: int
    total_chars: int

    # Document registry
    documents: dict  # doc_id -> DocumentMeta

    # Search indexes
    keyword_index: dict      # keyword -> [doc_ids]
    entity_index: dict       # entity -> [doc_ids]
    date_index: dict         # date -> [doc_ids]

    # Clustering/grouping
    clusters: dict           # cluster_id -> [doc_ids]
    topics: dict             # topic -> [doc_ids]

    # Relationships (if applicable)
    links: dict              # doc_id -> [referenced_doc_ids]


@dataclass
class DocumentMeta:
    """Metadata for a single document in corpus"""

    doc_id: str
    file_path: str
    file_size: int
    char_count: int

    # Content summary
    title: str
    summary: str             # 2-3 sentence summary
    keywords: list           # Key terms
    entities: list           # Named entities (people, orgs, places)

    # Metadata
    doc_type: str            # article, email, code, etc.
    date: str                # If applicable
    author: str              # If applicable

    # Index status
    has_detailed_index: bool # Whether Level 2 index exists
    detailed_index_path: str # Path to per-doc index
```

### 4.2 Level 2: Per-Document Index (On-Demand)

```python
@dataclass
class DocumentIndex:
    """Detailed index for a single document (loaded on demand)"""

    doc_id: str
    file_path: str

    # Structure (if applicable)
    has_structure: bool
    toc: dict                # Table of contents
    section_map: dict        # section -> (start, end)
    section_summaries: dict  # section -> summary

    # Content index
    keywords_by_section: dict
    entities_by_section: dict
```

### 4.3 Index Building

```python
class CorpusIndexer:
    """Build two-level index for document corpus"""

    def __init__(self, corpus_path: str, sub_llm):
        self.corpus_path = corpus_path
        self.sub_llm = sub_llm

    def build_corpus_index(self) -> CorpusIndex:
        """Build Level 1 corpus index"""

        documents = {}
        keyword_index = defaultdict(list)
        entity_index = defaultdict(list)

        # Iterate all documents
        for file_path in self._list_documents():
            doc_id = self._generate_doc_id(file_path)

            # Generate document metadata
            meta = self._index_document(file_path, doc_id)
            documents[doc_id] = meta

            # Build inverted indexes
            for keyword in meta.keywords:
                keyword_index[keyword.lower()].append(doc_id)
            for entity in meta.entities:
                entity_index[entity.lower()].append(doc_id)

        return CorpusIndex(
            corpus_path=self.corpus_path,
            total_docs=len(documents),
            total_chars=sum(d.char_count for d in documents.values()),
            documents=documents,
            keyword_index=dict(keyword_index),
            entity_index=dict(entity_index),
            date_index={},
            clusters={},
            topics={},
            links={},
        )

    def _index_document(self, file_path: str, doc_id: str) -> DocumentMeta:
        """Generate metadata for single document"""

        with open(file_path, 'r') as f:
            content = f.read()

        # Use sub-LLM to extract metadata
        extraction = self.sub_llm.query(f"""
            Extract metadata from this document.
            Return JSON with: title, summary (2-3 sentences), keywords (list of 5-10),
            entities (people, organizations, places), doc_type, date (if found), author (if found).

            Document:
            {content[:20000]}
        """)

        meta = json.loads(extraction)

        return DocumentMeta(
            doc_id=doc_id,
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            char_count=len(content),
            title=meta.get("title", "Untitled"),
            summary=meta.get("summary", ""),
            keywords=meta.get("keywords", []),
            entities=meta.get("entities", []),
            doc_type=meta.get("doc_type", "unknown"),
            date=meta.get("date", ""),
            author=meta.get("author", ""),
            has_detailed_index=False,
            detailed_index_path="",
        )

    def build_document_index(self, doc_id: str) -> DocumentIndex:
        """Build Level 2 index for specific document (on-demand)"""

        meta = self.corpus_index.documents[doc_id]

        with open(meta.file_path, 'r') as f:
            content = f.read()

        # Check if document has structure
        has_structure = self._detect_structure(content)

        if has_structure:
            toc = self._generate_toc(content)
            section_map = self._build_section_map(toc, content)
            section_summaries = self._summarize_sections(section_map, content)
        else:
            toc = {}
            section_map = {"full_document": (0, len(content))}
            section_summaries = {"full_document": meta.summary}

        return DocumentIndex(
            doc_id=doc_id,
            file_path=meta.file_path,
            has_structure=has_structure,
            toc=toc,
            section_map=section_map,
            section_summaries=section_summaries,
            keywords_by_section={},
            entities_by_section={},
        )
```

---

## 5. REPL Environment

### 5.1 Corpus-Level Functions

```python
class MultiDocREPL:
    """REPL environment for multi-document corpus"""

    def __init__(self, corpus_index: CorpusIndex, sub_llm):
        self.corpus = corpus_index
        self.sub_llm = sub_llm
        self.loaded_doc_indexes = {}  # Cache for Level 2 indexes

    # === Corpus Navigation ===

    def list_documents(self, limit: int = None) -> list:
        """List all documents in corpus"""
        docs = list(self.corpus.documents.keys())
        return docs[:limit] if limit else docs

    def get_doc_meta(self, doc_id: str) -> dict:
        """Get metadata for a document"""
        meta = self.corpus.documents[doc_id]
        return asdict(meta)

    def get_doc_summary(self, doc_id: str) -> str:
        """Get summary of a document"""
        return self.corpus.documents[doc_id].summary

    def get_all_summaries(self) -> dict:
        """Get summaries of all documents"""
        return {
            doc_id: meta.summary
            for doc_id, meta in self.corpus.documents.items()
        }

    # === Corpus Search ===

    def search_by_keyword(self, keyword: str) -> list:
        """Find documents containing keyword"""
        return self.corpus.keyword_index.get(keyword.lower(), [])

    def search_by_entity(self, entity: str) -> list:
        """Find documents mentioning entity"""
        return self.corpus.entity_index.get(entity.lower(), [])

    def search_by_keywords(self, keywords: list, mode: str = "any") -> list:
        """Find documents matching multiple keywords"""
        doc_sets = [
            set(self.search_by_keyword(kw))
            for kw in keywords
        ]

        if mode == "any":
            return list(set.union(*doc_sets)) if doc_sets else []
        elif mode == "all":
            return list(set.intersection(*doc_sets)) if doc_sets else []

    def grep_corpus(self, pattern: str, max_docs: int = 100) -> dict:
        """Search pattern across all documents"""
        results = {}
        for doc_id, meta in list(self.corpus.documents.items())[:max_docs]:
            with open(meta.file_path, 'r') as f:
                content = f.read()
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                results[doc_id] = matches
        return results

    def rank_documents(self, query: str, top_k: int = 10) -> list:
        """Rank documents by relevance to query"""
        scores = []

        for doc_id, meta in self.corpus.documents.items():
            # Simple keyword matching score
            query_terms = query.lower().split()
            doc_terms = meta.keywords + [meta.title.lower()]
            score = sum(1 for term in query_terms if any(term in dt for dt in doc_terms))
            scores.append((doc_id, score, meta.summary))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # === Document-Level Access ===

    def read_document(self, doc_id: str) -> str:
        """Read full document content"""
        meta = self.corpus.documents[doc_id]
        with open(meta.file_path, 'r') as f:
            return f.read()

    def read_document_chunk(self, doc_id: str, start: int, end: int) -> str:
        """Read chunk of document"""
        meta = self.corpus.documents[doc_id]
        with open(meta.file_path, 'r') as f:
            f.seek(start)
            return f.read(end - start)

    def grep_document(self, doc_id: str, pattern: str) -> list:
        """Search within specific document"""
        content = self.read_document(doc_id)
        return re.findall(pattern, content, re.IGNORECASE)

    # === Deep Document Access (Level 2) ===

    def get_doc_toc(self, doc_id: str) -> dict:
        """Get TOC for specific document (builds index if needed)"""
        index = self._get_or_build_doc_index(doc_id)
        return index.toc

    def get_doc_sections(self, doc_id: str) -> list:
        """Get section names for document"""
        index = self._get_or_build_doc_index(doc_id)
        return list(index.section_map.keys())

    def read_doc_section(self, doc_id: str, section_name: str) -> str:
        """Read specific section of document"""
        index = self._get_or_build_doc_index(doc_id)
        start, end = index.section_map[section_name]
        return self.read_document_chunk(doc_id, start, end)

    def _get_or_build_doc_index(self, doc_id: str) -> DocumentIndex:
        """Lazy load or build Level 2 index"""
        if doc_id not in self.loaded_doc_indexes:
            indexer = CorpusIndexer(self.corpus.corpus_path, self.sub_llm)
            indexer.corpus_index = self.corpus
            self.loaded_doc_indexes[doc_id] = indexer.build_document_index(doc_id)
        return self.loaded_doc_indexes[doc_id]

    # === LLM Functions ===

    def llm_query(self, prompt: str) -> str:
        """Call sub-LLM"""
        return self.sub_llm.query(prompt)

    def ask_about_document(self, question: str, doc_id: str) -> str:
        """Ask question about specific document"""
        content = self.read_document(doc_id)
        return self.llm_query(f"""
            Answer this question based on the document.

            Question: {question}

            Document:
            {content[:50000]}
        """)

    def ask_across_documents(self, question: str, doc_ids: list) -> dict:
        """Ask same question across multiple documents"""
        results = {}
        for doc_id in doc_ids:
            results[doc_id] = self.ask_about_document(question, doc_id)
        return results
```

### 5.2 Environment Setup

```python
def create_multi_doc_env(corpus_index: CorpusIndex, sub_llm) -> dict:
    """Create REPL environment for multi-document corpus"""

    repl = MultiDocREPL(corpus_index, sub_llm)

    return {
        # Corpus metadata
        "corpus_path": corpus_index.corpus_path,
        "total_docs": corpus_index.total_docs,
        "total_chars": corpus_index.total_chars,

        # Document list (preview)
        "doc_ids": list(corpus_index.documents.keys()),
        "doc_summaries": {
            doc_id: meta.summary
            for doc_id, meta in list(corpus_index.documents.items())[:20]
        },

        # Corpus functions
        "list_documents": repl.list_documents,
        "get_doc_meta": repl.get_doc_meta,
        "get_doc_summary": repl.get_doc_summary,
        "get_all_summaries": repl.get_all_summaries,
        "search_by_keyword": repl.search_by_keyword,
        "search_by_entity": repl.search_by_entity,
        "search_by_keywords": repl.search_by_keywords,
        "grep_corpus": repl.grep_corpus,
        "rank_documents": repl.rank_documents,

        # Document functions
        "read_document": repl.read_document,
        "read_document_chunk": repl.read_document_chunk,
        "grep_document": repl.grep_document,

        # Deep document functions
        "get_doc_toc": repl.get_doc_toc,
        "get_doc_sections": repl.get_doc_sections,
        "read_doc_section": repl.read_doc_section,

        # LLM functions
        "llm_query": repl.llm_query,
        "ask_about_document": repl.ask_about_document,
        "ask_across_documents": repl.ask_across_documents,
    }
```

---

## 6. Query Flow

### 6.1 System Prompt for Root LLM

```
You are analyzing a corpus of {total_docs} documents totaling {total_chars} characters.
Each document has metadata and can be explored in detail.

CORPUS INFO:
- Path: {corpus_path}
- Documents: {total_docs}
- Total size: {total_chars} characters

SAMPLE DOCUMENT SUMMARIES (first 20):
{doc_summaries}

AVAILABLE FUNCTIONS:

Corpus-Level:
- list_documents(limit=None) -> list           # List all doc IDs
- get_doc_meta(doc_id) -> dict                 # Full metadata for doc
- get_doc_summary(doc_id) -> str               # Summary of doc
- get_all_summaries() -> dict                  # All summaries
- search_by_keyword(keyword) -> list           # Find docs by keyword
- search_by_entity(entity) -> list             # Find docs by entity
- search_by_keywords(keywords, mode) -> list   # Multi-keyword search
- grep_corpus(pattern, max_docs) -> dict       # Regex across corpus
- rank_documents(query, top_k) -> list         # Rank by relevance

Document-Level:
- read_document(doc_id) -> str                 # Full document
- read_document_chunk(doc_id, start, end)      # Chunk of document
- grep_document(doc_id, pattern) -> list       # Search in document
- get_doc_toc(doc_id) -> dict                  # TOC (if structured)
- get_doc_sections(doc_id) -> list             # Section names
- read_doc_section(doc_id, section) -> str     # Read section

LLM Functions:
- llm_query(prompt) -> str                     # Sub-LLM call
- ask_about_document(question, doc_id) -> str  # Ask about one doc
- ask_across_documents(question, doc_ids)      # Ask across multiple

STRATEGY:
1. First, use corpus-level search to find relevant documents
2. Rank or filter to top candidates
3. Dive into specific documents for details
4. Use sub-LLM for semantic extraction
5. Aggregate findings across documents

Write Python code to answer the user's query.
When done, use FINAL(answer) to return your answer.
```

### 6.2 Example Execution

**Query**: "Which companies announced layoffs in Q4 2024?"

```python
# Root LLM generates:

# Step 1: Find relevant documents by keyword
layoff_docs = search_by_keyword("layoff")
print(f"Found {len(layoff_docs)} docs mentioning layoffs")

# Also search related terms
restructuring_docs = search_by_keyword("restructuring")
workforce_docs = search_by_keyword("workforce reduction")

all_relevant = list(set(layoff_docs + restructuring_docs + workforce_docs))
print(f"Total relevant docs: {len(all_relevant)}")

# Step 2: Filter to Q4 2024
q4_docs = []
for doc_id in all_relevant:
    meta = get_doc_meta(doc_id)
    if meta.get("date", "").startswith("2024-1"):  # Oct, Nov, Dec
        q4_docs.append(doc_id)

print(f"Q4 2024 docs: {len(q4_docs)}")

# Step 3: Extract company names from each relevant document
companies = []
for doc_id in q4_docs[:20]:  # Limit to top 20
    result = ask_about_document(
        "What company announced layoffs? Return just the company name.",
        doc_id
    )
    if result and result != "None":
        companies.append({
            "company": result,
            "doc_id": doc_id,
            "summary": get_doc_summary(doc_id)
        })

print(f"Companies found: {[c['company'] for c in companies]}")

# Step 4: Deduplicate and format
unique_companies = list(set(c["company"] for c in companies))

FINAL({
    "companies": unique_companies,
    "count": len(unique_companies),
    "details": companies
})
```

---

## 7. Optimizations

### 7.1 Parallel Document Processing

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_document_search(doc_ids: list, question: str, repl) -> dict:
    """Process multiple documents in parallel"""

    async def process_one(doc_id):
        answer = repl.ask_about_document(question, doc_id)
        return doc_id, answer

    tasks = [process_one(doc_id) for doc_id in doc_ids]
    results = await asyncio.gather(*tasks)

    return dict(results)
```

### 7.2 Incremental Index Updates

```python
class IncrementalIndexer:
    """Update index when documents are added/removed"""

    def add_document(self, corpus_index: CorpusIndex, file_path: str):
        """Add new document to existing index"""
        doc_id = self._generate_doc_id(file_path)
        meta = self._index_document(file_path, doc_id)

        # Update corpus index
        corpus_index.documents[doc_id] = meta

        # Update inverted indexes
        for keyword in meta.keywords:
            if keyword.lower() not in corpus_index.keyword_index:
                corpus_index.keyword_index[keyword.lower()] = []
            corpus_index.keyword_index[keyword.lower()].append(doc_id)

    def remove_document(self, corpus_index: CorpusIndex, doc_id: str):
        """Remove document from index"""
        meta = corpus_index.documents.pop(doc_id, None)
        if meta:
            # Clean up inverted indexes
            for keyword in meta.keywords:
                if keyword.lower() in corpus_index.keyword_index:
                    corpus_index.keyword_index[keyword.lower()].remove(doc_id)
```

### 7.3 Caching Strategy

```python
class CachedCorpusREPL(MultiDocREPL):
    """REPL with multi-level caching"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.doc_content_cache = LRUCache(maxsize=50)      # Full documents
        self.search_result_cache = LRUCache(maxsize=100)   # Search results
        self.llm_response_cache = LRUCache(maxsize=200)    # LLM responses

    def read_document(self, doc_id: str) -> str:
        if doc_id not in self.doc_content_cache:
            content = super().read_document(doc_id)
            self.doc_content_cache[doc_id] = content
        return self.doc_content_cache[doc_id]

    def search_by_keyword(self, keyword: str) -> list:
        cache_key = f"keyword:{keyword.lower()}"
        if cache_key not in self.search_result_cache:
            result = super().search_by_keyword(keyword)
            self.search_result_cache[cache_key] = result
        return self.search_result_cache[cache_key]
```

---

## 8. API

```python
from multi_doc_rlm import MultiDocRLM

# Initialize
rlm = MultiDocRLM(
    corpus_path="/path/to/documents/",
    root_model="gpt-5",
    sub_model="gpt-5-mini"
)

# Build corpus index (can take time for large corpus)
rlm.build_index(
    parallel=True,           # Parallel processing
    batch_size=100,          # Documents per batch
    save_path="/path/to/corpus.index.json"
)

# Or load existing index
rlm.load_index("/path/to/corpus.index.json")

# Query
answer = rlm.query("Find all merger announcements from 2024")

# Interactive mode
rlm.repl()

# Incremental updates
rlm.add_document("/path/to/new_document.txt")
rlm.remove_document("doc_id_123")
rlm.save_index()  # Persist updates
```

---

## 9. Comparison with BrowseComp Approach (Paper)

| Aspect | Paper (BrowseComp) | Proposed |
|--------|-------------------|----------|
| Indexing | None (grep at runtime) | Pre-built corpus index |
| Document selection | Regex across all | Keyword/entity index lookup |
| First step | grep_corpus() | search_by_keyword() |
| Doc metadata | None | Summary, keywords, entities |
| Hierarchical | No | Yes (corpus -> doc -> section) |
| Caching | None | Multi-level |
| Incremental | Rebuild all | Add/remove individual docs |

---

## 10. Limitations

1. **Index Build Time**: Large corpus takes time to index initially
2. **Storage**: Index adds ~5-10% overhead
3. **Staleness**: Index must be updated when corpus changes
4. **Cold Start**: First query to new document builds Level 2 index

---

## 11. Future Enhancements

- [ ] Vector embeddings for semantic search
- [ ] Document clustering by topic
- [ ] Cross-document relationship detection
- [ ] Streaming index updates
- [ ] Distributed corpus support
- [ ] Multi-modal documents (images, tables)
