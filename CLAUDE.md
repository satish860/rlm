# RLM Project - Claude Instructions

## Project Overview

Building an enhanced Recursive Language Model (RLM) implementation that improves upon the original paper by adding structure-aware navigation (TOC, indexes) on top of the code-based exploration approach.

**Paper**: "Recursive Language Models" (Zhang, Kraska, Khattab - Dec 2025)
**Paper PDF**: `recursive_language_models.pdf`

---

## Core Concept

```
Original RLM:  Long Doc -> REPL (flat string) -> grep/slice -> sub-LLM
Our RLM:       Long Doc -> Pre-index (TOC/structure) -> REPL -> scoped search -> sub-LLM
```

**Key Insight**: Documents have structure (TOC, sections, headings). Use it.

---

## Project Structure

```
C:/code/rlm/
    CLAUDE.md                        # This file - project instructions
    recursive_language_models.pdf    # Original paper

    # Specifications
    SPEC_SINGLE_DOC.md              # Single document approach
    SPEC_MULTI_DOC.md               # Multi-document corpus approach
    SPEC_BENCHMARKS.md              # Evaluation benchmarks

    # Source Code (to be implemented)
    src/
        __init__.py
        single_doc/                  # Single document RLM
            __init__.py
            indexer.py              # TOC generation, section mapping
            repl.py                 # REPL environment
            rlm.py                  # Main RLM class
        multi_doc/                   # Multi-document RLM
            __init__.py
            corpus_indexer.py       # Corpus-level indexing
            doc_indexer.py          # Per-document indexing
            repl.py                 # REPL environment
            rlm.py                  # Main RLM class
        llm/
            __init__.py
            base.py                 # LLM interface
            openai.py               # OpenAI implementation
            anthropic.py            # Anthropic implementation
        utils/
            __init__.py
            token_counter.py        # Track token usage
            cache.py                # Caching utilities

    # Benchmarks
    benchmarks/
        download.py                 # Download benchmark datasets
        runner.py                   # Benchmark evaluation harness
        metrics.py                  # Scoring functions

    # Data (downloaded benchmarks)
    data/
        oolong/
        browsecomp/
        longbench/
        ruler/

    # Results
    results/
        experiments.json
        plots/

    # Tests
    tests/
        test_single_doc.py
        test_multi_doc.py
        test_benchmarks.py
```

---

## Implementation Order

### Phase 1: Core Infrastructure
1. [ ] LLM interface (`src/llm/`)
2. [ ] Token counter utility
3. [ ] Basic REPL execution

### Phase 2: Single Document RLM
1. [ ] TOC generation from document
2. [ ] Section mapping (name -> char range)
3. [ ] Section summaries via sub-LLM
4. [ ] REPL environment with scoped functions
5. [ ] Main SingleDocRLM class

### Phase 3: Multi-Document RLM
1. [ ] Corpus indexer (doc summaries, keywords)
2. [ ] Inverted index (keyword -> doc_ids)
3. [ ] On-demand per-doc indexing
4. [ ] REPL environment with corpus functions
5. [ ] Main MultiDocRLM class

### Phase 4: Benchmarks
1. [ ] Download benchmark datasets
2. [ ] Implement Paper RLM baseline
3. [ ] Evaluation harness
4. [ ] Run comparisons

### Phase 5: Optimization
1. [ ] Caching (index, LLM responses)
2. [ ] Parallel sub-LLM calls
3. [ ] Lazy loading

---

## Key Design Decisions

### Single Document
- **TOC Detection**: Use LLM to identify document structure (headings, sections)
- **Section Boundaries**: Map section names to character offsets
- **Lazy Loading**: Read sections on-demand, not upfront
- **Scoped Search**: `grep_section(pattern, section_name)` instead of `grep_all(pattern)`

### Multi-Document
- **Two-Level Index**:
  - Level 1: Corpus index (doc summaries, keywords, entities)
  - Level 2: Per-doc index (TOC, sections) - built on-demand
- **Document Selection**: Use keyword/entity index before reading documents
- **Incremental Updates**: Add/remove docs without full rebuild

### Sub-LLM Strategy
- **Root LLM**: Orchestrates, writes code (e.g., GPT-4, Claude Opus)
- **Sub-LLM**: Semantic tasks, cheaper (e.g., GPT-4-mini, Claude Haiku)
- **Batching**: Group multiple items per sub-LLM call when possible

---

## API Design

### Single Document
```python
from rlm import SingleDocRLM

rlm = SingleDocRLM(
    doc_path="path/to/document.pdf",
    root_model="gpt-4",
    sub_model="gpt-4-mini"
)

# Build index (one-time)
rlm.build_index()

# Query
answer = rlm.query("What are the main findings?")

# Interactive
rlm.repl()
```

### Multi-Document
```python
from rlm import MultiDocRLM

rlm = MultiDocRLM(
    corpus_path="path/to/documents/",
    root_model="gpt-4",
    sub_model="gpt-4-mini"
)

# Build corpus index
rlm.build_index()

# Query
answer = rlm.query("Which documents mention X?")

# Add new document
rlm.add_document("path/to/new_doc.txt")
```

---

## REPL Functions Reference

### Single Document Functions
```python
# Navigation
get_toc() -> dict
get_section_names() -> list
get_summary(section_name) -> str
get_all_summaries() -> dict

# Reading
read_section(section_name) -> str
read_section_chunk(section_name, chunk_idx, chunk_size) -> str
read_range(start, end) -> str

# Search
grep_section(pattern, section_name) -> list
grep_all(pattern) -> dict
find_sections_by_keyword(keyword) -> list

# LLM
llm_query(prompt) -> str
ask_about_section(question, section_name) -> str
```

### Multi-Document Functions
```python
# Corpus Navigation
list_documents(limit) -> list
get_doc_meta(doc_id) -> dict
get_doc_summary(doc_id) -> str
get_all_summaries() -> dict

# Corpus Search
search_by_keyword(keyword) -> list
search_by_entity(entity) -> list
search_by_keywords(keywords, mode) -> list
grep_corpus(pattern, max_docs) -> dict
rank_documents(query, top_k) -> list

# Document Access
read_document(doc_id) -> str
read_document_chunk(doc_id, start, end) -> str
grep_document(doc_id, pattern) -> list

# Deep Document Access (Level 2)
get_doc_toc(doc_id) -> dict
get_doc_sections(doc_id) -> list
read_doc_section(doc_id, section_name) -> str

# LLM
llm_query(prompt) -> str
ask_about_document(question, doc_id) -> str
ask_across_documents(question, doc_ids) -> dict
```

---

## Benchmarks

| Benchmark | Type | GitHub | What It Tests |
|-----------|------|--------|---------------|
| S-NIAH | Single | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | Baseline retrieval |
| OOLONG | Single | [abertsch72/oolong](https://github.com/abertsch72/oolong) | Semantic aggregation |
| OOLONG-Pairs | Single | Paper Appendix | Pairwise aggregation |
| BrowseComp-Plus | Multi | [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) | Multi-doc QA |
| LongBench-v2 | Multi | [THUDM/LongBench](https://github.com/THUDM/LongBench) | Code understanding |

---

## Development Guidelines

### Code Style
- Python 3.10+
- Type hints required
- Docstrings for public functions
- No emojis in code or comments

### Testing
- Test each component independently
- Mock LLM calls for unit tests
- Integration tests with real LLM for benchmarks

### Performance
- Track token usage for all LLM calls
- Cache aggressively (indexes, LLM responses)
- Lazy load whenever possible

### Error Handling
- Graceful degradation if TOC detection fails
- Fallback to flat search if structure not found
- Log all LLM calls for debugging

---

## Environment Variables

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Model Configuration
RLM_ROOT_MODEL=gpt-4
RLM_SUB_MODEL=gpt-4-mini

# Paths
RLM_DATA_DIR=./data
RLM_RESULTS_DIR=./results
RLM_CACHE_DIR=./cache
```

---

## References

- **Paper**: `recursive_language_models.pdf` in this folder
- **Single Doc Spec**: `SPEC_SINGLE_DOC.md`
- **Multi Doc Spec**: `SPEC_MULTI_DOC.md`
- **Benchmarks Spec**: `SPEC_BENCHMARKS.md`

---

## Quick Commands

```bash
# Run single doc RLM on a document
python -m rlm.single_doc --doc path/to/doc.pdf --query "What is X?"

# Run multi doc RLM on a corpus
python -m rlm.multi_doc --corpus path/to/docs/ --query "Find documents about X"

# Download benchmarks
python benchmarks/download.py

# Run benchmark evaluation
python benchmarks/runner.py --method our_rlm --benchmark oolong

# Run all experiments
python benchmarks/runner.py --all
```
