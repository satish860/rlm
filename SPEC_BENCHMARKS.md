# RLM Benchmarks Specification

## Overview

This document defines the benchmarks used to evaluate our RLM implementation against the paper's approach and other baselines.

---

## 1. Benchmark Sources

| Benchmark | Type | GitHub | HuggingFace | Paper |
|-----------|------|--------|-------------|-------|
| **RULER (S-NIAH)** | Single Doc | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) | - | [arXiv:2404.06654](https://arxiv.org/abs/2404.06654) |
| **OOLONG** | Single Doc | [abertsch72/oolong](https://github.com/abertsch72/oolong) | [oolongbench](https://huggingface.co/oolongbench/datasets) | [arXiv:2511.02817](https://arxiv.org/abs/2511.02817) |
| **OOLONG-Pairs** | Single Doc | Paper Appendix E.1 | - | [arXiv:2512.24601](https://arxiv.org/abs/2512.24601) |
| **BrowseComp-Plus** | Multi Doc | [texttron/BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) | [Tevatron/browsecomp-plus](https://huggingface.co/datasets/Tevatron/browsecomp-plus) | [arXiv:2508.06600](https://arxiv.org/abs/2508.06600) |
| **LongBench-v2 CodeQA** | Multi Doc (Code) | [THUDM/LongBench](https://github.com/THUDM/LongBench) | [THUDM/LongBench-v2](https://huggingface.co/datasets/zai-org/LongBench-v2) | [arXiv:2412.15204](https://arxiv.org/abs/2412.15204) |

---

## 2. Benchmark Details

### 2.1 S-NIAH (from RULER)

**Type**: Single Document - Needle in Haystack

**Task**: Find a specific phrase/number in large unrelated text

**Complexity**: O(1) - constant regardless of input size

**Configuration**:
```
- Tasks: 50 single needle tasks
- Input sizes: 8K, 16K, 33K, 66K, 131K, 262K, 524K, 1M tokens
- Needle types: words, numbers, UUIDs
- Haystack: Paul Graham essays (default)
```

**What it tests**:
- Basic retrieval capability
- Grep-style search effectiveness
- NOT structure-aware navigation (needle location is random)

**Our hypothesis**:
- TOC won't help much (needle is randomly placed)
- Flat grep should work fine
- This is baseline capability test

**Setup**:
```bash
git clone https://github.com/NVIDIA/RULER
cd RULER
pip install -r requirements.txt
```

---

### 2.2 OOLONG

**Type**: Single Document - Semantic Aggregation

**Task**: Examine and transform chunks semantically, then aggregate

**Complexity**: O(N) - linear with input size

**Configuration**:
```
- Tasks: 50 tasks (trec_coarse split)
- Input sizes: 1K to 1M tokens
- Dataset: Questions with semantic labels
- Scoring: 0.75^|y-y_hat| for numerical, exact match for others
```

**What it tests**:
- Semantic classification of each entry
- Aggregation across entire document
- Every line matters (can't skip)

**Our hypothesis**:
- TOC helps identify data structure
- Section-scoped processing reduces errors
- Sub-LLM batching improves efficiency

**Setup**:
```bash
git clone https://github.com/abertsch72/oolong
cd oolong
pip install -r requirements.txt

# Load data
from datasets import load_dataset
dataset = load_dataset("oolongbench/oolong", "trec_coarse")
```

---

### 2.3 OOLONG-Pairs

**Type**: Single Document - Pairwise Aggregation

**Task**: Find all pairs of entries satisfying semantic properties

**Complexity**: O(N^2) - quadratic with input size

**Configuration**:
```
- Tasks: 20 queries (defined in paper Appendix E.1)
- Input sizes: 1K to 1M tokens
- Scoring: F1 over answer pairs
```

**What it tests**:
- Pairwise reasoning across document
- Extreme aggregation requirements
- Sub-LLM call efficiency

**Our hypothesis**:
- Structure helps identify entry boundaries
- Parallel sub-LLM calls critical for performance
- Without recursion, O(N^2) is intractable

**Setup**:
```python
# Queries from paper Appendix E.1
OOLONG_PAIRS_QUERIES = [
    "list all pairs of user IDs where both users have at least one instance with a numeric value or location",
    "list all pairs of user IDs where both users have at least one instance with an entity or human being",
    # ... (20 total queries from paper)
]
```

---

### 2.4 BrowseComp-Plus (1K Documents)

**Type**: Multi-Document - Multi-hop QA

**Task**: Answer questions requiring reasoning over multiple documents

**Complexity**: Constant documents needed, but must find them in 1000

**Configuration**:
```
- Tasks: 150 randomly sampled (paper uses this subset)
- Documents: 1000 per task (from 100K corpus)
- Total tokens: 6M - 11M per task
- Gold + evidence + hard negatives guaranteed in corpus
- Scoring: Exact match accuracy
```

**What it tests**:
- Document retrieval from large corpus
- Multi-hop reasoning across documents
- Corpus-level indexing effectiveness

**Our hypothesis**:
- 2-level index dramatically helps (corpus -> doc -> section)
- Keyword/entity index better than flat grep
- Document summaries enable smart filtering

**Setup**:
```bash
git clone https://github.com/texttron/BrowseComp-Plus
cd BrowseComp-Plus
pip install -r requirements.txt

# Or via HuggingFace
from datasets import load_dataset
dataset = load_dataset("Tevatron/browsecomp-plus")
```

---

### 2.5 LongBench-v2 CodeQA

**Type**: Multi-Document (Code Repository)

**Task**: Answer questions about code repositories

**Complexity**: Fixed files needed, variable repo size

**Configuration**:
```
- Format: Multiple choice
- Context: 23K - 4.2M tokens (code repos)
- Scoring: Accuracy (% correct)
```

**What it tests**:
- Code structure understanding
- File navigation in repositories
- AST-like navigation benefits

**Our hypothesis**:
- File tree (like TOC) helps significantly
- Import/dependency tracking helps
- Code-specific indexing (functions, classes) valuable

**Setup**:
```python
from datasets import load_dataset
dataset = load_dataset("THUDM/LongBench-v2", split="train")

# Filter to CodeQA tasks
codeqa = [x for x in dataset if x["task"] == "codeqa"]
```

---

## 3. Methods to Compare

### 3.1 Baselines

| Method | Description | From |
|--------|-------------|------|
| **Base LLM** | Direct call, stuff context | - |
| **Paper RLM** | REPL + flat grep + sub-calls | Paper |
| **Paper RLM (no sub-calls)** | REPL + flat grep only | Paper ablation |
| **Summary Agent** | Iterative summarization | Paper baseline |
| **CodeAct + BM25** | ReAct + retrieval | Paper baseline |
| **RAG** | Embedding retrieval | Standard baseline |

### 3.2 Our Methods

| Method | Description | Tests |
|--------|-------------|-------|
| **Our RLM (Single Doc)** | TOC + section-scoped + sub-calls | SPEC_SINGLE_DOC.md |
| **Our RLM (Multi Doc)** | 2-level index + sub-calls | SPEC_MULTI_DOC.md |
| **Our RLM (no TOC)** | Ablation without structure | TOC value |
| **Our RLM (no sub-calls)** | Ablation without recursion | Sub-call value |

---

## 4. Evaluation Metrics

### 4.1 Accuracy Metrics

| Benchmark | Metric | Formula |
|-----------|--------|---------|
| S-NIAH | Exact Match | 1 if correct else 0 |
| OOLONG | Custom Score | 0.75^(\|y-y_hat\|) for numbers, EM for others |
| OOLONG-Pairs | F1 Score | 2 * (P * R) / (P + R) |
| BrowseComp-Plus | Exact Match | 1 if correct else 0 |
| LongBench-v2 | Accuracy | % correct multiple choice |

### 4.2 Cost Metrics

| Metric | Unit | How to Measure |
|--------|------|----------------|
| Input Tokens | tokens | Sum of all LLM input tokens |
| Output Tokens | tokens | Sum of all LLM output tokens |
| API Cost | USD | Based on model pricing |
| Sub-LLM Calls | count | Number of recursive calls |
| Latency | seconds | Wall clock time |

### 4.3 Efficiency Metrics

| Metric | Formula |
|--------|---------|
| Cost per Correct Answer | Total Cost / Correct Answers |
| Tokens per Correct Answer | Total Tokens / Correct Answers |
| Sub-calls per Task | Avg sub-LLM calls per query |

---

## 5. Experiment Design

### 5.1 Single Document Experiments

**Benchmarks**: S-NIAH, OOLONG, OOLONG-Pairs

**Input Size Scaling**:
```
Sizes: [8K, 16K, 33K, 66K, 131K, 262K, 524K, 1M] tokens
```

**Comparisons**:
```
1. Base LLM vs Paper RLM vs Our RLM
2. Paper RLM (no sub-calls) vs Our RLM (no TOC) vs Our RLM (no sub-calls)
3. Scaling curves: Accuracy vs Input Size
4. Scaling curves: Cost vs Input Size
```

**Key Questions**:
- Does TOC-based navigation improve accuracy?
- Does section-scoped search reduce cost?
- At what input size does our approach win?

### 5.2 Multi-Document Experiments

**Benchmarks**: BrowseComp-Plus, LongBench-v2 CodeQA

**Corpus Size Scaling** (for BrowseComp-Plus):
```
Documents: [10, 50, 100, 500, 1000] docs per task
```

**Comparisons**:
```
1. Base LLM vs Paper RLM vs Our RLM
2. Flat grep vs Corpus index lookup
3. No per-doc index vs On-demand per-doc index
4. Scaling curves: Accuracy vs Corpus Size
5. Scaling curves: Cost vs Corpus Size
```

**Key Questions**:
- Does 2-level indexing improve document retrieval?
- Does corpus index reduce unnecessary document reads?
- How much does on-demand Level 2 indexing cost?

---

## 6. Implementation Plan

### Phase 1: Setup Benchmarks

```bash
# Create benchmark directory
mkdir -p benchmarks/{ruler,oolong,browsecomp,longbench}

# Clone repositories
git clone https://github.com/NVIDIA/RULER benchmarks/ruler
git clone https://github.com/abertsch72/oolong benchmarks/oolong
git clone https://github.com/texttron/BrowseComp-Plus benchmarks/browsecomp
git clone https://github.com/THUDM/LongBench benchmarks/longbench

# Install dependencies
pip install datasets transformers torch
```

### Phase 2: Data Preparation

```python
# download_benchmarks.py

from datasets import load_dataset
import json
import os

def download_oolong():
    """Download OOLONG benchmark"""
    dataset = load_dataset("oolongbench/oolong", "trec_coarse")
    os.makedirs("data/oolong", exist_ok=True)
    dataset.save_to_disk("data/oolong/trec_coarse")

def download_browsecomp():
    """Download BrowseComp-Plus"""
    dataset = load_dataset("Tevatron/browsecomp-plus")
    os.makedirs("data/browsecomp", exist_ok=True)
    dataset.save_to_disk("data/browsecomp")

def download_longbench():
    """Download LongBench-v2"""
    dataset = load_dataset("THUDM/LongBench-v2")
    os.makedirs("data/longbench", exist_ok=True)
    dataset.save_to_disk("data/longbench")

if __name__ == "__main__":
    download_oolong()
    download_browsecomp()
    download_longbench()
```

### Phase 3: Evaluation Harness

```python
# evaluate.py

class BenchmarkRunner:
    """Run benchmarks and collect metrics"""

    def __init__(self, method, benchmark, config):
        self.method = method
        self.benchmark = benchmark
        self.config = config
        self.results = []

    def run(self, tasks):
        for task in tasks:
            # Track metrics
            start_time = time.time()
            token_counter = TokenCounter()

            # Run method
            answer = self.method.query(
                task["query"],
                task["context"],
                token_counter=token_counter
            )

            # Evaluate
            correct = self.evaluate(answer, task["answer"])

            self.results.append({
                "task_id": task["id"],
                "correct": correct,
                "latency": time.time() - start_time,
                "input_tokens": token_counter.input,
                "output_tokens": token_counter.output,
                "sub_calls": token_counter.sub_calls,
                "cost": token_counter.cost,
            })

        return self.aggregate_results()

    def evaluate(self, prediction, ground_truth):
        """Benchmark-specific evaluation"""
        if self.benchmark == "oolong":
            return oolong_score(prediction, ground_truth)
        elif self.benchmark == "oolong_pairs":
            return f1_score(prediction, ground_truth)
        else:
            return exact_match(prediction, ground_truth)

    def aggregate_results(self):
        return {
            "accuracy": mean([r["correct"] for r in self.results]),
            "avg_latency": mean([r["latency"] for r in self.results]),
            "avg_cost": mean([r["cost"] for r in self.results]),
            "total_cost": sum([r["cost"] for r in self.results]),
            "avg_sub_calls": mean([r["sub_calls"] for r in self.results]),
        }
```

### Phase 4: Run Experiments

```python
# run_experiments.py

METHODS = {
    "base_llm": BaseLLM(model="gpt-4"),
    "paper_rlm": PaperRLM(root="gpt-4", sub="gpt-4-mini"),
    "our_rlm_single": OurRLMSingleDoc(root="gpt-4", sub="gpt-4-mini"),
    "our_rlm_multi": OurRLMMultiDoc(root="gpt-4", sub="gpt-4-mini"),
}

BENCHMARKS = {
    "s_niah": load_sniah_tasks(),
    "oolong": load_oolong_tasks(),
    "oolong_pairs": load_oolong_pairs_tasks(),
    "browsecomp": load_browsecomp_tasks(),
    "codeqa": load_codeqa_tasks(),
}

def run_all_experiments():
    results = {}

    for method_name, method in METHODS.items():
        results[method_name] = {}

        for bench_name, tasks in BENCHMARKS.items():
            runner = BenchmarkRunner(method, bench_name, {})
            results[method_name][bench_name] = runner.run(tasks)

    return results

if __name__ == "__main__":
    results = run_all_experiments()
    with open("results/all_experiments.json", "w") as f:
        json.dump(results, f, indent=2)
```

---

## 7. Expected Results

### 7.1 Hypothesis Matrix

| Benchmark | Paper RLM | Our RLM | Why |
|-----------|-----------|---------|-----|
| S-NIAH | Good | Similar | TOC doesn't help random needle |
| OOLONG | Good | Better | Section-scoped reduces errors |
| OOLONG-Pairs | Moderate | Better | Structured iteration helps |
| BrowseComp-Plus | Good | Better | 2-level index faster retrieval |
| CodeQA | Moderate | Better | File tree navigation helps |

### 7.2 Cost Hypothesis

| Benchmark | Paper RLM Cost | Our RLM Cost | Why |
|-----------|----------------|--------------|-----|
| S-NIAH | Low | Similar | Both grep effectively |
| OOLONG | Moderate | Lower | Scoped search = fewer tokens |
| OOLONG-Pairs | High | Lower | Better batching |
| BrowseComp-Plus | High | Lower | Index avoids full corpus scan |
| CodeQA | Moderate | Lower | File tree avoids reading all |

---

## 8. Success Criteria

### Accuracy
- [ ] Match or exceed paper's accuracy on all benchmarks
- [ ] Show improvement on structure-heavy tasks (OOLONG, CodeQA)
- [ ] Maintain performance on flat tasks (S-NIAH)

### Cost
- [ ] Lower average cost per query on multi-doc tasks
- [ ] Lower token usage on section-scoped tasks
- [ ] Acceptable overhead for index building

### Scalability
- [ ] Better scaling curve (accuracy vs input size)
- [ ] Handle 10M+ token corpora efficiently
- [ ] Support incremental index updates

---

## 9. Appendix: OOLONG-Pairs Queries

From paper Appendix E.1 (20 queries):

```python
OOLONG_PAIRS_QUERIES = [
    # Task 1
    "list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a numeric value or location",

    # Task 2
    "list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or human being",

    # Task 3
    "list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a description and abstract concept or abbreviation",

    # Task 4
    "list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with a human being or location, and all instances that are a human being for both users must be after January 6, 2023",

    # Task 5
    "list all pairs of user IDs (no duplicate pairs, list lower ID first) where both users have at least one instance with an entity or numeric value, and all instances that are an entity for both users must be before March 15, 2023",

    # Task 6-20: See paper Appendix E.1 for full list
    # ...
]
```

---

## 10. References

1. **RULER**: Hsieh et al., "RULER: What's the Real Context Size of Your Long-Context Language Models?", COLM 2024
   - Paper: https://arxiv.org/abs/2404.06654
   - Code: https://github.com/NVIDIA/RULER

2. **OOLONG**: Bertsch et al., "Oolong: Evaluating Long Context Reasoning and Aggregation Capabilities", 2025
   - Paper: https://arxiv.org/abs/2511.02817
   - Code: https://github.com/abertsch72/oolong

3. **BrowseComp-Plus**: Chen et al., "BrowseComp-Plus: A More Fair and Transparent Evaluation Benchmark of Deep-Research Agent", 2025
   - Paper: https://arxiv.org/abs/2508.06600
   - Code: https://github.com/texttron/BrowseComp-Plus

4. **LongBench-v2**: Bai et al., "LongBench v2: Towards Deeper Understanding and Reasoning on Realistic Long-context Multitasks", 2024
   - Paper: https://arxiv.org/abs/2412.15204
   - Code: https://github.com/THUDM/LongBench

5. **RLM Paper**: Zhang et al., "Recursive Language Models", 2025
   - Paper: https://arxiv.org/abs/2512.24601
