"""OOLONG Benchmark Loader.

Downloads and filters OOLONG-synth dataset for the trec_coarse split.
"""

from dataclasses import dataclass
from typing import Optional
import json
from pathlib import Path


@dataclass
class OolongTask:
    """A single OOLONG benchmark task."""
    id: int
    context_len: int
    context: str
    question: str
    answer: str
    answer_type: str
    task_group: str
    task: str
    dataset: str


def load_oolong_synth(
    dataset_name: str = "trec",
    context_len: Optional[int] = None,
    max_tasks: Optional[int] = None,
    cache_dir: str = "./data/oolong",
) -> list[OolongTask]:
    """
    Load OOLONG-synth benchmark tasks.

    Args:
        dataset_name: Filter by dataset (e.g., "trec", "spam"). None for all.
        context_len: Filter by context length. None for all.
        max_tasks: Maximum number of tasks to load.
        cache_dir: Directory to cache downloaded data.

    Returns:
        List of OolongTask objects.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    print(f"Loading OOLONG-synth from HuggingFace...")
    ds = load_dataset("oolongbench/oolong-synth", split="test")

    tasks = []
    for row in ds:
        # Filter by dataset name
        if dataset_name and dataset_name not in row["dataset"]:
            continue

        # Filter by context length
        if context_len and row["context_len"] != context_len:
            continue

        task = OolongTask(
            id=row["id"],
            context_len=row["context_len"],
            context=row["context_window_text"],
            question=row["question"],
            answer=row["answer"],
            answer_type=row["answer_type"],
            task_group=row["task_group"],
            task=row["task"],
            dataset=row["dataset"],
        )
        tasks.append(task)

        if max_tasks and len(tasks) >= max_tasks:
            break

    print(f"Loaded {len(tasks)} tasks (dataset={dataset_name}, context_len={context_len})")
    return tasks


def get_available_datasets(sample_size: int = 1000) -> dict:
    """Get unique dataset names and context lengths in OOLONG-synth."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    ds = load_dataset("oolongbench/oolong-synth", split="test")

    datasets = {}
    for i, row in enumerate(ds):
        if i >= sample_size:
            break
        name = row["dataset"]
        ctx_len = row["context_len"]
        if name not in datasets:
            datasets[name] = set()
        datasets[name].add(ctx_len)

    return {k: sorted(v) for k, v in datasets.items()}


def save_tasks_to_json(tasks: list[OolongTask], path: str) -> None:
    """Save tasks to JSON for offline use."""
    data = [
        {
            "id": t.id,
            "context_len": t.context_len,
            "context": t.context,
            "question": t.question,
            "answer": t.answer,
            "answer_type": t.answer_type,
            "task_group": t.task_group,
            "task": t.task,
            "dataset": t.dataset,
        }
        for t in tasks
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(tasks)} tasks to {path}")


def load_tasks_from_json(path: str) -> list[OolongTask]:
    """Load tasks from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [OolongTask(**d) for d in data]


# =============================================================================
# Scoring (from OOLONG paper)
# =============================================================================

def score_oolong(predicted: str, expected: str, answer_type: str) -> float:
    """
    Score OOLONG answer.

    For numeric: score = 0.75^|y - y_hat|
    For others: exact match (1.0 or 0.0)
    """
    predicted = str(predicted).strip().lower()
    expected = str(expected).strip().lower()

    if answer_type == "NUMERIC":
        try:
            pred_num = float(predicted)
            exp_num = float(expected)
            diff = abs(pred_num - exp_num)
            return 0.75 ** diff
        except (ValueError, TypeError):
            return 0.0
    else:
        # Exact match for LABEL, etc.
        return 1.0 if predicted == expected else 0.0


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "list":
        # List available datasets
        print("Scanning OOLONG-synth for available datasets...")
        datasets = get_available_datasets(sample_size=5000)
        print("\nAvailable datasets and context lengths:")
        for name, lengths in sorted(datasets.items()):
            print(f"  {name}: {lengths}")
    else:
        # Download trec dataset
        print("Downloading OOLONG trec_coarse benchmark...")
        tasks = load_oolong_synth(
            dataset_name="trec",
            context_len=131072,  # 128K - same as paper
            max_tasks=50,
        )

        if tasks:
            save_tasks_to_json(tasks, "./data/oolong/trec_coarse_128k.json")
            print(f"\nSample task:")
            t = tasks[0]
            print(f"  Question: {t.question[:100]}...")
            print(f"  Answer: {t.answer}")
            print(f"  Context length: {len(t.context)} chars")
        else:
            print("No tasks found. Try different filters.")
            print("\nAvailable datasets:")
            datasets = get_available_datasets(sample_size=2000)
            for name, lengths in sorted(datasets.items()):
                print(f"  {name}: {lengths}")
