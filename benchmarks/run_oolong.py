"""OOLONG Benchmark Runner.

Runs our structure-aware SingleDocRLM against OOLONG benchmark
and compares with paper baselines.

Paper Baselines (Table 1):
- GPT-5 Base: 44.00%
- RLM (GPT-5): 56.50%
- Qwen3-Coder Base: 36.00%
- RLM (Qwen3-Coder): 48.00%
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv()

from benchmarks.oolong_loader import (
    OolongTask,
    load_oolong_synth,
    load_tasks_from_json,
    save_tasks_to_json,
    score_oolong,
)
from src.single_doc import SingleDocRLM
from src.config import ROOT_MODEL, SUB_MODEL


# =============================================================================
# Paper Baselines
# =============================================================================

PAPER_BASELINES = {
    "GPT-5 Base": 44.00,
    "RLM (GPT-5)": 56.50,
    "Qwen3-Coder Base": 36.00,
    "RLM (Qwen3-Coder)": 48.00,
}


# =============================================================================
# Benchmark Runner
# =============================================================================

def run_single_task(
    task: OolongTask,
    root_model: str = ROOT_MODEL,
    sub_model: str = SUB_MODEL,
    verbose: bool = False,
    skip_summaries: bool = False,
) -> dict:
    """
    Run a single OOLONG task through our RLM.

    Args:
        task: OolongTask to run
        root_model: Root model for code generation
        sub_model: Sub model for segmentation
        verbose: Print execution details
        skip_summaries: Skip summary generation (faster)

    Returns:
        dict with task_id, predicted, expected, score, time_s
    """
    start_time = time.time()

    try:
        # Create RLM from task context
        if verbose:
            print(f"\n[Task {task.id}] Creating RLM from {len(task.context):,} chars...")

        rlm = SingleDocRLM.from_text(
            text=task.context,
            source_name=f"oolong_task_{task.id}",
            root_model=root_model,
            sub_model=sub_model,
            generate_summaries=not skip_summaries,
        )

        # Query the document
        if verbose:
            print(f"[Task {task.id}] Question: {task.question[:100]}...")

        predicted = rlm.query(task.question, verbose=verbose)

        # Clean up prediction
        predicted_clean = predicted.strip()

        # Score
        score = score_oolong(predicted_clean, task.answer, task.answer_type)

        elapsed = time.time() - start_time

        if verbose:
            print(f"[Task {task.id}] Predicted: {predicted_clean[:100]}")
            print(f"[Task {task.id}] Expected: {task.answer}")
            print(f"[Task {task.id}] Score: {score:.3f} (time: {elapsed:.1f}s)")

        return {
            "task_id": task.id,
            "question": task.question,
            "predicted": predicted_clean,
            "expected": task.answer,
            "answer_type": task.answer_type,
            "score": score,
            "time_s": elapsed,
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        if verbose:
            print(f"[Task {task.id}] ERROR: {error_msg}")

        return {
            "task_id": task.id,
            "question": task.question,
            "predicted": "",
            "expected": task.answer,
            "answer_type": task.answer_type,
            "score": 0.0,
            "time_s": elapsed,
            "error": error_msg,
        }


def run_benchmark(
    tasks: list[OolongTask],
    root_model: str = ROOT_MODEL,
    sub_model: str = SUB_MODEL,
    verbose: bool = False,
    skip_summaries: bool = False,
    output_path: Optional[str] = None,
) -> dict:
    """
    Run OOLONG benchmark on a list of tasks.

    Args:
        tasks: List of OolongTask objects
        root_model: Root model for code generation
        sub_model: Sub model for segmentation
        verbose: Print execution details
        skip_summaries: Skip summary generation
        output_path: Path to save results JSON

    Returns:
        dict with overall score, per-task results, timing
    """
    print(f"\n{'='*60}")
    print(f"OOLONG Benchmark Runner")
    print(f"{'='*60}")
    print(f"Tasks: {len(tasks)}")
    print(f"Root Model: {root_model}")
    print(f"Sub Model: {sub_model}")
    print(f"Skip Summaries: {skip_summaries}")
    print(f"{'='*60}\n")

    results = []
    total_time = 0

    for i, task in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Running task {task.id}...")

        result = run_single_task(
            task=task,
            root_model=root_model,
            sub_model=sub_model,
            verbose=verbose,
            skip_summaries=skip_summaries,
        )
        results.append(result)
        total_time += result["time_s"]

        # Progress update
        scores_so_far = [r["score"] for r in results]
        avg_score = sum(scores_so_far) / len(scores_so_far) * 100
        print(f"  Score: {result['score']:.3f} | Running avg: {avg_score:.1f}%")

    # Calculate overall metrics
    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores) * 100
    errors = [r for r in results if r["error"]]

    # Build summary
    summary = {
        "benchmark": "OOLONG-synth",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "root_model": root_model,
            "sub_model": sub_model,
            "skip_summaries": skip_summaries,
            "num_tasks": len(tasks),
        },
        "results": {
            "avg_score_percent": avg_score,
            "total_correct": sum(1 for s in scores if s == 1.0),
            "total_partial": sum(1 for s in scores if 0 < s < 1.0),
            "total_wrong": sum(1 for s in scores if s == 0.0),
            "total_errors": len(errors),
        },
        "timing": {
            "total_time_s": total_time,
            "avg_time_per_task_s": total_time / len(tasks) if tasks else 0,
        },
        "per_task": results,
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Average Score: {avg_score:.2f}%")
    print(f"Correct (1.0): {summary['results']['total_correct']}/{len(tasks)}")
    print(f"Partial (0-1): {summary['results']['total_partial']}/{len(tasks)}")
    print(f"Wrong (0.0): {summary['results']['total_wrong']}/{len(tasks)}")
    print(f"Errors: {summary['results']['total_errors']}/{len(tasks)}")
    print(f"Total Time: {total_time:.1f}s ({total_time/len(tasks):.1f}s avg)")

    # Compare with paper baselines
    print(f"\n{'='*60}")
    print(f"COMPARISON WITH PAPER BASELINES")
    print(f"{'='*60}")
    for method, baseline in PAPER_BASELINES.items():
        diff = avg_score - baseline
        sign = "+" if diff > 0 else ""
        print(f"  {method}: {baseline:.1f}% -> Our RLM: {avg_score:.1f}% ({sign}{diff:.1f}%)")

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    return summary


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run OOLONG benchmark with our structure-aware RLM"
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=50,
        help="Maximum number of tasks to run (default: 50)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="trec",
        help="Dataset filter (default: trec)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=None,
        help="Filter by context length (default: any)",
    )
    parser.add_argument(
        "--root-model",
        type=str,
        default=ROOT_MODEL,
        help=f"Root model for code generation (default: {ROOT_MODEL})",
    )
    parser.add_argument(
        "--sub-model",
        type=str,
        default=SUB_MODEL,
        help=f"Sub model for segmentation (default: {SUB_MODEL})",
    )
    parser.add_argument(
        "--skip-summaries",
        action="store_true",
        help="Skip summary generation (faster but less accurate)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed execution trace",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON",
    )
    parser.add_argument(
        "--from-json",
        type=str,
        default=None,
        help="Load tasks from JSON file instead of HuggingFace",
    )

    args = parser.parse_args()

    # Load tasks
    if args.from_json:
        print(f"Loading tasks from {args.from_json}...")
        tasks = load_tasks_from_json(args.from_json)
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
    else:
        tasks = load_oolong_synth(
            dataset_name=args.dataset,
            context_len=args.context_len,
            max_tasks=args.max_tasks,
        )

    if not tasks:
        print("No tasks loaded. Check filters or network connection.")
        return 1

    # Set default output path
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"./results/oolong_{timestamp}.json"

    # Run benchmark
    run_benchmark(
        tasks=tasks,
        root_model=args.root_model,
        sub_model=args.sub_model,
        verbose=args.verbose,
        skip_summaries=args.skip_summaries,
        output_path=args.output,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
