#!/usr/bin/env python3
"""End-to-end Kaggle-style execution script for the ARC AGI Solver project.

This script orchestrates training, evaluation, prediction, validation, and
interpretability steps using the repository's Typer CLI. It is tailored for the
Kaggle runtime environment where the ARC datasets and this repository are
mounted read-only under `/kaggle/input` and writable space is available at
`/kaggle/working`.

The pipeline performs the following high-level steps:

1.  Copy the repository snapshot from `/kaggle/input/arc-agi-solver` to the
    writable working directory so that training artifacts can be saved.
2.  Prepare lightweight task bundles from the official ARC dataset located at
    `/kaggle/input/arc-prize-2025` to keep the demo fast while still exercising
    the code paths.
3.  Run the Typer CLI commands `train`, `evaluate`, `predict`, `validate`, and
    `interpret` with conservative overrides that keep compute requirements low.
4.  Persist all intermediate and final artifacts to `/kaggle/working`.

The script is defensive and can also be executed outside Kaggle (e.g., during
local development) where the datasets might live in different locations. The
default paths can be overridden via command-line flags.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Constants describing the default Kaggle layout.
# ---------------------------------------------------------------------------
DEFAULT_ARC_DATASET = Path("/kaggle/input/arc-prize-2025")
DEFAULT_REPO_DATASET = Path("/kaggle/input/arc-agi-solver")
DEFAULT_WORKING_DIR = Path("/kaggle/working")

# Lightweight overrides so that the heavy research code can run quickly within a
# Kaggle notebook or during automated testing.
DEFAULT_HYDRA_OVERRIDES: Tuple[str, ...] = (
    "device=cpu",  # Use CPU to avoid GPU-specific dependencies.
    "training.trainer.epochs=1",
    "training.trainer.batch_size=2",
    "training.curriculum.stages[0].num_tasks_per_epoch=2",
    "training.curriculum.stages[0].epochs=1",
    "compute_budget.inference_time_per_task_ms=1000",
    "compute_budget.node_expansion_budget=128",
)

# ---------------------------------------------------------------------------
# Helper utilities.
# ---------------------------------------------------------------------------

def debug(msg: str) -> None:
    """Lightweight stdout logger that flushes immediately."""
    print(f"[arc-agi-solver][kaggle] {msg}")
    sys.stdout.flush()


def run_command(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    """Execute a subprocess command, streaming stdout/stderr to the notebook."""
    debug(f"Running command: {' '.join(cmd)} (cwd={cwd})")
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command {' '.join(cmd)} failed with exit code {result.returncode}"
        )


def prepare_repo_copy(source_repo: Path, working_dir: Path) -> Path:
    """Ensure a writable copy of the repository exists in the working directory."""
    if not source_repo.exists():
        # Assume the script is already running from a writable clone.
        debug(
            "Source repository under /kaggle/input not found; using current directory."
        )
        return Path(__file__).resolve().parent

    repo_destination = working_dir / source_repo.name
    if repo_destination.exists():
        debug(f"Writable repository already present at {repo_destination}")
        return repo_destination

    debug(f"Copying repository from {source_repo} to {repo_destination}")
    shutil.copytree(source_repo, repo_destination)
    return repo_destination


def collect_arc_tasks(
    dataset_root: Path, split: str, limit: int, destination: Path
) -> Tuple[Path, List[str]]:
    """Aggregate a few ARC tasks into a single JSON file for fast demos."""
    split_dir = dataset_root / split
    if not split_dir.exists():
        raise FileNotFoundError(f"ARC dataset split '{split}' missing at {split_dir}")

    tasks: Dict[str, Dict] = {}
    for json_path in sorted(split_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as handle:
            tasks[json_path.stem] = json.load(handle)
        if len(tasks) >= limit:
            break

    if not tasks:
        raise RuntimeError(
            f"No tasks were collected from {split_dir}. Check the dataset contents."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(tasks, handle)

    debug(
        f"Wrote {len(tasks)} '{split}' tasks to {destination} for downstream stages."
    )
    return destination, list(tasks.keys())


def ensure_dummy_program(destination: Path) -> Path:
    """Create a simple paint program used by the interpret command."""
    if destination.exists():
        return destination

    program = [
        {
            "name": "paint_rectangle",
            "args": {
                "r_start": 0,
                "c_start": 0,
                "r_end": 2,
                "c_end": 2,
                "color": 1,
            },
        }
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(program, handle, indent=2)
    debug(f"Dummy interpretability program saved to {destination}")
    return destination


def build_overrides(extra_overrides: Iterable[str] | None = None) -> List[str]:
    """Combine default Hydra overrides with any user-provided ones."""
    overrides = list(DEFAULT_HYDRA_OVERRIDES)
    if extra_overrides:
        overrides.extend(extra_overrides)
    return overrides


def append_pythonpath(repo_root: Path) -> None:
    """Ensure the repository (and its src directory) are importable."""
    src_dir = repo_root / "src"
    for path in (repo_root, src_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
            debug(f"Added {path_str} to PYTHONPATH")


# ---------------------------------------------------------------------------
# Main orchestration logic.
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ARC AGI Solver pipeline on Kaggle.")
    parser.add_argument(
        "--arc-dataset",
        type=Path,
        default=DEFAULT_ARC_DATASET,
        help="Root directory of the ARC dataset (contains training/evaluation/test folders).",
    )
    parser.add_argument(
        "--repo-dataset",
        type=Path,
        default=DEFAULT_REPO_DATASET,
        help="Read-only snapshot of this repository (as provided by Kaggle datasets).",
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=DEFAULT_WORKING_DIR,
        help="Writable directory where all artifacts should be stored.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        dest="extra_overrides",
        default=None,
        help="Additional Hydra overrides to pass to every Typer CLI command.",
    )
    parser.add_argument(
        "--max-train-tasks",
        type=int,
        default=3,
        help="Number of ARC tasks to bundle for evaluation/prediction demos.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip the training stage and reuse an existing checkpoint.",
    )

    args = parser.parse_args()

    working_dir = args.working_dir
    working_dir.mkdir(parents=True, exist_ok=True)
    repo_root = prepare_repo_copy(args.repo_dataset, working_dir)
    append_pythonpath(repo_root)

    # Prepare datasets for evaluate/predict/validate stages.
    sample_eval_path, sample_eval_task_ids = collect_arc_tasks(
        args.arc_dataset,
        "training",
        args.max_train_tasks,
        working_dir / "artifacts" / "sample_eval.json",
    )
    submission_input_path, _ = collect_arc_tasks(
        args.arc_dataset,
        "evaluation",
        args.max_train_tasks,
        working_dir / "artifacts" / "sample_submission_input.json",
    )
    dummy_program_path = ensure_dummy_program(working_dir / "artifacts" / "dummy_program.json")

    mlflow_dir = working_dir / "mlruns"
    env = os.environ.copy()
    env.setdefault("MLFLOW_TRACKING_URI", f"file:{mlflow_dir}")
    env.setdefault("HYDRA_FULL_ERROR", "1")

    path_specific_overrides: List[str] = [
        f"output_dir={working_dir / 'outputs'}",
        f"log_dir={working_dir / 'logs'}",
        f"mlflow.tracking_uri=file:{mlflow_dir}",
    ]
    if args.extra_overrides:
        path_specific_overrides.extend(args.extra_overrides)

    overrides = build_overrides(path_specific_overrides)

    def command_with_overrides(base: List[str]) -> List[str]:
        cmd = base.copy()
        for override in overrides:
            cmd.extend(["--override", override])
        return cmd

    outputs_dir = working_dir / "outputs"
    model_checkpoint = outputs_dir / "checkpoint_epoch_1.pt"

    if not args.skip_train:
        train_cmd = command_with_overrides([
            sys.executable,
            "src/main.py",
            "train",
            "--config-path",
            "config/config.yaml",
        ])
        run_command(train_cmd, cwd=repo_root, env=env)
    else:
        debug("Skipping training stage as requested.")

    if not model_checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {model_checkpoint}. "
            "Ensure training completed successfully or provide a checkpoint."
        )

    evaluate_cmd = command_with_overrides([
        sys.executable,
        "src/main.py",
        "evaluate",
        "--model-path",
        str(model_checkpoint),
        "--dataset-path",
        str(sample_eval_path),
        "--config-path",
        "config/config.yaml",
    ])
    run_command(evaluate_cmd, cwd=repo_root, env=env)

    submission_output_path = working_dir / "submission.json"
    predict_cmd = command_with_overrides([
        sys.executable,
        "src/main.py",
        "predict",
        "--model-path",
        str(model_checkpoint),
        "--input-path",
        str(submission_input_path),
        "--output-path",
        str(submission_output_path),
        "--config-path",
        "config/config.yaml",
    ])
    run_command(predict_cmd, cwd=repo_root, env=env)

    validate_cmd = command_with_overrides([
        sys.executable,
        "src/main.py",
        "validate",
        "--submission-path",
        str(submission_output_path),
        "--challenges-path",
        str(submission_input_path),
        "--config-path",
        "config/config.yaml",
    ])
    run_command(validate_cmd, cwd=repo_root, env=env)

    interpret_task_id = sample_eval_task_ids[0] if sample_eval_task_ids else "sample_task"
    interpret_cmd = command_with_overrides([
        sys.executable,
        "src/main.py",
        "interpret",
        "--task-id",
        interpret_task_id,
        "--program-path",
        str(dummy_program_path),
        "--output-dir",
        str(working_dir / "interpret"),
        "--config-path",
        "config/config.yaml",
    ])
    run_command(interpret_cmd, cwd=repo_root, env=env)

    debug("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
