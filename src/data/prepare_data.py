# SPDX-License-Identifier: MPL-2.0
import argparse
import json
import logging
import os
from typing import Any, Dict, Tuple

logger = logging.getLogger(__name__)

def load_arc_dataset(path: str) -> Dict[str, Any]:
    """Loads ARC dataset from a given directory.

    If the directory does not exist (e.g. optional competition data on Kaggle),
    an empty dataset is returned so callers can safely proceed.
    """

    if not os.path.isdir(path):
        logger.warning("ARC dataset path '%s' does not exist. Returning empty dataset.", path)
        return {}

    dataset = {}
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            task_id = filename.replace(".json", "")
            filepath = os.path.join(path, filename)
            with open(filepath, 'r') as f:
                dataset[task_id] = json.load(f)
    return dataset

def split_dataset(dataset: Dict[str, Any], train_ratio: float = 0.8) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Splits the dataset into training and evaluation sets."""
    task_ids = list(dataset.keys())
    num_train = int(len(task_ids) * train_ratio)
    train_ids = sorted(task_ids[:num_train])
    eval_ids = sorted(task_ids[num_train:])

    train_set = {tid: dataset[tid] for tid in train_ids}
    eval_set = {tid: dataset[tid] for tid in eval_ids}
    return train_set, eval_set

def main():
    parser = argparse.ArgumentParser(description="Prepare ARC dataset for training and evaluation.")
    parser.add_argument("--raw_path", type=str, default="data/raw/arc-dataset", help="Path to the raw ARC dataset directory.")
    parser.add_argument("--processed_path", type=str, default="data/processed", help="Path to save processed data.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of tasks to use for training.")
    args = parser.parse_args()

    os.makedirs(args.processed_path, exist_ok=True)

    logger.info(f"Loading raw ARC dataset from {args.raw_path}")
    full_dataset = load_arc_dataset(args.raw_path)
    logger.info(f"Loaded {len(full_dataset)} tasks.")

    train_set, eval_set = split_dataset(full_dataset, args.train_ratio)
    logger.info(f"Split into {len(train_set)} training tasks and {len(eval_set)} evaluation tasks.")

    train_output_path = os.path.join(args.processed_path, "train_tasks.json")
    eval_output_path = os.path.join(args.processed_path, "eval_tasks.json")

    with open(train_output_path, 'w') as f:
        json.dump(train_set, f, indent=4)
    logger.info(f"Training tasks saved to {train_output_path}")

    with open(eval_output_path, 'w') as f:
        json.dump(eval_set, f, indent=4)
    logger.info(f"Evaluation tasks saved to {eval_output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
