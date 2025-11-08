# SPDX-License-Identifier: MPL-2.0
import json
import os
from typing import List, Dict, Any
from src.core.grid import ARCGrid
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SubmissionGenerator:
    """Generates the submission.json file in the required format."""
    def __init__(self, predictor: Any): # Use Any to avoid circular import with ARCPredictor
        self.predictor = predictor

    def generate(self, input_path: str, output_path: str):
        """Reads input challenges, generates predictions, and writes submission.json."""
        logger.info(f"Reading input challenges from {input_path}")
        with open(input_path, 'r') as f:
            challenges = json.load(f)

        submission_data = {}
        for task_id, task_data in challenges.items():
            logger.info(f"Generating predictions for task: {task_id}")
            # The 'train' examples are used to infer the underlying rule.
            # The 'test' examples are what we need to predict outputs for.
            train_input_grid = ARCGrid.from_array(task_data["train"][0]["input"]) # Use first train input as base for rule inference
            test_examples = task_data["test"]

            task_predictions = {}
            for i, test_example in enumerate(test_examples):
                # Each test input needs its own prediction pair (attempt1, attempt2)
                # The predictor's `predict_single_task` needs to be adapted to take a single test input
                # and its corresponding (unknown) output, and potentially the training examples for context.
                # For simplicity, we'll pass the current test input and the *first* test output as a dummy target for search.
                # A robust solution would use the training examples to learn the rule, then apply to test inputs.
                predictions_for_test_input = self.predictor.predict_single_task(
                    task_id,
                    ARCGrid.from_array(test_example["input"]),
                    [test_example] # Pass the single test example as a list
                )
                task_predictions[f"output_{i}"] = predictions_for_test_input

            submission_data[task_id] = task_predictions

        logger.info(f"Writing submission file to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(submission_data, f, indent=4)
        logger.info("Submission file generated successfully.")

    def _format_grid_for_submission(self, grid: ARCGrid) -> List[List[int]]:
        """Ensures grid is in the correct list of lists format."""
        return grid.to_array()
