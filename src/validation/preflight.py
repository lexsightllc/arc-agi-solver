import json
from typing import List, Dict, Any
from src.core.grid import ARCGrid
from src.utils.logging import get_logger
from omegaconf import DictConfig

logger = get_logger(__name__)

class PreflightValidator:
    """Validates submission files against schema, task coverage, and grid constraints."""
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.color_alphabet = set(cfg.validation.color_alphabet)
        self.max_grid_dim = cfg.validation.max_grid_dim

    def validate_submission(self, submission_path: str, challenges_path: str) -> bool:
        """Performs all preflight checks on a submission file."""
        logger.info(f"Starting preflight validation for {submission_path} against {challenges_path}")
        try:
            with open(submission_path, 'r') as f:
                submission = json.load(f)
            with open(challenges_path, 'r') as f:
                challenges = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format for submission or challenges file: {e}")
            return False
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return False

        if not self._check_task_coverage(submission, challenges):
            return False
        if not self._check_multi_output_ordering(submission, challenges):
            return False
        if not self._check_grid_constraints(submission):
            return False

        logger.info("All preflight checks passed.")
        return True

    def _check_task_coverage(self, submission: Dict[str, Any], challenges: Dict[str, Any]) -> bool:
        """Confirms one-to-one coverage of task identifiers."""
        submission_task_ids = set(submission.keys())
        challenge_task_ids = set(challenges.keys())

        if submission_task_ids != challenge_task_ids:
            missing_in_submission = challenge_task_ids - submission_task_ids
            extra_in_submission = submission_task_ids - challenge_task_ids
            if missing_in_submission:
                logger.error(f"Missing tasks in submission: {missing_in_submission}")
            if extra_in_submission:
                logger.error(f"Extra tasks in submission: {extra_in_submission}")
            return False
        logger.debug("Task coverage check passed.")
        return True

    def _check_multi_output_ordering(self, submission: Dict[str, Any], challenges: Dict[str, Any]) -> bool:
        """Verifies ordering of multiple test outputs for tasks."""
        for task_id, task_data in challenges.items():
            if task_id not in submission: continue # Already caught by coverage check

            num_test_outputs = len(task_data["test"])
            for i in range(num_test_outputs):
                expected_key = f"output_{i}"
                if expected_key not in submission[task_id]:
                    logger.error(f"Task {task_id}: Missing expected output key '{expected_key}'.")
                    return False
                # Check if there are unexpected keys (e.g., output_0, output_1, output_3 when only 2 outputs)
                # This is implicitly handled by iterating up to num_test_outputs.
            
            # Check for extra output keys beyond expected count
            for key in submission[task_id].keys():
                if key.startswith("output_"):
                    try:
                        idx = int(key.split('_')[1])
                        if idx >= num_test_outputs:
                            logger.error(f"Task {task_id}: Unexpected output key '{key}'. Expected only {num_test_outputs} outputs.")
                            return False
                    except ValueError:
                        logger.error(f"Task {task_id}: Malformed output key '{key}'.")
                        return False

        logger.debug("Multi-output ordering check passed.")
        return True

    def _check_grid_constraints(self, submission: Dict[str, Any]) -> bool:
        """Asserts shape and color alphabet constraints of predicted grids."""
        for task_id, task_predictions in submission.items():
            for output_key, attempts in task_predictions.items():
                for attempt_type in ["attempt1", "attempt2"]:
                    if attempt_type not in attempts: continue

                    grid_array = attempts[attempt_type]
                    if not isinstance(grid_array, list) or not all(isinstance(row, list) for row in grid_array):
                        logger.error(f"Task {task_id}, {output_key}, {attempt_type}: Grid is not a list of lists.")
                        return False

                    if not grid_array: # Empty grid is valid
                        continue

                    height = len(grid_array)
                    width = len(grid_array[0])

                    if not (1 <= height <= self.max_grid_dim and 1 <= width <= self.max_grid_dim):
                        logger.error(f"Task {task_id}, {output_key}, {attempt_type}: Grid dimensions ({height}x{width}) out of bounds (1-{self.max_grid_dim}).")
                        return False

                    for r_idx, row in enumerate(grid_array):
                        if len(row) != width:
                            logger.error(f"Task {task_id}, {output_key}, {attempt_type}: Row {r_idx} has inconsistent width ({len(row)} vs {width}).")
                            return False
                        for pixel in row:
                            if not isinstance(pixel, int) or pixel not in self.color_alphabet:
                                logger.error(f"Task {task_id}, {output_key}, {attempt_type}: Pixel value {pixel} is not a valid color (0-9).")
                                return False
        logger.debug("Grid constraints check passed.")
        return True
