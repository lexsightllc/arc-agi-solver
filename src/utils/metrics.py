# SPDX-License-Identifier: MPL-2.0
import numpy as np
from typing import List, Dict, Any
from src.core.grid import ARCGrid

def calculate_accuracy(predicted_grid: ARCGrid, target_grid: ARCGrid) -> float:
    """Calculates pixel-wise accuracy between two grids."""
    if predicted_grid.grid.shape != target_grid.grid.shape:
        return 0.0
    return np.mean(predicted_grid.grid == target_grid.grid).item()

def calculate_iou(predicted_grid: ARCGrid, target_grid: ARCGrid, background_color: int = 0) -> float:
    """Calculates Intersection over Union (IoU) for non-background pixels."""
    if predicted_grid.grid.shape != target_grid.grid.shape:
        return 0.0

    pred_mask = (predicted_grid.grid != background_color)
    target_mask = (target_grid.grid != background_color)

    intersection = np.sum(np.logical_and(pred_mask, target_mask))
    union = np.sum(np.logical_or(pred_mask, target_mask))

    if union == 0: # Both are empty or only background
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def evaluate_predictions(predictions: Dict[str, Dict[str, List[List[int]]]], ground_truth_tasks: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluates predictions against ground truth tasks and returns metrics."""
    total_tasks = len(ground_truth_tasks)
    exact_match_count_attempt1 = 0
    exact_match_count_attempt2 = 0
    avg_accuracy_attempt1 = []
    avg_accuracy_attempt2 = []
    avg_iou_attempt1 = []
    avg_iou_attempt2 = []

    for task_id, task_data in ground_truth_tasks.items():
        if task_id not in predictions:
            continue # Skip if task not predicted

        test_inputs = task_data["test"]
        predicted_outputs = predictions[task_id]

        for i, test_example in enumerate(test_inputs):
            if f"output_{i}" not in predicted_outputs:
                continue # Skip if output not predicted

            target_grid = ARCGrid.from_array(test_example["output"])

            # Attempt 1
            pred_attempt1_array = predicted_outputs[f"output_{i}"]["attempt1"]
            pred_attempt1_grid = ARCGrid.from_array(pred_attempt1_array)
            if pred_attempt1_grid == target_grid:
                exact_match_count_attempt1 += 1
            avg_accuracy_attempt1.append(calculate_accuracy(pred_attempt1_grid, target_grid))
            avg_iou_attempt1.append(calculate_iou(pred_attempt1_grid, target_grid))

            # Attempt 2
            pred_attempt2_array = predicted_outputs[f"output_{i}"]["attempt2"]
            pred_attempt2_grid = ARCGrid.from_array(pred_attempt2_array)
            if pred_attempt2_grid == target_grid:
                exact_match_count_attempt2 += 1
            avg_accuracy_attempt2.append(calculate_accuracy(pred_attempt2_grid, target_grid))
            avg_iou_attempt2.append(calculate_iou(pred_attempt2_grid, target_grid))

    num_predictions = len(avg_accuracy_attempt1)
    if num_predictions == 0: # No predictions made
        return {"overall_accuracy": 0.0, "overall_iou": 0.0, "exact_match_rate": 0.0}

    results = {
        "attempt1_exact_match_rate": exact_match_count_attempt1 / num_predictions,
        "attempt1_avg_accuracy": np.mean(avg_accuracy_attempt1).item(),
        "attempt1_avg_iou": np.mean(avg_iou_attempt1).item(),
        "attempt2_exact_match_rate": exact_match_count_attempt2 / num_predictions,
        "attempt2_avg_accuracy": np.mean(avg_accuracy_attempt2).item(),
        "attempt2_avg_iou": np.mean(avg_iou_attempt2).item(),
        "total_evaluated_predictions": num_predictions,
        "total_tasks_in_gt": total_tasks
    }
    return results
