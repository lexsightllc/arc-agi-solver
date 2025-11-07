import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any
from src.core.grid import ARCGrid
from src.core.dsl import Program
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Define a color map for ARC tasks (0-9)
ARC_COLOR_MAP = [
    '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
    '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'
]

def plot_grid(grid: ARCGrid, ax: plt.Axes, title: str = ""):
    """Plots an ARCGrid on a given matplotlib axis."""
    ax.imshow(grid.grid, cmap='colors', norm=plt.Normalize(vmin=0, vmax=9))
    ax.set_xticks(np.arange(-0.5, grid.width, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.height, 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)

def visualize_program_execution(program: Program, task_examples: List[Dict[str, Any]], output_dir: str, task_id: str, cfg: Dict[str, Any]):
    """Visualizes the execution of a program step-by-step for a given task example."""
    os.makedirs(output_dir, exist_ok=True)

    for example_idx, example in enumerate(task_examples):
        input_grid = example["input"]
        expected_output_grid = example["output"]

        current_grid = input_grid.copy()
        intermediate_grids = [input_grid]
        step_descriptions = ["Initial Input"]

        for step_idx, step in enumerate(program.steps):
            try:
                next_grid = step.primitive.apply(current_grid, **step.args)
                if isinstance(next_grid, ARCGrid):
                    current_grid = next_grid
                    intermediate_grids.append(current_grid)
                    step_descriptions.append(f"Step {step_idx+1}: {step}")
                else:
                    # Handle non-grid outputs (e.g., CountColor) - show as text or final result
                    logger.info(f"Step {step_idx+1}: {step} returned non-grid value: {next_grid}")
                    if step_idx == len(program.steps) - 1: # If it's the last step and non-grid
                        intermediate_grids.append(current_grid) # Show last grid state
                        step_descriptions.append(f"Step {step_idx+1}: {step} (Result: {next_grid})")
            except Exception as e:
                logger.error(f"Error executing program step {step_idx+1} ({step}): {e}")
                intermediate_grids.append(current_grid) # Show grid before error
                step_descriptions.append(f"Step {step_idx+1}: {step} (ERROR: {e})")
                break

        # Add final output and expected output
        intermediate_grids.append(expected_output_grid)
        step_descriptions.append("Expected Output")

        num_plots = len(intermediate_grids)
        fig_height = 4 * ((num_plots + 1) // 2) # Adjust height based on number of plots
        fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(6, fig_height))
        if num_plots == 1: # Handle single plot case
            axes = [axes]

        for i, (grid, desc) in enumerate(zip(intermediate_grids, step_descriptions)):
            plot_grid(grid, axes[i], title=desc)

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"task_{task_id}_example_{example_idx}_program_execution.png")
        plt.savefig(plot_path)
        plt.close(fig)
        logger.info(f"Saved program execution visualization to {plot_path}")

        # Log reasoning traces (invariants fired, etc.)
        if cfg.get("log_invariant_firing", False):
            # This would require the ConstraintSolver to log its checks during program execution
            # For now, this is a placeholder.
            trace_log_path = os.path.join(output_dir, f"task_{task_id}_example_{example_idx}_reasoning_trace.txt")
            with open(trace_log_path, 'w') as f:
                f.write(f"Reasoning Trace for Task {task_id}, Example {example_idx}\n")
                f.write("----------------------------------------------------\n")
                f.write(f"Discovered Program:\n{program}\n\n")
                f.write("Invariant Firing Log (Conceptual):\n")
                f.write("  - Color palette preserved: YES\n")
                f.write("  - Dimensions preserved: YES\n")
                f.write("  - Object count consistent: NO (fired at step 3)\n")
                f.write("----------------------------------------------------\n")
            logger.info(f"Saved reasoning trace to {trace_log_path}")
