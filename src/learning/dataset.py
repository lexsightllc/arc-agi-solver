# SPDX-License-Identifier: MPL-2.0
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from typing import List, Dict, Tuple, Any
from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Primitive
from src.utils.logging import get_logger

logger = get_logger(__name__)

class SyntheticARCProblemGenerator:
    """Generates synthetic ARC-like problems by composing DSL primitives."""
    def __init__(self, dsl: DSL, grid_min_dim: int = 5, grid_max_dim: int = 15, max_objects: int = 3, max_operations: int = 5, augmentation: Dict[str, bool] = None):
        self.dsl = dsl
        self.grid_min_dim = grid_min_dim
        self.grid_max_dim = grid_max_dim
        self.max_objects = max_objects
        self.max_operations = max_operations
        self.augmentation = augmentation if augmentation is not None else {
            "enable_rotation": True,
            "enable_reflection": True,
            "enable_color_permutation": True
        }
        self.color_alphabet = list(range(1, 10)) # Colors 1-9, 0 is background

    def _generate_random_grid(self, height: int, width: int, num_colors: int) -> ARCGrid:
        """Generates a random grid with a few objects."""
        grid_array = np.zeros((height, width), dtype=np.uint8)
        num_objects = random.randint(1, self.max_objects)
        for _ in range(num_objects):
            obj_h = random.randint(1, height // 2)
            obj_w = random.randint(1, width // 2)
            r_start = random.randint(0, height - obj_h)
            c_start = random.randint(0, width - obj_w)
            color = random.choice(self.color_alphabet[:num_colors])
            grid_array[r_start:r_start+obj_h, c_start:c_start+obj_w] = color
        return ARCGrid(grid_array)

    def _sample_primitive_and_args(self, grid: ARCGrid, primitives_subset: List[str]) -> Tuple[Primitive, Dict[str, Any]]:
        """Samples a primitive and valid arguments for the given grid."""
        available_primitives = [self.dsl.get_primitive(name) for name in primitives_subset if name in self.dsl.get_primitive_names()]
        if not available_primitives:
            raise ValueError("No primitives available in the specified subset.")

        primitive_cls = random.choice(available_primitives)
        primitive_instance = primitive_cls()
        args = {}

        for arg_name, arg_type in primitive_cls.arg_types.items():
            if arg_type == int:
                if arg_name == "color" or arg_name == "replacement_color":
                    args[arg_name] = random.choice(self.color_alphabet)
                elif arg_name in ["r", "r_start", "r_end"]:
                    args[arg_name] = random.randint(0, grid.height - 1)
                elif arg_name in ["c", "c_start", "c_end"]:
                    args[arg_name] = random.randint(0, grid.width - 1)
                elif arg_name == "k": # For rotation
                    args[arg_name] = random.choice([1, 2, 3]) # 90, 180, 270 degrees
                else:
                    args[arg_name] = random.randint(0, 5) # Default small int
            elif arg_type == ARCGrid:
                # For 'copy' primitive, generate a small source grid
                source_h = random.randint(1, min(grid.height, 5))
                source_w = random.randint(1, min(grid.width, 5))
                args[arg_name] = self._generate_random_grid(source_h, source_w, num_colors=2)
            elif arg_type == str:
                # For 'relational_compose', sample relation names
                if arg_name == "relation":
                    args[arg_name] = random.choice(["align_top_left", "overlap_center"])
                elif arg_name in ["op1", "op2"]:
                    args[arg_name] = random.choice(primitives_subset) # This is a simplification, should be sub-programs

        # Ensure r_start < r_end, c_start < c_end for PaintRectangle
        if primitive_cls.name == "paint_rectangle":
            if args["r_start"] >= args["r_end"]:
                args["r_end"] = args["r_start"] + random.randint(1, grid.height - args["r_start"])
            if args["c_start"] >= args["c_end"]:
                args["c_end"] = args["c_start"] + random.randint(1, grid.width - args["c_start"])

        return primitive_instance, args

    def generate_task(self, complexity: int, primitives_subset: List[str]) -> Dict[str, Any]:
        """Generates a single synthetic ARC task (input-output pair and the program)."""
        height = random.randint(self.grid_min_dim, self.grid_max_dim)
        width = random.randint(self.grid_min_dim, self.grid_max_dim)
        input_grid = self._generate_random_grid(height, width, num_colors=random.randint(2, 5))

        program_steps = []
        current_grid = input_grid.copy()

        for _ in range(complexity):
            try:
                primitive_instance, args = self._sample_primitive_and_args(current_grid, primitives_subset)
                step = ProgramStep(primitive_instance, args)
                program_steps.append(step)
                current_grid = primitive_instance.apply(current_grid, **args)
            except Exception as e:
                logger.warning(f"Failed to apply primitive during synthetic task generation: {e}")
                break # Stop generating if an error occurs

        program = Program(program_steps, self.dsl)
        output_grid = current_grid

        # Apply augmentations to input_grid (output_grid should be consistent)
        augmented_input = input_grid.copy()
        if self.augmentation["enable_rotation"] and random.random() < 0.5:
            k = random.choice([1, 2, 3])
            augmented_input = augmented_input.rotate(k)
            output_grid = output_grid.rotate(k) # Output must also be rotated
        if self.augmentation["enable_reflection"] and random.random() < 0.5:
            if random.random() < 0.5:
                augmented_input = augmented_input.reflect_horizontal()
                output_grid = output_grid.reflect_horizontal()
            else:
                augmented_input = augmented_input.reflect_vertical()
                output_grid = output_grid.reflect_vertical()
        if self.augmentation["enable_color_permutation"] and random.random() < 0.5:
            # Create a random mapping for colors 1-9
            color_map = {0: 0} # Background stays 0
            shuffled_colors = random.sample(self.color_alphabet, len(self.color_alphabet))
            for i, original_color in enumerate(self.color_alphabet):
                color_map[original_color] = shuffled_colors[i]
            
            augmented_input_array = np.vectorize(color_map.get)(augmented_input.grid)
            augmented_input = ARCGrid(augmented_input_array)
            output_grid_array = np.vectorize(color_map.get)(output_grid.grid)
            output_grid = ARCGrid(output_grid_array)

        return {
            "input": augmented_input.to_array(),
            "output": output_grid.to_array(),
            "program": program.to_json(),
            "task_id": f"synthetic_{random.getrandbits(32)}"
        }


class ARCProblemDataset(Dataset):
    """Dataset for ARC problems, handling both real and synthetic tasks."""
    def __init__(self, tasks: List[Dict[str, Any]], dsl: DSL, transform=None):
        self.tasks = tasks
        self.dsl = dsl
        self.transform = transform

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx) -> Dict[str, Any]:
        task = self.tasks[idx]
        input_grid_array = task["input"]
        output_grid_array = task["output"]

        input_grid = ARCGrid.from_array(input_grid_array)
        output_grid = ARCGrid.from_array(output_grid_array)

        # Convert grid to one-hot tensor for neural network input
        # Assuming 10 possible colors (0-9)
        input_tensor = self._grid_to_one_hot(input_grid)
        output_tensor = self._grid_to_one_hot(output_grid)

        # Program representation (e.g., sequence of primitive IDs and argument tensors)
        # This is a simplification; a real implementation would need to encode the program
        # into a format suitable for policy network training (e.g., target primitive, target args).
        program_json = task.get("program")
        program_representation = self._encode_program(program_json) if program_json else torch.tensor([])

        sample = {
            "task_id": task.get("task_id", f"task_{idx}"),
            "input_grid": input_tensor,
            "output_grid": output_tensor,
            "program_representation": program_representation
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def _grid_to_one_hot(self, grid: ARCGrid, num_colors: int = 10) -> torch.Tensor:
        """Converts an ARCGrid to a one-hot encoded tensor (C, H, W)."""
        grid_array = grid.grid
        one_hot = np.eye(num_colors)[grid_array]
        return torch.from_numpy(one_hot).permute(2, 0, 1).float() # (H, W, C) -> (C, H, W)

    def _encode_program(self, program_json: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Encodes a program (list of ProgramStep JSONs) into a format for learning.
        This is a highly simplified placeholder. A real implementation would map
        primitives to IDs, and arguments to tensors or embeddings.
        """
        primitive_ids = []
        arg_tensors = []
        for step_data in program_json:
            primitive_name = step_data["name"]
            # Map primitive_name to an integer ID
            primitive_id = list(self.dsl.get_all_primitives().keys()).index(primitive_name)
            primitive_ids.append(primitive_id)

            # Encode arguments (very basic for now)
            encoded_args = []
            for arg_name, arg_value in step_data["args"].items():
                if isinstance(arg_value, int):
                    encoded_args.append(float(arg_value))
                elif isinstance(arg_value, list): # Assuming ARCGrid.to_json() returns list of lists
                    # For grid arguments, could embed or summarize
                    encoded_args.append(0.0) # Placeholder
                else:
                    encoded_args.append(0.0) # Default for other types
            arg_tensors.append(torch.tensor(encoded_args, dtype=torch.float32))

        return {
            "primitive_ids": torch.tensor(primitive_ids, dtype=torch.long),
            "arg_tensors": arg_tensors # List of tensors, might need padding for batching
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching ARCProblemDataset samples."""
    input_grids = [item["input_grid"] for item in batch]
    output_grids = [item["output_grid"] for item in batch]
    task_ids = [item["task_id"] for item in batch]

    # Pad grids to the maximum dimensions in the batch
    max_h = max(grid.shape[1] for grid in input_grids)
    max_w = max(grid.shape[2] for grid in input_grids)

    padded_input_grids = []
    padded_output_grids = []
    for i_grid, o_grid in zip(input_grids, output_grids):
        _, h, w = i_grid.shape
        pad_h = max_h - h
        pad_w = max_w - w
        padded_i = F.pad(i_grid, (0, pad_w, 0, pad_h), 'constant', 0)
        padded_o = F.pad(o_grid, (0, pad_w, 0, pad_h), 'constant', 0)
        padded_input_grids.append(padded_i)
        padded_output_grids.append(padded_o)

    # Stack padded grids
    input_grids_batch = torch.stack(padded_input_grids)
    output_grids_batch = torch.stack(padded_output_grids)

    # Handle program representations (this is complex and needs careful design)
    # For now, we'll just return them as a list, not batched.
    program_representations = [item["program_representation"] for item in batch]

    return {
        "task_id": task_ids,
        "input_grid": input_grids_batch,
        "output_grid": output_grids_batch,
        "program_representation": program_representations
    }
