# SPDX-License-Identifier: MPL-2.0
"""Shared utility functions for solver components."""

import numpy as np
from typing import Dict, Any, List
from src.core.grid import ARCGrid
from src.core.dsl import Primitive


def generate_random_args(primitive: Primitive, current_grid: ARCGrid,
                        primitive_names: List[str] = None) -> Dict[str, Any]:
    """Generates random but valid arguments for a primitive based on current grid state.

    Args:
        primitive: The primitive instance to generate arguments for
        current_grid: The current grid state
        primitive_names: List of valid primitive names (for relational composition)

    Returns:
        Dictionary of argument name to value mappings
    """
    args = {}
    for arg_name, arg_type in primitive.arg_types.items():
        if arg_type == int:
            if arg_name == "color" or arg_name == "replacement_color":
                args[arg_name] = np.random.randint(0, 10)  # ARC colors 0-9
            elif arg_name in ["r", "r_start", "r_end"]:
                args[arg_name] = np.random.randint(0, current_grid.height)
            elif arg_name in ["c", "c_start", "c_end"]:
                args[arg_name] = np.random.randint(0, current_grid.width)
            elif arg_name == "k":  # For rotation
                args[arg_name] = np.random.choice([1, 2, 3])  # 90, 180, 270 degrees
            else:
                args[arg_name] = np.random.randint(0, 5)  # Default small int
        elif arg_type == ARCGrid:
            # For 'copy' primitive, generate a small source grid
            source_h = np.random.randint(1, min(current_grid.height, 5))
            source_w = np.random.randint(1, min(current_grid.width, 5))
            args[arg_name] = ARCGrid(np.random.randint(0, 10, size=(source_h, source_w), dtype=np.uint8))
        elif arg_type == str:
            if arg_name == "relation":
                args[arg_name] = np.random.choice(["align_top_left", "overlap_center"])
            elif arg_name in ["op1", "op2"]:
                if primitive_names:
                    args[arg_name] = np.random.choice(primitive_names)
                else:
                    args[arg_name] = "paint"  # Default fallback

    # Ensure r_start < r_end, c_start < c_end for PaintRectangle
    if primitive.name == "paint_rectangle":
        if "r_start" in args and "r_end" in args and args["r_start"] >= args["r_end"]:
            args["r_end"] = args["r_start"] + 1
        if "c_start" in args and "c_end" in args and args["c_start"] >= args["c_end"]:
            args["c_end"] = args["c_start"] + 1

    return args
