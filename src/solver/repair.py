# SPDX-License-Identifier: MPL-2.0
import random
from typing import List, Dict, Any, Optional
from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Primitive
from src.solver.verifier import ProgramVerifier
from src.solver.constraints import ConstraintSolver
from src.solver.utils import generate_random_args
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ProgramRepairer:
    """Attempts small edits to a program when verification fails."""
    def __init__(self, dsl: DSL, verifier: ProgramVerifier, constraint_solver: ConstraintSolver, max_repair_attempts: int = 5, edit_types: List[str] = None, repair_budget_per_attempt_ms: int = 1000):
        self.dsl = dsl
        self.verifier = verifier
        self.constraint_solver = constraint_solver
        self.max_repair_attempts = max_repair_attempts
        self.edit_types = edit_types if edit_types is not None else ["argument_change", "primitive_swap", "add_primitive", "remove_primitive"]
        self.repair_budget_per_attempt_ms = repair_budget_per_attempt_ms
        self.all_primitives = list(self.dsl.get_all_primitives().values())

    def _propose_argument_change(self, program: Program, step_idx: int, task_input_grid: ARCGrid) -> Optional[Program]:
        """Proposes a small change to an argument of a primitive."""
        original_step = program.steps[step_idx]
        primitive = original_step.primitive
        original_args = original_step.args.copy()

        if not original_args: return None

        arg_to_change = random.choice(list(original_args.keys()))
        original_value = original_args[arg_to_change]
        new_args = original_args.copy()

        # Simple perturbation based on type
        if isinstance(original_value, int):
            if arg_to_change in ["r", "c", "r_start", "c_start", "r_end", "c_end"]:
                # Perturb coordinates by +/- 1 or 2
                new_value = original_value + random.choice([-2, -1, 1, 2])
                # Clamp to grid dimensions
                if arg_to_change.startswith('r'):
                    new_value = max(0, min(new_value, task_input_grid.height - 1))
                else:
                    new_value = max(0, min(new_value, task_input_grid.width - 1))
            elif arg_to_change in ["color", "replacement_color"]:
                # Change to a different random color (0-9)
                new_value = random.choice([c for c in range(10) if c != original_value])
            elif arg_to_change == "k": # Rotation
                new_value = random.choice([1, 2, 3])
            else:
                new_value = original_value + random.choice([-1, 1])
            new_args[arg_to_change] = new_value
        elif isinstance(original_value, str):
            # For string arguments like 'relation', try another valid option
            if arg_to_change == "relation":
                new_args[arg_to_change] = random.choice([r for r in ["align_top_left", "overlap_center"] if r != original_value])
            else:
                return None # Cannot easily perturb other strings
        elif isinstance(original_value, ARCGrid):
            # For grid arguments, could try rotating/reflecting the sub-grid
            if random.random() < 0.5:
                new_args[arg_to_change] = original_value.rotate(random.choice([1,2,3]))
            else:
                new_args[arg_to_change] = original_value.reflect_horizontal()
        else:
            return None

        # Create new program with modified step
        new_steps = list(program.steps)
        new_steps[step_idx] = ProgramStep(primitive, new_args)
        return Program(new_steps, self.dsl)

    def _propose_primitive_swap(self, program: Program, step_idx: int) -> Optional[Program]:
        """Swaps a primitive with another compatible primitive."""
        if len(program.steps) < 2: return None
        if step_idx >= len(program.steps) - 1: return None # Cannot swap last with next

        new_steps = list(program.steps)
        new_steps[step_idx], new_steps[step_idx+1] = new_steps[step_idx+1], new_steps[step_idx]
        return Program(new_steps, self.dsl)

    def _propose_add_primitive(self, program: Program, task_input_grid: ARCGrid) -> Optional[Program]:
        """Adds a new primitive at a random position."""
        if len(program.steps) >= program.dsl.max_program_length: return None # Assuming DSL has max_program_length

        insert_idx = random.randint(0, len(program.steps))
        primitive_cls = random.choice(self.all_primitives)
        primitive_instance = primitive_cls()
        # Need to determine the grid state at insert_idx to generate valid args
        # This requires partial execution of the program up to insert_idx.
        # For simplicity, we'll use the initial grid for arg generation, which might be invalid.
        # A proper implementation would execute up to insert_idx.
        current_grid_at_insert = task_input_grid # Simplified
        try:
            # Get primitive names from DSL
            primitive_names = self.dsl.get_primitive_names()
            args = generate_random_args(primitive_instance, current_grid_at_insert, primitive_names)
            new_step = ProgramStep(primitive_instance, args)
            new_steps = list(program.steps)
            new_steps.insert(insert_idx, new_step)
            return Program(new_steps, self.dsl)
        except Exception:
            return None

    def _propose_remove_primitive(self, program: Program) -> Optional[Program]:
        """Removes a primitive from a random position."""
        if not program.steps: return None
        remove_idx = random.randint(0, len(program.steps) - 1)
        new_steps = list(program.steps)
        new_steps.pop(remove_idx)
        return Program(new_steps, self.dsl)

    def repair_program(self, program: Program, task_input_grid: ARCGrid, task_output_grid: ARCGrid) -> Optional[Program]:
        """Attempts to repair a program that failed verification."""
        logger.debug(f"Attempting to repair program: {program}")
        best_repaired_program = None

        for attempt in range(self.max_repair_attempts):
            edit_type = random.choice(self.edit_types)
            step_idx = random.randint(0, len(program.steps) - 1) if program.steps else 0
            candidate_program = None

            if edit_type == "argument_change":
                candidate_program = self._propose_argument_change(program, step_idx, task_input_grid)
            elif edit_type == "primitive_swap":
                candidate_program = self._propose_primitive_swap(program, step_idx)
            elif edit_type == "add_primitive":
                candidate_program = self._propose_add_primitive(program, task_input_grid)
            elif edit_type == "remove_primitive":
                candidate_program = self._propose_remove_primitive(program)

            if candidate_program is None: continue

            # Verify the repaired program
            if self.verifier.verify_program_on_examples(candidate_program, [
                {"input": task_input_grid, "output": task_output_grid}
            ]):
                logger.info(f"Program repaired successfully on attempt {attempt+1} with {edit_type} edit.")
                return candidate_program
            else:
                logger.debug(f"Repaired program failed verification on attempt {attempt+1}.")

        logger.warning("Program repair failed after all attempts.")
        return None
