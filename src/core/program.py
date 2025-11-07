from typing import List, Dict, Any, Type
from src.core.grid import ARCGrid
from src.core.dsl import ProgramStep, DSL, Primitive

class Program:
    """Represents a sequence of DSL primitive applications."""
    def __init__(self, steps: List[ProgramStep], dsl: DSL):
        self.steps = steps
        self.dsl = dsl

    def execute(self, input_grid: ARCGrid) -> Any:
        current_grid = input_grid.copy()
        for step in self.steps:
            # Special handling for primitives that return non-grid values (e.g., CountColor)
            if step.primitive.name == "count_color":
                # If a non-grid value is returned, it's the final output of the program
                # This implies programs with non-grid outputs must end with such a primitive.
                # For ARC, the final output is always a grid, so this case might be for intermediate values.
                # For now, we'll assume the final output must be a grid.
                # If an intermediate step returns a non-grid, it needs to be stored/passed as an argument.
                # This requires a more sophisticated program execution model (e.g., stack-based).
                # For simplicity, we assume all intermediate steps produce grids.
                # If CountColor is the *last* step, its integer output is the program's output.
                if step == self.steps[-1]:
                    return step.primitive.apply(current_grid, **step.args)
                else:
                    # If CountColor is an intermediate step, its output needs to be captured
                    # and potentially used as an argument for a subsequent primitive.
                    # This is a limitation of the current simple sequential execution.
                    # For ARC, this is less common, as programs usually transform grids.
                    # For now, we'll just ignore intermediate non-grid outputs if not last step.
                    pass # Or raise an error for unsupported intermediate non-grid output
            else:
                current_grid = step.primitive.apply(current_grid, **step.args)
        return current_grid

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return "\n".join(str(step) for step in self.steps)

    def to_json(self) -> List[Dict[str, Any]]:
        return [step.to_json() for step in self.steps]

    @classmethod
    def from_json(cls, data: List[Dict[str, Any]], dsl: DSL) -> 'Program':
        steps = [ProgramStep.from_json(step_data, dsl.get_all_primitives()) for step_data in data]
        return cls(steps, dsl)

    def calculate_mdl(self) -> float:
        """Calculates the Minimum Description Length of the program.
        This is a simplified MDL calculation. A more robust one would consider
        argument complexity, primitive frequency, and compression ratios.
        """
        mdl = 0.0
        for step in self.steps:
            mdl += 1.0 # Base cost for the primitive itself
            for arg_name, arg_value in step.args.items():
                if isinstance(arg_value, int):
                    # Cost for integer arguments: e.g., log2(value) or fixed cost
                    # For ARC, colors are 0-9, coords are small. Fixed cost is reasonable.
                    mdl += 0.1
                elif isinstance(arg_value, ARCGrid):
                    # Cost for grid arguments: e.g., proportional to grid size or complexity
                    mdl += (arg_value.height * arg_value.width) / 100.0 # Example: proportional to grid size
                elif isinstance(arg_value, str):
                    mdl += len(arg_value) * 0.05 # Cost for string arguments (e.g., relation names)
        return mdl
