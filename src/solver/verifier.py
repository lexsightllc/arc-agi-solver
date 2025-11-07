from src.core.grid import ARCGrid
from src.core.dsl import Program
from typing import List, Dict, Any
from src.utils.logging import get_logger

logger = get_logger(__name__)

class ProgramVerifier:
    """Executes candidate programs and checks exact equality with example outputs."""
    def __init__(self, strict_equality: bool = True):
        self.strict_equality = strict_equality

    def verify(self, candidate_output_grid: ARCGrid, target_output_grid: ARCGrid) -> bool:
        """Checks if the candidate output grid exactly matches the target output grid."""
        if self.strict_equality:
            return candidate_output_grid == target_output_grid
        else:
            # Implement a more lenient comparison if strict_equality is False
            # e.g., allow for minor color permutations, or ignore background differences
            # For ARC, strict equality is usually required.
            logger.warning("Non-strict equality verification is not fully implemented for ARC.")
            return candidate_output_grid == target_output_grid

    def verify_program_on_examples(self, program: Program, examples: List[Dict[str, Any]]) -> bool:
        """Verifies a program against a list of input-output examples.
        Each example is expected to have 'input' (ARCGrid) and 'output' (ARCGrid).
        """
        for i, example in enumerate(examples):
            input_grid = example["input"]
            expected_output_grid = example["output"]

            try:
                actual_output_grid = program.execute(input_grid)
                if not isinstance(actual_output_grid, ARCGrid):
                    logger.warning(f"Program for example {i} returned non-grid output. Verification failed.")
                    return False

                if not self.verify(actual_output_grid, expected_output_grid):
                    logger.debug(f"Program failed verification for example {i}.")
                    return False
            except Exception as e:
                logger.debug(f"Error during program execution for example {i}: {e}")
                return False
        logger.debug("Program passed verification on all examples.")
        return True
