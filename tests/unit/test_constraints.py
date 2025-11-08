# SPDX-License-Identifier: MPL-2.0
import pytest
from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Paint, Rotate
from src.solver.constraints import ConstraintSolver

@pytest.fixture
def dsl():
    return DSL()

@pytest.fixture
def constraint_solver():
    return ConstraintSolver()

def test_infer_invariants_color_palette_preserved(constraint_solver):
    input_grid = ARCGrid.from_array([[1, 2], [3, 4]])
    output_grid = ARCGrid.from_array([[1, 2], [3, 4]])
    invariants = constraint_solver.infer_invariants(input_grid, output_grid)
    assert invariants["color_palette_preserved"] is True

    output_grid_changed = ARCGrid.from_array([[1, 2], [3, 5]])
    invariants_changed = constraint_solver.infer_invariants(input_grid, output_grid_changed)
    assert invariants_changed["color_palette_preserved"] is False

def test_infer_invariants_dimensions_preserved(constraint_solver):
    input_grid = ARCGrid.from_array([[1, 2], [3, 4]])
    output_grid = ARCGrid.from_array([[1, 2], [3, 4]])
    invariants = constraint_solver.infer_invariants(input_grid, output_grid)
    assert invariants["dimensions_preserved"] is True

    output_grid_resized = ARCGrid.from_array([[1, 2, 3], [4, 5, 6]])
    invariants_resized = constraint_solver.infer_invariants(input_grid, output_grid_resized)
    assert invariants_resized["dimensions_preserved"] is False

def test_check_program_invariants_pass(constraint_solver, dsl):
    input_grid = ARCGrid.from_array([[0, 0], [0, 0]])
    output_grid = ARCGrid.from_array([[1, 0], [0, 0]])

    # Program: Paint (0,0,1)
    paint_step = ProgramStep(dsl.get_primitive("paint")(), {"r": 0, "c": 0, "color": 1})
    program = Program([paint_step], dsl)

    # Invariants for this task: dimensions preserved, color palette changes (0->{0,1}), non-bg pixels change
    # The `infer_invariants` will detect these changes.
    # `check_program_invariants` will then verify if the program's output is consistent with these inferred changes.
    # For this simple case, the program should be consistent.
    assert constraint_solver.check_program_invariants(program, input_grid, output_grid) is True

def test_check_program_invariants_fail_dimensions(constraint_solver, dsl):
    input_grid = ARCGrid.from_array([[0, 0], [0, 0]])
    output_grid = ARCGrid.from_array([[0, 0], [0, 0]]) # Expected output is same size

    # Program: Rotate (k=1) on a 2x2 grid results in 2x2 grid, but let's simulate a program that changes size
    # (e.g., a crop or pad that makes it inconsistent with inferred invariants)
    # For this test, we'll make the program *produce* a different size, which should fail the check.
    class MockProgram(Program):
        def execute(self, input_grid: ARCGrid) -> ARCGrid:
            return ARCGrid.from_array([[0,0,0],[0,0,0],[0,0,0]]) # Returns 3x3

    mock_program = MockProgram([], dsl)
    # The inferred invariant will be dimensions_preserved = True (since input and output are 2x2)
    # But the mock_program produces 3x3, so it should fail.
    assert constraint_solver.check_program_invariants(mock_program, input_grid, output_grid) is False

def test_check_program_invariants_fail_color_palette(constraint_solver, dsl):
    input_grid = ARCGrid.from_array([[0, 0], [0, 0]])
    output_grid = ARCGrid.from_array([[1, 0], [0, 0]]) # Expected output introduces color 1

    # Program that produces a different color palette than expected
    class MockProgram(Program):
        def execute(self, input_grid: ARCGrid) -> ARCGrid:
            return ARCGrid.from_array([[2, 0], [0, 0]]) # Introduces color 2, not 1

    mock_program = MockProgram([], dsl)
    # Inferred invariant: color_palette_preserved = False, but the specific colors should be {0,1}
    # Mock program produces {0,2}, which should fail.
    assert constraint_solver.check_program_invariants(mock_program, input_grid, output_grid) is False

def test_solve_for_parameters_paint(constraint_solver):
    input_grid = ARCGrid.from_array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    output_grid = ARCGrid.from_array([[0, 0, 0], [0, 5, 0], [0, 0, 0]])

    # We expect a Paint operation at (1,1) with color 5
    params = constraint_solver.solve_for_parameters(input_grid, output_grid, Paint)
    assert params == {"r": 1, "c": 1, "color": 5}

    # No single pixel change
    output_grid_multi_change = ARCGrid.from_array([[1, 0, 0], [0, 5, 0], [0, 0, 0]])
    params_multi = constraint_solver.solve_for_parameters(input_grid, output_grid_multi_change, Paint)
    assert params_multi is None

    # No change
    params_no_change = constraint_solver.solve_for_parameters(input_grid, input_grid, Paint)
    assert params_no_change is None
