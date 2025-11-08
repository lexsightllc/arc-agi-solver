# SPDX-License-Identifier: MPL-2.0
import pytest
import numpy as np
from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Paint, Rotate, Copy, CountColor

@pytest.fixture
def sample_dsl():
    return DSL()

@pytest.fixture
def sample_grid():
    return ARCGrid.from_array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

def test_dsl_registration(sample_dsl):
    assert "paint" in sample_dsl.get_primitive_names()
    assert "rotate" in sample_dsl.get_primitive_names()
    assert issubclass(sample_dsl.get_primitive("paint"), Paint)

def test_primitive_paint(sample_grid):
    paint_primitive = Paint()
    result_grid = paint_primitive.apply(sample_grid, r=0, c=0, color=5)
    expected_grid = ARCGrid.from_array([[5, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert result_grid == expected_grid

def test_primitive_rotate(sample_grid):
    rotate_primitive = Rotate()
    result_grid = rotate_primitive.apply(sample_grid, k=1)
    expected_grid = ARCGrid.from_array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) # 90 deg clockwise
    assert result_grid == expected_grid # Symmetric grid, so it looks the same

    non_symmetric_grid = ARCGrid.from_array([[1,0],[0,0]])
    rotated_grid = rotate_primitive.apply(non_symmetric_grid, k=1)
    expected_rotated = ARCGrid.from_array([[0,1],[0,0]])
    assert rotated_grid == expected_rotated

def test_primitive_copy(sample_grid):
    copy_primitive = Copy()
    source_grid = ARCGrid.from_array([[2, 2], [2, 2]])
    result_grid = copy_primitive.apply(sample_grid, source_grid=source_grid, r_offset=0, c_offset=0)
    expected_grid = ARCGrid.from_array([[2, 2, 0], [2, 2, 0], [0, 0, 0]])
    assert result_grid == expected_grid

def test_primitive_count_color(sample_grid):
    count_primitive = CountColor()
    result_count = count_primitive.apply(sample_grid, color=1)
    assert result_count == 1

    result_count_zero = count_primitive.apply(sample_grid, color=5)
    assert result_count_zero == 0

def test_program_execution(sample_dsl, sample_grid):
    # Program: Paint (0,0,5) -> Rotate (k=1)
    paint_step = ProgramStep(sample_dsl.get_primitive("paint")(), {"r": 0, "c": 0, "color": 5})
    rotate_step = ProgramStep(sample_dsl.get_primitive("rotate")(), {"k": 1})
    program = Program([paint_step, rotate_step], sample_dsl)

    result_grid = program.execute(sample_grid)
    expected_intermediate = ARCGrid.from_array([[5, 0, 0], [0, 1, 0], [0, 0, 0]])
    expected_final = expected_intermediate.rotate(k=1)
    assert result_grid == expected_final

def test_program_mdl_calculation(sample_dsl, sample_grid):
    paint_step = ProgramStep(sample_dsl.get_primitive("paint")(), {"r": 0, "c": 0, "color": 5})
    rotate_step = ProgramStep(sample_dsl.get_primitive("rotate")(), {"k": 1})
    copy_step = ProgramStep(sample_dsl.get_primitive("copy")(), {"source_grid": sample_grid, "r_offset": 0, "c_offset": 0})

    program1 = Program([paint_step], sample_dsl)
    program2 = Program([paint_step, rotate_step], sample_dsl)
    program3 = Program([paint_step, rotate_step, copy_step], sample_dsl)

    mdl1 = program1.calculate_mdl()
    mdl2 = program2.calculate_mdl()
    mdl3 = program3.calculate_mdl()

    assert mdl1 < mdl2
    assert mdl2 < mdl3

    # Check specific values (approximate due to fixed costs)
    assert mdl1 == pytest.approx(1.3) # 1 (primitive) + 3*0.1 (int args)
    assert mdl2 == pytest.approx(1.3 + 1.1) # mdl1 + 1 (primitive) + 1*0.1 (int arg)
    assert mdl3 > mdl2 # Should be higher due to grid argument
