# SPDX-License-Identifier: MPL-2.0
import pytest
import numpy as np
from src.core.grid import ARCGrid

def test_arcgrid_creation():
    grid_array = [[0, 1, 2], [3, 4, 5]]
    grid = ARCGrid.from_array(grid_array)
    assert grid.height == 2
    assert grid.width == 3
    assert np.array_equal(grid.grid, np.array(grid_array, dtype=np.uint8))

def test_arcgrid_equality():
    grid1 = ARCGrid.from_array([[1, 2], [3, 4]])
    grid2 = ARCGrid.from_array([[1, 2], [3, 4]])
    grid3 = ARCGrid.from_array([[1, 2], [3, 5]])
    assert grid1 == grid2
    assert grid1 != grid3

def test_arcgrid_rotate():
    grid = ARCGrid.from_array([[1, 2], [3, 4]])
    rotated_90 = grid.rotate(k=1) # 90 deg clockwise
    expected_90 = ARCGrid.from_array([[3, 1], [4, 2]])
    assert rotated_90 == expected_90

    rotated_180 = grid.rotate(k=2)
    expected_180 = ARCGrid.from_array([[4, 3], [2, 1]])
    assert rotated_180 == expected_180

def test_arcgrid_reflect_horizontal():
    grid = ARCGrid.from_array([[1, 2, 3], [4, 5, 6]])
    reflected = grid.reflect_horizontal()
    expected = ARCGrid.from_array([[3, 2, 1], [6, 5, 4]])
    assert reflected == expected

def test_arcgrid_reflect_vertical():
    grid = ARCGrid.from_array([[1, 2, 3], [4, 5, 6]])
    reflected = grid.reflect_vertical()
    expected = ARCGrid.from_array([[4, 5, 6], [1, 2, 3]])
    assert reflected == expected

def test_arcgrid_crop():
    grid = ARCGrid.from_array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    cropped = grid.crop(0, 1, 2, 3)
    expected = ARCGrid.from_array([[2, 3], [5, 6]])
    assert cropped == expected

def test_arcgrid_pad():
    grid = ARCGrid.from_array([[1]])
    padded = grid.pad(1, 1, 1, 1, color=0)
    expected = ARCGrid.from_array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert padded == expected

def test_arcgrid_paint_rectangle():
    grid = ARCGrid.from_array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    painted = grid.paint_rectangle(0, 0, 2, 2, 1)
    expected = ARCGrid.from_array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    assert painted == expected

def test_arcgrid_flood_fill():
    grid = ARCGrid.from_array([[1, 1, 2], [1, 2, 2], [3, 3, 3]])
    filled = grid.flood_fill(0, 0, 5)
    expected = ARCGrid.from_array([[5, 5, 2], [5, 2, 2], [3, 3, 3]])
    assert filled == expected

def test_arcgrid_get_colors():
    grid = ARCGrid.from_array([[1, 0, 2], [3, 1, 0]])
    assert grid.get_colors() == [0, 1, 2, 3]

def test_arcgrid_count_color():
    grid = ARCGrid.from_array([[1, 0, 2], [3, 1, 0]])
    assert grid.count_color(0) == 2
    assert grid.count_color(1) == 2
    assert grid.count_color(5) == 0

def test_arcgrid_get_bounding_box():
    grid = ARCGrid.from_array([[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]])
    bbox = grid.get_bounding_box(color=1)
    assert bbox == (1, 1, 3, 2) # r_min, c_min, r_max (exclusive), c_max (exclusive)

    bbox_all = grid.get_bounding_box() # Non-background
    assert bbox_all == (1, 1, 3, 2)

    empty_grid = ARCGrid.from_array([[0,0],[0,0]])
    assert empty_grid.get_bounding_box() is None

def test_arcgrid_get_objects():
    grid = ARCGrid.from_array([
        [0, 1, 0, 0],
        [1, 1, 0, 2],
        [0, 0, 0, 2],
        [0, 3, 3, 0]
    ])
    objects = grid.get_objects()
    assert len(objects) == 3

    # Object 1 (top-left L-shape)
    expected_obj1 = ARCGrid.from_array([
        [0, 1],
        [1, 1]
    ])
    assert any(obj == expected_obj1 for obj in objects)

    # Object 2 (right-side vertical line)
    expected_obj2 = ARCGrid.from_array([
        [2],
        [2]
    ])
    assert any(obj == expected_obj2 for obj in objects)

    # Object 3 (bottom-left 3s)
    expected_obj3 = ARCGrid.from_array([
        [3, 3]
    ])
    assert any(obj == expected_obj3 for obj in objects)

def test_arcgrid_overlay():
    base_grid = ARCGrid.from_array([[1,1,1],[1,1,1],[1,1,1]])
    overlay_grid = ARCGrid.from_array([[0,2],[3,0]])
    result = base_grid.overlay(overlay_grid, r_offset=0, c_offset=1, transparent_color=0)
    expected = ARCGrid.from_array([[1,2,1],[1,3,1],[1,1,1]])
    assert result == expected

    result_no_transparent = base_grid.overlay(overlay_grid, r_offset=0, c_offset=1, transparent_color=99) # No transparent color
    expected_no_transparent = ARCGrid.from_array([[1,0,2],[1,3,0],[1,1,1]])
    assert result_no_transparent == expected_no_transparent
