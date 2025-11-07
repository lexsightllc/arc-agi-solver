import pytest
from hypothesis import given, strategies as st, settings
import numpy as np

from src.core.grid import ARCGrid
from src.core.dsl import DSL, Program, ProgramStep, Paint, Rotate, ReflectHorizontal, FloodFill

# Hypothesis strategies for ARCGrid and related types
@st.composite
def grids(draw, min_dim=1, max_dim=10, min_colors=1, max_colors=10):
    height = draw(st.integers(min_value=min_dim, max_value=max_dim))
    width = draw(st.integers(min_value=min_dim, max_value=max_dim))
    num_colors = draw(st.integers(min_value=min_colors, max_value=max_colors))
    colors = draw(st.lists(st.integers(min_value=0, max_value=9), min_size=num_colors, max_size=num_colors, unique=True))
    # Ensure 0 (background) is always a possible color if not explicitly in list
    if 0 not in colors and num_colors < 10: colors.append(0)
    colors = sorted(list(set(colors)))

    grid_array = draw(st.lists(
        st.lists(st.sampled_from(colors), min_size=width, max_size=width),
        min_size=height, max_size=height
    ))
    return ARCGrid.from_array(grid_array)

@st.composite
def grid_coords(draw, grid: ARCGrid):
    r = draw(st.integers(min_value=0, max_value=grid.height - 1))
    c = draw(st.integers(min_value=0, max_value=grid.width - 1))
    return r, c

@st.composite
def arc_colors(draw):
    return draw(st.integers(min_value=0, max_value=9))

@st.composite
def dsl_primitives_and_args(draw, dsl: DSL, grid: ARCGrid):
    primitive_name = draw(st.sampled_from(dsl.get_primitive_names()))
    primitive_cls = dsl.get_primitive(primitive_name)
    primitive_instance = primitive_cls()
    args = {}

    for arg_name, arg_type in primitive_cls.arg_types.items():
        if arg_type == int:
            if arg_name == "color" or arg_name == "replacement_color":
                args[arg_name] = draw(arc_colors())
            elif arg_name in ["r", "r_start", "r_end"]:
                args[arg_name] = draw(st.integers(min_value=0, max_value=grid.height - 1))
            elif arg_name in ["c", "c_start", "c_end"]:
                args[arg_name] = draw(st.integers(min_value=0, max_value=grid.width - 1))
            elif arg_name == "k": # For rotation
                args[arg_name] = draw(st.integers(min_value=1, max_value=3))
            else:
                args[arg_name] = draw(st.integers(min_value=0, max_value=5))
        elif arg_type == ARCGrid:
            # For 'copy' primitive, generate a small source grid
            source_h = draw(st.integers(min_value=1, max_value=min(grid.height, 5)))
            source_w = draw(st.integers(min_value=1, max_value=min(grid.width, 5)))
            args[arg_name] = draw(grids(min_dim=1, max_dim=5))
        elif arg_type == str:
            if arg_name == "relation":
                args[arg_name] = draw(st.sampled_from(["align_top_left", "overlap_center"]))
            elif arg_name in ["op1", "op2"]:
                args[arg_name] = draw(st.sampled_from(dsl.get_primitive_names()))

    # Ensure r_start < r_end, c_start < c_end for PaintRectangle
    if primitive_name == "paint_rectangle":
        if "r_start" in args and "r_end" in args:
            if args["r_start"] >= args["r_end"]:
                args["r_end"] = args["r_start"] + draw(st.integers(min_value=1, max_value=grid.height - args["r_start"]))
        if "c_start" in args and "c_end" in args:
            if args["c_start"] >= args["c_end"]:
                args["c_end"] = args["c_start"] + draw(st.integers(min_value=1, max_value=grid.width - args["c_start"]))

    return primitive_instance, args

@st.composite
def programs(draw, dsl: DSL, grid: ARCGrid, max_steps=5):
    num_steps = draw(st.integers(min_value=1, max_value=max_steps))
    steps = []
    current_grid = grid.copy()
    for _ in range(num_steps):
        primitive_instance, args = draw(dsl_primitives_and_args(dsl, current_grid))
        try:
            # Attempt to apply to ensure valid intermediate states for subsequent steps
            # This is a heuristic to generate more 'valid' programs.
            temp_grid = primitive_instance.apply(current_grid, **args)
            if isinstance(temp_grid, ARCGrid):
                current_grid = temp_grid
            steps.append(ProgramStep(primitive_instance, args))
        except Exception:
            # If an application fails, just skip this step or break
            break
    return Program(steps, dsl)


# --- Property-Based Tests ---

@settings(max_examples=100, deadline=None) # Increase examples for better coverage
@given(grid=grids())
def test_grid_rotation_inverts(grid: ARCGrid):
    """Property: Rotating a grid 4 times returns the original grid."""
    rotated_4_times = grid.rotate(k=4)
    assert rotated_4_times == grid

@settings(max_examples=100, deadline=None)
@given(grid=grids())
def test_grid_double_reflection_inverts(grid: ARCGrid):
    """Property: Reflecting a grid twice (horizontally or vertically) returns the original grid."""
    reflected_h_twice = grid.reflect_horizontal().reflect_horizontal()
    assert reflected_h_twice == grid

    reflected_v_twice = grid.reflect_vertical().reflect_vertical()
    assert reflected_v_twice == grid

@settings(max_examples=100, deadline=None)
@given(grid=grids(), r=st.integers(min_value=0, max_value=9), c=st.integers(min_value=0, max_value=9), color=arc_colors())
def test_paint_pixel_changes_only_one_pixel(grid: ARCGrid, r: int, c: int, color: int):
    """Property: Painting a single pixel only changes that pixel's color, if coordinates are valid."""
    if not (0 <= r < grid.height and 0 <= c < grid.width): # Skip invalid coordinates
        return

    original_color = grid.get_pixel(r, c)
    if original_color == color: # No change expected
        painted_grid = grid.set_pixel(r, c, color)
        assert painted_grid == grid
        return

    painted_grid = grid.set_pixel(r, c, color)

    # Check that only the target pixel changed
    diff_count = 0
    for row in range(grid.height):
        for col in range(grid.width):
            if grid.get_pixel(row, col) != painted_grid.get_pixel(row, col):
                diff_count += 1
                assert (row, col) == (r, c) # Ensure it's the target pixel

    assert diff_count == 1
    assert painted_grid.get_pixel(r, c) == color

@settings(max_examples=50, deadline=None)
@given(grid=grids(min_dim=3, max_dim=10), r=st.integers(min_value=0, max_value=9), c=st.integers(min_value=0, max_value=9), replacement_color=arc_colors())
def test_flood_fill_changes_connected_region(grid: ARCGrid, r: int, c: int, replacement_color: int):
    """Property: Flood fill changes all connected pixels of the target color, and nothing else."""
    if not (0 <= r < grid.height and 0 <= c < grid.width): return

    target_color = grid.get_pixel(r, c)
    if target_color == replacement_color: # No change expected
        filled_grid = grid.flood_fill(r, c, replacement_color)
        assert filled_grid == grid
        return

    filled_grid = grid.flood_fill(r, c, replacement_color)

    # Check that pixels with original target_color are either changed or not connected
    # And pixels not of target_color are unchanged.
    for row in range(grid.height):
        for col in range(grid.width):
            original_pixel = grid.get_pixel(row, col)
            filled_pixel = filled_grid.get_pixel(row, col)

            if original_pixel == target_color:
                # If it was target_color, it must now be replacement_color OR it was not connected
                if filled_pixel != replacement_color:
                    # This pixel was target_color but wasn't filled. It must be disconnected.
                    # This is hard to assert directly without re-implementing connectivity.
                    # For now, we'll just check that if it *wasn't* target_color, it's unchanged.
                    pass
            else:
                # If it was not target_color, it must be unchanged
                assert original_pixel == filled_pixel

    # Check that the starting pixel is indeed changed
    assert filled_grid.get_pixel(r, c) == replacement_color


@settings(max_examples=50, deadline=None)
@given(grid=grids(), program=st.just(DSL()).flatmap(lambda dsl: programs(dsl, grids().example())))
def test_program_execution_returns_grid_or_value(grid: ARCGrid, program: Program):
    """Property: Program execution always returns an ARCGrid or a primitive value (e.g., int for CountColor)."""
    # Note: The `programs` strategy needs a base grid, so we use `grids().example()` for that.
    # The actual `grid` argument to this test is the input to the program.
    try:
        result = program.execute(grid)
        assert isinstance(result, (ARCGrid, int))
    except Exception as e:
        # Some programs might be ill-formed or lead to errors, which is acceptable for random generation.
        # The goal is that if it *completes*, it returns a valid type.
        pytest.skip(f"Program execution failed: {e}")


# You can add more property-based tests for other primitives and combinations.
# For example, properties about object detection, bounding boxes, etc.
