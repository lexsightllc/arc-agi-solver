import numpy as np
from numba import njit, prange
from typing import List, Tuple, Optional, Dict

@njit(cache=True)
def _rotate_grid_90(grid: np.ndarray) -> np.ndarray:
    """Rotates a 2D grid 90 degrees clockwise."""
    return np.rot90(grid, k=-1)

@njit(cache=True)
def _reflect_grid_horizontal(grid: np.ndarray) -> np.ndarray:
    """Reflects a 2D grid horizontally (left-right)."""
    return np.fliplr(grid)

@njit(cache=True)
def _reflect_grid_vertical(grid: np.ndarray) -> np.ndarray:
    """Reflects a 2D grid vertically (up-down)."""
    return np.flipud(grid)

@njit(cache=True)
def _flood_fill_numba(grid: np.ndarray, start_row: int, start_col: int, target_color: int, replacement_color: int) -> np.ndarray:
    """Numba-optimized flood fill implementation."""
    if target_color == replacement_color:
        return grid

    rows, cols = grid.shape
    new_grid = grid.copy()
    q = [(start_row, start_col)]

    while len(q) > 0:
        r, c = q.pop(0)
        if not (0 <= r < rows and 0 <= c < cols and new_grid[r, c] == target_color):
            continue

        new_grid[r, c] = replacement_color

        # Add neighbors to queue
        if r > 0: q.append((r - 1, c))
        if r < rows - 1: q.append((r + 1, c))
        if c > 0: q.append((r, c - 1))
        if c < cols - 1: q.append((r, c + 1))
    return new_grid

class ARCGrid:
    """Represents an ARC grid with various utility methods."""
    def __init__(self, grid_array: np.ndarray):
        if grid_array.ndim != 2:
            raise ValueError("Grid array must be 2-dimensional.")
        self.grid = grid_array.astype(np.uint8) # ARC colors are 0-9
        self.height, self.width = self.grid.shape

    @classmethod
    def from_array(cls, array: List[List[int]]) -> 'ARCGrid':
        return cls(np.array(array, dtype=np.uint8))

    def to_array(self) -> List[List[int]]:
        return self.grid.tolist()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ARCGrid):
            return NotImplemented
        return np.array_equal(self.grid, other.grid)

    def __hash__(self) -> int:
        return hash(self.grid.tobytes())

    def __repr__(self) -> str:
        return f"ARCGrid({self.height}x{self.width})\n{self.grid}"

    def copy(self) -> 'ARCGrid':
        return ARCGrid(self.grid.copy())

    def get_pixel(self, r: int, c: int) -> int:
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise IndexError(f"Coordinates ({r},{c}) out of bounds for {self.height}x{self.width} grid.")
        return self.grid[r, c].item()

    def set_pixel(self, r: int, c: int, color: int) -> 'ARCGrid':
        new_grid = self.grid.copy()
        if not (0 <= r < self.height and 0 <= c < self.width):
            raise IndexError(f"Coordinates ({r},{c}) out of bounds for {self.height}x{self.width} grid.")
        new_grid[r, c] = color
        return ARCGrid(new_grid)

    def rotate(self, k: int = 1) -> 'ARCGrid':
        """Rotates the grid 90 degrees clockwise k times."""
        rotated_grid = np.rot90(self.grid, k=-k) # np.rot90 k=1 is counter-clockwise, so -k for clockwise
        return ARCGrid(rotated_grid)

    def reflect_horizontal(self) -> 'ARCGrid':
        """Reflects the grid horizontally (left-right)."""
        return ARCGrid(_reflect_grid_horizontal(self.grid))

    def reflect_vertical(self) -> 'ARCGrid':
        """Reflects the grid vertically (up-down)."""
        return ARCGrid(_reflect_grid_vertical(self.grid))

    def crop(self, r_start: int, c_start: int, r_end: int, c_end: int) -> 'ARCGrid':
        """Crops the grid to the specified bounding box [r_start:r_end, c_start:c_end)."""
        if not (0 <= r_start < r_end <= self.height and 0 <= c_start < c_end <= self.width):
            raise ValueError("Invalid crop coordinates.")
        return ARCGrid(self.grid[r_start:r_end, c_start:c_end].copy())

    def pad(self, top: int, bottom: int, left: int, right: int, color: int = 0) -> 'ARCGrid':
        """Pads the grid with a specified color."""
        padded_grid = np.pad(self.grid, ((top, bottom), (left, right)), 'constant', constant_values=color)
        return ARCGrid(padded_grid)

    def paint_rectangle(self, r_start: int, c_start: int, r_end: int, c_end: int, color: int) -> 'ARCGrid':
        """Paints a rectangular region with a given color."""
        new_grid = self.grid.copy()
        new_grid[r_start:r_end, c_start:c_end] = color
        return ARCGrid(new_grid)

    def flood_fill(self, r: int, c: int, replacement_color: int) -> 'ARCGrid':
        """Performs a flood fill starting from (r, c) with replacement_color."""
        if not (0 <= r < self.height and 0 <= c < self.width):
            return self.copy() # No-op if start point is out of bounds
        target_color = self.grid[r, c]
        return ARCGrid(_flood_fill_numba(self.grid, r, c, target_color, replacement_color))

    def get_colors(self) -> List[int]:
        """Returns a sorted list of unique colors present in the grid."""
        return sorted(list(np.unique(self.grid)))

    def count_color(self, color: int) -> int:
        """Counts occurrences of a specific color."""
        return np.sum(self.grid == color).item()

    def find_color_positions(self, color: int) -> List[Tuple[int, int]]:
        """Returns a list of (row, col) tuples for all pixels of a given color."""
        return list(zip(*np.where(self.grid == color)))

    def get_bounding_box(self, color: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
        """Returns (r_min, c_min, r_max, c_max) for non-background pixels or a specific color."""
        if color is None:
            coords = np.argwhere(self.grid != 0) # Assuming 0 is background
        else:
            coords = np.argwhere(self.grid == color)

        if coords.size == 0:
            return None

        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        return r_min, c_min, r_max + 1, c_max + 1 # +1 for exclusive end coordinates

    def get_objects(self, background_color: int = 0) -> List['ARCGrid']:
        """Identifies connected components (objects) using scikit-image's label function."""
        from skimage.measure import label
        labeled_array = label(self.grid != background_color, connectivity=1) # 1 for 4-connectivity, 2 for 8-connectivity
        num_objects = labeled_array.max()

        objects = []
        for i in range(1, num_objects + 1):
            obj_mask = (labeled_array == i)
            r_min, c_min, r_max, c_max = self.get_bounding_box_from_mask(obj_mask)
            cropped_obj_grid = self.grid[r_min:r_max, c_min:c_max].copy()
            # Mask out other objects in the cropped view
            cropped_mask = obj_mask[r_min:r_max, c_min:c_max]
            obj_grid_array = np.where(cropped_mask, cropped_obj_grid, background_color)
            objects.append(ARCGrid(obj_grid_array))
        return objects

    def get_bounding_box_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Helper to get bounding box from a boolean mask."""
        coords = np.argwhere(mask)
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        return r_min, c_min, r_max + 1, c_max + 1

    def overlay(self, other: 'ARCGrid', r_offset: int = 0, c_offset: int = 0, transparent_color: int = 0) -> 'ARCGrid':
        """Overlays another grid onto this grid at a specified offset, treating transparent_color as transparent."""
        new_grid = self.grid.copy()
        h1, w1 = self.height, self.width
        h2, w2 = other.height, other.width

        for r_other in range(h2):
            for c_other in range(w2):
                if other.grid[r_other, c_other] != transparent_color:
                    r_self = r_other + r_offset
                    c_self = c_other + c_offset
                    if 0 <= r_self < h1 and 0 <= c_self < w1:
                        new_grid[r_self, c_self] = other.grid[r_other, c_other]
        return ARCGrid(new_grid)

    def compress(self) -> 'ARCGrid':
        """Attempts to compress the grid by finding repeating patterns (simple RLE for rows/cols)."""
        # This is a placeholder for a more sophisticated compression primitive.
        # For ARC, compression often means finding a smaller repeating unit.
        # A simple approach could be to find the smallest repeating row/column pattern.
        # For now, let's just return the original grid, or a simplified version.
        # A more advanced version would involve finding a minimal tile that can reconstruct the grid.
        return self.copy()

    def to_json(self) -> List[List[int]]:
        return self.to_array()

    @classmethod
    def from_json(cls, data: List[List[int]]) -> 'ARCGrid':
        return cls.from_array(data)
