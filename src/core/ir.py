import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Any, Optional
from src.core.grid import ARCGrid
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation

class ObjectProposal:
    """Represents a proposed object within an ARCGrid."""
    def __init__(self, grid: ARCGrid, bounding_box: Tuple[int, int, int, int], mask: np.ndarray, color: int = 0):
        self.grid = grid # The original grid or a cropped view
        self.r_min, self.c_min, self.r_max, self.c_max = bounding_box
        self.mask = mask # Boolean mask of the object within its bounding box
        self.color = color # Dominant color or representative color of the object

        self.height = self.r_max - self.r_min
        self.width = self.c_max - self.c_min
        self.area = np.sum(mask)

    def get_cropped_grid(self) -> ARCGrid:
        """Returns an ARCGrid containing only this object, with background elsewhere."""
        cropped_array = np.where(self.mask, self.grid.grid[self.r_min:self.r_max, self.c_min:self.c_max], 0)
        return ARCGrid(cropped_array)

    def get_relative_coords(self) -> List[Tuple[int, int]]:
        """Returns coordinates of object pixels relative to its bounding box top-left."""
        rows, cols = np.where(self.mask)
        return list(zip(rows, cols))

    def __repr__(self) -> str:
        return f"ObjectProposal(color={self.color}, bbox={self.bounding_box}, area={self.area})"

    @property
    def bounding_box(self) -> Tuple[int, int, int, int]:
        return (self.r_min, self.c_min, self.r_max, self.c_max)


class SymmetryDetector:
    """Detects symmetries within an ARCGrid or ObjectProposal."""
    @staticmethod
    def detect_reflection(grid: ARCGrid) -> Dict[str, bool]:
        symmetries = {"horizontal": False, "vertical": False, "diagonal_tl_br": False, "diagonal_tr_bl": False}
        if grid.height == 0 or grid.width == 0: return symmetries

        # Horizontal reflection (left-right)
        if np.array_equal(grid.grid, np.fliplr(grid.grid)):
            symmetries["horizontal"] = True

        # Vertical reflection (up-down)
        if np.array_equal(grid.grid, np.flipud(grid.grid)):
            symmetries["vertical"] = True

        # Diagonal reflection (top-left to bottom-right)
        # Requires square grid
        if grid.height == grid.width:
            if np.array_equal(grid.grid, grid.grid.T):
                symmetries["diagonal_tl_br"] = True

            # Diagonal reflection (top-right to bottom-left)
            if np.array_equal(grid.grid, np.fliplr(grid.grid).T):
                symmetries["diagonal_tr_bl"] = True

        return symmetries

    @staticmethod
    def detect_rotation(grid: ARCGrid) -> Dict[int, bool]:
        rotations = {90: False, 180: False, 270: False}
        if grid.height == 0 or grid.width == 0: return rotations

        # 90 degree rotation
        if np.array_equal(grid.grid, np.rot90(grid.grid, k=-1)): # -1 for 90 deg clockwise
            rotations[90] = True

        # 180 degree rotation
        if np.array_equal(grid.grid, np.rot90(grid.grid, k=-2)): # -2 for 180 deg clockwise
            rotations[180] = True

        # 270 degree rotation
        if np.array_equal(grid.grid, np.rot90(grid.grid, k=-3)): # -3 for 270 deg clockwise
            rotations[270] = True

        return rotations


class ARCIntermediateRepresentation:
    """A typed intermediate representation of an ARC grid, including object proposals and symmetries."""
    def __init__(self, grid: ARCGrid, background_color: int = 0):
        self.original_grid = grid
        self.background_color = background_color
        self.height, self.width = grid.height, grid.width

        self.object_proposals: List[ObjectProposal] = self._generate_object_proposals()
        self.grid_symmetries: Dict[str, Any] = self._detect_grid_symmetries()
        self.object_graphs: List[nx.Graph] = self._generate_object_graphs()

    def _generate_object_proposals(self) -> List[ObjectProposal]:
        """Generates object proposals using connected components."""
        # Label connected components of non-background pixels
        labeled_array = label(self.original_grid.grid != self.background_color, connectivity=1)
        regions = regionprops(labeled_array)

        proposals = []
        for region in regions:
            # region.bbox gives (min_row, min_col, max_row, max_col)
            r_min, c_min, r_max, c_max = region.bbox
            mask = region.image # Boolean mask of the object within its local bounding box

            # Determine dominant color within the object
            object_pixels = self.original_grid.grid[r_min:r_max, c_min:c_max][mask]
            if object_pixels.size > 0:
                colors, counts = np.unique(object_pixels, return_counts=True)
                dominant_color = colors[np.argmax(counts)]
            else:
                dominant_color = 0 # Default if object is empty (shouldn't happen)

            proposals.append(ObjectProposal(self.original_grid, region.bbox, mask, dominant_color))
        return proposals

    def _detect_grid_symmetries(self) -> Dict[str, Any]:
        """Detects global symmetries of the grid."""
        symmetries = {
            "reflection": SymmetryDetector.detect_reflection(self.original_grid),
            "rotation": SymmetryDetector.detect_rotation(self.original_grid)
        }
        return symmetries

    def _generate_object_graphs(self) -> List[nx.Graph]:
        """Generates graph representations for each object proposal.
        Nodes are pixels, edges represent adjacency, attributes can be color, position.
        """
        object_graphs = []
        for obj_proposal in self.object_proposals:
            graph = nx.grid_2d_graph(obj_proposal.height, obj_proposal.width)
            # Remove nodes not part of the object mask
            for r in range(obj_proposal.height):
                for c in range(obj_proposal.width):
                    if not obj_proposal.mask[r, c]:
                        if (r, c) in graph:
                            graph.remove_node((r, c))
                    else:
                        # Add pixel color as node attribute
                        global_r, global_c = r + obj_proposal.r_min, c + obj_proposal.c_min
                        graph.nodes[(r, c)]['color'] = self.original_grid.get_pixel(global_r, global_c)
                        graph.nodes[(r, c)]['global_pos'] = (global_r, global_c)
            object_graphs.append(graph)
        return object_graphs

    def get_typed_features(self) -> Dict[str, Any]:
        """Returns a dictionary of typed features extracted from the IR."""
        features = {
            "grid_dimensions": (self.height, self.width),
            "unique_colors": self.original_grid.get_colors(),
            "num_objects": len(self.object_proposals),
            "grid_symmetries": self.grid_symmetries,
            "object_features": []
        }
        for obj in self.object_proposals:
            obj_features = {
                "bbox": obj.bounding_box,
                "area": obj.area,
                "dominant_color": obj.color,
                "symmetries": SymmetryDetector.detect_reflection(obj.get_cropped_grid()), # Object-level symmetries
                "aspect_ratio": obj.width / obj.height if obj.height > 0 else 0,
                "perimeter": self._calculate_perimeter(obj.mask)
            }
            features["object_features"].append(obj_features)
        return features

    def _calculate_perimeter(self, mask: np.ndarray) -> int:
        """Calculates the perimeter of a binary mask."""
        if mask.size == 0: return 0
        eroded_mask = binary_erosion(mask)
        perimeter_mask = mask ^ eroded_mask # XOR to get the boundary
        return np.sum(perimeter_mask)

    def __repr__(self) -> str:
        return (
            f"ARCIntermediateRepresentation(grid_dim={self.height}x{self.width}, "
            f"num_objects={len(self.object_proposals)}, "
            f"grid_symmetries={self.grid_symmetries['reflection']})")
