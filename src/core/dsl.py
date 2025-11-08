# SPDX-License-Identifier: MPL-2.0
from abc import ABC, abstractmethod
from typing import List, Tuple, Any, Dict, Type, Callable
from src.core.grid import ARCGrid
import inspect

class Primitive(ABC):
    """Abstract base class for all DSL primitives."""
    name: str
    arg_types: Dict[str, Type]

    @abstractmethod
    def apply(self, grid: ARCGrid, **kwargs) -> ARCGrid:
        """Applies the primitive to a grid."""
        pass

    def __repr__(self) -> str:
        return f"<{self.name}>"

    def to_json(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Converts the primitive and its arguments to a JSON-serializable dictionary."""
        json_args = {}
        for k, v in args.items():
            if isinstance(v, ARCGrid):
                json_args[k] = v.to_json()
            else:
                json_args[k] = v
        return {"name": self.name, "args": json_args}

    @classmethod
    def from_json(cls, data: Dict[str, Any], dsl_registry: Dict[str, Type['Primitive']]) -> Tuple['Primitive', Dict[str, Any]]:
        primitive_name = data["name"]
        primitive_cls = dsl_registry[primitive_name]
        args = {}
        for k, v in data["args"].items():
            if primitive_cls.arg_types.get(k) == ARCGrid:
                args[k] = ARCGrid.from_json(v)
            else:
                args[k] = v
        return primitive_cls(), args


class Paint(Primitive):
    name = "paint"
    arg_types = {"r": int, "c": int, "color": int}

    def apply(self, grid: ARCGrid, r: int, c: int, color: int) -> ARCGrid:
        return grid.set_pixel(r, c, color)

class PaintRectangle(Primitive):
    name = "paint_rectangle"
    arg_types = {"r_start": int, "c_start": int, "r_end": int, "c_end": int, "color": int}

    def apply(self, grid: ARCGrid, r_start: int, c_start: int, r_end: int, c_end: int, color: int) -> ARCGrid:
        return grid.paint_rectangle(r_start, c_start, r_end, c_end, color)

class Copy(Primitive):
    name = "copy"
    arg_types = {"source_grid": ARCGrid, "r_offset": int, "c_offset": int}

    def apply(self, grid: ARCGrid, source_grid: ARCGrid, r_offset: int, c_offset: int) -> ARCGrid:
        return grid.overlay(source_grid, r_offset, c_offset)

class Rotate(Primitive):
    name = "rotate"
    arg_types = {"k": int}

    def apply(self, grid: ARCGrid, k: int) -> ARCGrid:
        return grid.rotate(k)

class ReflectHorizontal(Primitive):
    name = "reflect_horizontal"
    arg_types = {}

    def apply(self, grid: ARCGrid) -> ARCGrid:
        return grid.reflect_horizontal()

class ReflectVertical(Primitive):
    name = "reflect_vertical"
    arg_types = {}

    def apply(self, grid: ARCGrid) -> ARCGrid:
        return grid.reflect_vertical()

class FloodFill(Primitive):
    name = "flood_fill"
    arg_types = {"r": int, "c": int, "replacement_color": int}

    def apply(self, grid: ARCGrid, r: int, c: int, replacement_color: int) -> ARCGrid:
        return grid.flood_fill(r, c, replacement_color)

class CountColor(Primitive):
    name = "count_color"
    arg_types = {"color": int}

    def apply(self, grid: ARCGrid, color: int) -> int:
        """This primitive returns an int, not a grid. Special handling needed in Program execution."""
        return grid.count_color(color)

class Compress(Primitive):
    name = "compress"
    arg_types = {}

    def apply(self, grid: ARCGrid) -> ARCGrid:
        return grid.compress()

class RelationalCompose(Primitive):
    name = "relational_compose"
    arg_types = {"op1": str, "op2": str, "relation": str}

    def apply(self, grid: ARCGrid, op1: str, op2: str, relation: str) -> ARCGrid:
        """Placeholder for a complex relational composition primitive.
        This would typically involve applying op1, then op2 based on a relation
        (e.g., 'align_top_left', 'overlap_center').
        For now, it's a no-op or simple pass-through.
        """
        # In a real implementation, op1 and op2 would be sub-programs or primitives
        # and 'relation' would define how their outputs are combined or aligned.
        # This is highly abstract and would require a more complex DSL structure.
        return grid.copy()


class DSL:
    """Manages the Domain Specific Language primitives."""
    def __init__(self):
        self._primitives: Dict[str, Type[Primitive]] = {}
        self._register_default_primitives()

    def _register_default_primitives(self):
        self.register_primitive(Paint)
        self.register_primitive(PaintRectangle)
        self.register_primitive(Copy)
        self.register_primitive(Rotate)
        self.register_primitive(ReflectHorizontal)
        self.register_primitive(ReflectVertical)
        self.register_primitive(FloodFill)
        self.register_primitive(CountColor)
        self.register_primitive(Compress)
        self.register_primitive(RelationalCompose)

    def register_primitive(self, primitive_cls: Type[Primitive]):
        if not issubclass(primitive_cls, Primitive) or inspect.isabstract(primitive_cls):
            raise ValueError(f"Class {primitive_cls.__name__} must be a concrete subclass of Primitive.")
        self._primitives[primitive_cls.name] = primitive_cls

    def get_primitive(self, name: str) -> Type[Primitive]:
        if name not in self._primitives:
            raise ValueError(f"Primitive '{name}' not found in DSL.")
        return self._primitives[name]

    def get_all_primitives(self) -> Dict[str, Type[Primitive]]:
        return self._primitives

    def get_primitive_names(self) -> List[str]:
        return list(self._primitives.keys())


class ProgramStep:
    """Represents a single step in a program, consisting of a primitive and its arguments."""
    def __init__(self, primitive: Primitive, args: Dict[str, Any]):
        self.primitive = primitive
        self.args = args

    def __repr__(self) -> str:
        arg_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"{self.primitive.name}({arg_str})"

    def to_json(self) -> Dict[str, Any]:
        return self.primitive.to_json(self.args)

    @classmethod
    def from_json(cls, data: Dict[str, Any], dsl_registry: Dict[str, Type[Primitive]]) -> 'ProgramStep':
        primitive, args = Primitive.from_json(data, dsl_registry)
        return cls(primitive, args)


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
                return step.primitive.apply(current_grid, **step.args)
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
        """Calculates the Minimum Description Length of the program."""
        # This is a simplified MDL calculation. A more robust one would consider
        # argument complexity, primitive frequency, and compression ratios.
        mdl = 0.0
        for step in self.steps:
            mdl += 1.0 # Cost for the primitive itself
            for arg_name, arg_value in step.args.items():
                if isinstance(arg_value, int):
                    # Cost for integer arguments (e.g., log2(value) or fixed cost)
                    mdl += 0.1 # Small fixed cost for now
                elif isinstance(arg_value, ARCGrid):
                    # Cost for grid arguments (e.g., size, complexity)
                    mdl += (arg_value.height * arg_value.width) / 100.0 # Example: proportional to grid size
                elif isinstance(arg_value, str):
                    mdl += len(arg_value) * 0.05 # Cost for string arguments
        return mdl
