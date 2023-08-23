from dataclasses import dataclass, field


@dataclass
class Point:
    x: float
    y: float


@dataclass
class GridCoord:
    row: int
    col: int
