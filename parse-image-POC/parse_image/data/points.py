from collections import namedtuple
from dataclasses import dataclass, field


# Using namedtuple allows for splatting, unlike dataclass.
# See usage in `draw_bounding_boxes`.
# Wonder if there is a clean way to do this with dataclass?
Point = namedtuple('Point', ['x', 'y'])


@dataclass
class GridCoord:
    row: int
    col: int
