from collections import namedtuple
from dataclasses import dataclass

from pento_parser_poc.settings import CLASS_NAMES, GRID
from pento_parser_poc.pieces import PIECES_BY_NAME


Point = namedtuple('Point', ['x', 'y'])
GridCoord = namedtuple('GridCoord', ['col', 'row'])
FloatingRect = namedtuple('FloatingRect', ['height', 'width'])


@dataclass
class PixelGrid:
    rows = GRID.height
    cols = GRID.width

    top_left: Point
    bot_right: Point

    height = property(lambda self: self.bot_right.y - self.top_left.y)
    width = property(lambda self: self.bot_right.x - self.top_left.x)

    cell_size = property(lambda self: FloatingRect(
        height=self.height / GRID.height,
        width=self.width / GRID.width,
    ))


@dataclass
class PieceGridBox:
    """ Bounding Box for a piece, aligned to the puzzle grid """
    class_id: int
    top_left: Point
    height: int
    width: int

    piece_type = property(lambda self: CLASS_NAMES[self.class_id])
    piece = property(lambda self: PIECES_BY_NAME[self.piece_type])

    bot_right = property(lambda self: GridCoord(
        col=self.top_left.col + self.width,
        row=self.top_left.row + self.height,
    ))
 
    @property
    def possible_orientations(self):
        return self.piece.get_orientations_by_bbox_size(
            height=self.height,
            width=self.width,
        )

    def print_grid(self, *args, **kwargs):
        return self.piece.print_grid(*args, **kwargs)


def get_grid(boxes):
    return PixelGrid(
        top_left=Point(
            min(bb.top_left.x for bb in boxes),
            min(bb.top_left.y for bb in boxes),
        ),
        bot_right=Point(
            max(bb.bot_right.x for bb in boxes),
            max(bb.bot_right.y for bb in boxes),
        ),
    )


def bounding_boxes_to_grid_boxes(bounding_boxes):
    pixel_grid = get_grid(bounding_boxes)
    cell_size = pixel_grid.cell_size

    # Convert bounding boxes to grid boxes
    grid_boxes = []
    for bb in bounding_boxes:
        top_left = GridCoord(
            col=int(round((bb.top_left.x - pixel_grid.top_left.x) / cell_size.width)),
            row=int(round((bb.top_left.y - pixel_grid.top_left.y) / cell_size.height)),
        )
        grid_box = PieceGridBox(
            class_id=bb.class_id,
            top_left=top_left,
            # ---------------------------------------------------------------
            # TODO: We know this definitionally `class_id` / `piece_type`...
            # ---------------------------------------------------------------
            height=int(round(bb.height / cell_size.height)),
            width=int(round(bb.width / cell_size.width)),
        )
        grid_boxes.append(grid_box)

    return grid_boxes
