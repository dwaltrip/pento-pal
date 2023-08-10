from collections import namedtuple
from types import SimpleNamespace

from parse_image.boxes_to_grid.pieces import Piece, PIECES_BY_NAME


# TODO: clean this up
MAIN_GRID = SimpleNamespace(height=10, width=6)

# AlignedBox = namedtuple('AlignedBox', ['height', 'width', 'class'])
# class AlignedPieceBoundingBox:
class GridBoxForPiece:
    def __init__(self, name, top_left, height, width):
        self.name = name
        self.top_left = top_left
        self.height = height
        self.width = width
        self._canonical_piece = PIECES_BY_NAME[name]

        tl = self.top_left
        assert 0 <= tl.y < MAIN_GRID.height, f'Invalid top_y: {tl.y}'
        assert 0 <= tl.x < MAIN_GRID.width, f'Invalid top_x: {tl.x}'

    def print_grid(self, *args, **kwargs):
        return self._canonical_piece.print_grid(*args, **kwargs)
    
    def get_possible_variants(self):
        return self._canonical_piece.get_variants_by_bbox_size(
            height=self.height,
            width=self.width,
        )

    def clone(self):
        return self.__class__(
            name=self.name,
            top_left=self.top_left,
            height=self.height,
            width=self.width,
        )


def extract_grid_from_boxes(boxes):
    pass


def try_to_get_filled_grid_from_grid_boxes(grid_boxes):
    for box in grid_boxes:
        print()
        print(f'box: {box.name}, shape: ({box.height}, {box.width})')
        print('------------ variants:') 

        for i, variant_grid in enumerate(box.get_possible_variants()):
            print(f'variant {i+1}:')
            Piece._print_grid(variant_grid, prefix=f'\t')


def fit_boxes_to_grid(boxes):
    return 


