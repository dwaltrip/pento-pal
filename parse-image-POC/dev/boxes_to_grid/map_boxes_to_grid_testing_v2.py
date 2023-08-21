from collections import namedtuple
from parse_image.boxes_to_grid.pieces import PIECES_BY_NAME

from utils.print_puzzle_grid import print_puzzle_grid
from parse_image.boxes_to_grid.map_boxes_to_grid import (
    PieceBox,
    get_puzzle_grid_from_piece_boxes,
)


if __name__ == '__main__':
    Point = namedtuple('Point', ['x', 'y'])
    piece_boxes = [
        PieceBox(piece_type="p", top_left=Point(0, 0), height=3, width=2),
        PieceBox(piece_type="y", top_left=Point(1, 0), height=2, width=4),
        PieceBox(piece_type="n", top_left=Point(4, 0), height=4, width=2),
        PieceBox(piece_type="f", top_left=Point(1, 1), height=3, width=3),
        PieceBox(piece_type="l", top_left=Point(4, 2), height=4, width=2),
        PieceBox(piece_type="i", top_left=Point(0, 3), height=5, width=1),
        PieceBox(piece_type="x", top_left=Point(2, 3), height=3, width=3),
        PieceBox(piece_type="w", top_left=Point(1, 4), height=3, width=3),
        PieceBox(piece_type="t", top_left=Point(1, 6), height=3, width=3),
        PieceBox(piece_type="z", top_left=Point(3, 6), height=3, width=3),
        PieceBox(piece_type="v", top_left=Point(3, 7), height=3, width=3),
        PieceBox(piece_type="u", top_left=Point(0, 8), height=2, width=3)
    ]
    puzzle_grid = get_puzzle_grid_from_piece_boxes(piece_boxes)

    print()
    print('----- puzzle_grid -----')
    print_puzzle_grid(puzzle_grid)
