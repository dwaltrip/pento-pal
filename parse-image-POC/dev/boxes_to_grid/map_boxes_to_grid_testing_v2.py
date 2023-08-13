from collections import namedtuple
from parse_image.boxes_to_grid.pieces import PIECES_BY_NAME

import rich
from rich.console import Console
from rich.style import Style

from settings import CLASS_MAPS, CLASS_NAMES, GRID
from parse_image.boxes_to_grid.map_boxes_to_grid import (
    PieceBox,
    get_puzzle_grid_from_piece_boxes,
)


rich_console = Console()
palette = rich.color.EIGHT_BIT_PALETTE
tui_colors_for_piece = dict()

for name, hex_color in CLASS_MAPS.name_to_color.items():
    if name == 'i':
        color_triplet = palette[243] # black won't show in terminal
    else:
        color_triplet = palette[palette.match(rich.color.parse_rgb_hex(hex_color[1:]))]
    tui_colors_for_piece[name] = rich.color.Color.from_triplet(color_triplet)


def print_puzzle_grid_state(puzzle_grid_state):
    for row in puzzle_grid_state:
        for cell in row:
            if cell:
                color = tui_colors_for_piece[cell]
                rich_console.print(cell, style=Style(color=color), end=' ')
            else:
                print('_', end=' ')
        print()


if __name__ == '__main__':
    Point = namedtuple('Point', ['y', 'x'])
    piece_boxes = [
        PieceBox(piece_type="p", top_left=Point(0, 0), height=3, width=2),
        PieceBox(piece_type="y", top_left=Point(0, 1), height=2, width=4),
        PieceBox(piece_type="n", top_left=Point(0, 4), height=4, width=2),
        PieceBox(piece_type="f", top_left=Point(1, 1), height=3, width=3),
        PieceBox(piece_type="l", top_left=Point(2, 4), height=4, width=2),
        PieceBox(piece_type="i", top_left=Point(3, 0), height=5, width=1),
        PieceBox(piece_type="x", top_left=Point(3, 2), height=3, width=3),
        PieceBox(piece_type="w", top_left=Point(4, 1), height=3, width=3),
        PieceBox(piece_type="t", top_left=Point(6, 1), height=3, width=3),
        PieceBox(piece_type="z", top_left=Point(6, 3), height=3, width=3),
        PieceBox(piece_type="v", top_left=Point(7, 3), height=3, width=3),
        PieceBox(piece_type="u", top_left=Point(8, 0), height=2, width=3)
    ]
    puzzle_grid = get_puzzle_grid_from_piece_boxes(piece_boxes)

    print()
    print('----- puzzle_grid -----')
    print_puzzle_grid_state(puzzle_grid)
