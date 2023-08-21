import rich
from rich.console import Console
from rich.style import Style

from settings import CLASS_MAPS, CLASS_NAMES, GRID


PALETTE = rich.color.EIGHT_BIT_PALETTE
tui_colors_for_piece = dict()

for name, hex_color in CLASS_MAPS.name_to_color.items():
    if name == 'i':
        color_triplet = PALETTE[243] # black won't show in terminal
    else:
        color_triplet = PALETTE[PALETTE.match(rich.color.parse_rgb_hex(hex_color[1:]))]
    tui_colors_for_piece[name] = rich.color.Color.from_triplet(color_triplet)


def print_puzzle_grid(puzzle_grid):
    console = Console()
    for row in puzzle_grid:
        for piece_type in row:
            if piece_type:
                color = tui_colors_for_piece[piece_type]
                console.print(piece_type, style=Style(color=color), end=' ')
            else:
                print('_', end=' ')
        print()