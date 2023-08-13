from collections import namedtuple
from parse_image.boxes_to_grid.pieces import PIECES_BY_NAME

import rich
from rich.console import Console
from rich.style import Style

from settings import CLASS_MAPS, CLASS_NAMES, GRID


rich_console = Console()
palette = rich.color.EIGHT_BIT_PALETTE
tui_colors_for_piece = dict()

for name, hex_color in CLASS_MAPS.name_to_color.items():
    if name == 'i':
        color_triplet = palette[243] # black won't show in terminal
    else:
        color_triplet = palette[palette.match(rich.color.parse_rgb_hex(hex_color[1:]))]
    tui_colors_for_piece[name] = rich.color.Color.from_triplet(color_triplet)



Point = namedtuple('Point', ['y', 'x'])
Orientation = namedtuple('Orientation', ['piece_type', 'height', 'width', 'grid'])

def get_orientations_for_piece(piece_type, height, width):
    piece = PIECES_BY_NAME[piece_type]
    return [
        Orientation(
            piece_type=piece_type,
            height=height,
            width=width,
            grid=grid,
        )
        for grid in piece.get_variant_grids_by_bbox_size(height, width)
    ]


class TreeNode:
    def __init__(self, puzzle_grid_state, parent=None):
        self.puzzle_grid_state = puzzle_grid_state
        self.parent = parent
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)


def try_to_get_filled_grid_from_piece_boxes(piece_boxes):
    """
    Given a list of bounding boxes, return a solved puzzle grid if possible.
    """
    solved_puzzle_grid = None
    piece_boxes = sorted(piece_boxes, key=lambda box: box.top_left)
    
    puzzle_grid_state = [
        [None for _ in range(GRID.width)]
        for _ in range(GRID.height)
    ]
    root = TreeNode(puzzle_grid_state)
    nodes_to_check = [root]

    while nodes_to_check:
        current_node = nodes_to_check.pop(0)
        puzzle_grid_state = current_node.puzzle_grid_state
        placed_pieces = set([
            piece_type 
            for row in puzzle_grid_state for piece_type in row
            if piece_type 
        ])

        # If all pieces are placed, this is a solution
        if placed_pieces == set(CLASS_NAMES):
            solved_puzzle_grid = puzzle_grid_state
            break

        # Find the next nearest bounding box
        current_box = next(
            (box for box in piece_boxes if box.piece_type not in placed_pieces),
            None,
        )

        possible_orientations = get_orientations_for_piece(
            current_box.piece_type,
            current_box.height,
            current_box.width,
        )
        for orientation in possible_orientations:
            # Check if the orientation is valid and update the puzzle grid
            if is_valid_orientation(puzzle_grid_state, current_box, orientation):
                new_puzzle_grid_state = update_puzzle_grid(
                    puzzle_grid_state, current_box, orientation,
                )
                new_node = TreeNode(new_puzzle_grid_state, parent=current_node)
                current_node.add_child(new_node)
                nodes_to_check.append(new_node)

    return solved_puzzle_grid


def is_valid_orientation(puzzle_grid_state, piece_box, orientation):
    top_left = piece_box.top_left
    for row in range(orientation.height):
        for col in range(orientation.width):
            if (
                orientation.grid[row][col] and
                (puzzle_grid_state[top_left.y + row][top_left.x + col] is not None)
            ):
                return False
    return True


def update_puzzle_grid(puzzle_grid_state, piece_box, orientation):
    piece_type, top_left = piece_box.piece_type, piece_box.top_left
    puzzle_grid_state = [row.copy() for row in puzzle_grid_state] # make copy

    for row in range(orientation.height):
        for col in range(orientation.width):
            if orientation.grid[row][col]:
                puzzle_grid_state[top_left.y + row][top_left.x + col] = piece_type
    return puzzle_grid_state


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
    bboxes = [
        { "name": "p", "top_left": [0, 0], "height": 3, "width": 2 },
        { "name": "y", "top_left": [0, 1], "height": 2, "width": 4 },
        { "name": "n", "top_left": [ 0, 4 ], "height": 4, "width": 2 },
        { "name": "f", "top_left": [ 1, 1 ], "height": 3, "width": 3 },
        { "name": "l", "top_left": [ 2, 4 ], "height": 4, "width": 2 },
        { "name": "i", "top_left": [ 3, 0 ], "height": 5, "width": 1 },
        { "name": "x", "top_left": [ 3, 2 ], "height": 3, "width": 3 },
        { "name": "w", "top_left": [ 4, 1 ], "height": 3, "width": 3 },
        { "name": "t", "top_left": [ 6, 1 ], "height": 3, "width": 3 },
        { "name": "z", "top_left": [ 6, 3 ], "height": 3, "width": 3 },
        { "name": "v", "top_left": [ 7, 3 ], "height": 3, "width": 3 },
        { "name": "u", "top_left": [ 8, 0 ], "height": 2, "width": 3 }
    ]

    PieceBox = namedtuple('PieceBox', ['piece_type', 'top_left', 'height', 'width'])
    def make_piece_box(bbox):
        y, x = bbox['top_left']
        return PieceBox(
            piece_type=bbox['name'],
            top_left=Point(y=y, x=x),
            height=bbox['height'],
            width=bbox['width'],
        )
    piece_boxes = [ make_piece_box(bbox) for bbox in bboxes ]

    puzzle_grid = try_to_get_filled_grid_from_piece_boxes(piece_boxes)
    print()
    print('----- puzzle_grid -----')
    print_puzzle_grid_state(puzzle_grid)
