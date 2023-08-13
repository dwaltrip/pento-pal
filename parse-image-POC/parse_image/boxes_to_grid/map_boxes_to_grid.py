from collections import namedtuple

from settings import CLASS_MAPS, CLASS_NAMES, GRID as PUZZLE_GRID
from parse_image.boxes_to_grid.pieces import Piece, PIECES_BY_NAME


class PieceBox:
    """ Bounding Box for a piece, aligned to the puzzle grid """

    def __init__(self, piece_type, top_left, height, width):
        self.piece_type = piece_type
        self.piece = PIECES_BY_NAME[self.piece_type]

        self.top_left = top_left
        self.height = height
        self.width = width

        tl = self.top_left
        assert 0 <= tl.y < PUZZLE_GRID.height, f'Invalid top_y: {tl.y}'
        assert 0 <= tl.x < PUZZLE_GRID.width, f'Invalid top_x: {tl.x}'

    def print_grid(self, *args, **kwargs):
        return self.piece.print_grid(*args, **kwargs)
    
    @property
    def possible_orientations(self):
        return self.piece.get_orientations_by_bbox_size(
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

def fit_boxes_to_grid(boxes):
    return 


class TreeNode:
    def __init__(self, parent=None, data=None):
        self.parent = parent
        self.children = []
        self.data = data or {}

    is_root = property(lambda self: self.parent is None)

    def add_child(self, child):
        self.children.append(child)


# TODO: Handle the case where no solution can be found.
def get_puzzle_grid_from_piece_boxes(piece_boxes):
    """ Given list of bounding boxes, return solved puzzle grid if possible."""
    solved_puzzle_grid = None
    piece_boxes = sorted(piece_boxes, key=lambda box: box.top_left)
    
    empty_puzzle_grid = [
        [None for _ in range(PUZZLE_GRID.width)]
        for _ in range(PUZZLE_GRID.height)
    ]
    root = TreeNode(data={ 'puzzle_grid_state': empty_puzzle_grid })
    nodes_to_check = [root]

    while nodes_to_check:
        current_node = nodes_to_check.pop(0)
        puzzle_grid_state = current_node.data['puzzle_grid_state']
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

        for orientation in current_box.possible_orientations:
            # Check if the orientation is valid and update the puzzle grid
            if is_valid_orientation(puzzle_grid_state, current_box, orientation):
                new_puzzle_grid_state = update_puzzle_grid(
                    puzzle_grid_state, current_box, orientation,
                )
                new_node = TreeNode(
                    parent=current_node,
                    data={ 'puzzle_grid_state': new_puzzle_grid_state },
                )
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
