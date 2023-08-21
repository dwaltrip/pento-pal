from settings import CLASS_NAMES, GRID


class TreeNode:
    def __init__(self, parent=None, data=None):
        self.parent = parent
        self.children = []
        self.data = data or {}

    is_root = property(lambda self: self.parent is None)

    def add_child(self, child):
        self.children.append(child)


# TODO: Handle the case where no solution can be found.
def get_puzzle_grid_from_piece_boxes(piece_grid_boxes):
    """ Given list of PieceGridBox, return solved puzzle grid if possible."""
    boxes = sorted(
        piece_grid_boxes,
        # sort by by row first, the column
        key=lambda box: (box.top_left.row, box.top_left.col),
    )

    solved_puzzle_grid = None
    
    empty_puzzle_grid = [
        [None for _ in range(GRID.width)]
        for _ in range(GRID.height)
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
            (box for box in boxes if box.piece_type not in placed_pieces),
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
                (puzzle_grid_state[top_left.row + row][top_left.col + col] is not None)
            ):
                return False
    return True


def update_puzzle_grid(puzzle_grid_state, piece_box, orientation):
    piece_type, top_left = piece_box.piece_type, piece_box.top_left
    puzzle_grid_state = [row.copy() for row in puzzle_grid_state] # make copy

    for row in range(orientation.height):
        for col in range(orientation.width):
            if orientation.grid[row][col]:
                puzzle_grid_state[top_left.row + row][top_left.col + col] = piece_type
    return puzzle_grid_state