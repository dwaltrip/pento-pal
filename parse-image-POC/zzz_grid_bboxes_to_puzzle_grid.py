from collections import namedtuple
from parse_image.boxes_to_grid.pieces import PIECES_BY_NAME

Point = namedtuple('Point', ['y', 'x'])
Orientation = namedtuple('Orientation', ['piece_name', 'height', 'width', 'grid'])

def get_orientations_for_piece(piece_name, height, width):
    piece = PIECES_BY_NAME[piece_name]
    return [
        Orientation(
            piece_name=piece_name,
            height=height,
            width=width,
            grid=grid,
        )
        for grid in piece.get_variant_grids_by_bbox_size(height, width)
    ]


class Piece:
    def __init__(self, name, top_left, orientation):
        self.name = name
        self.top_left = top_left
        self.orientation = orientation

    def __repr__(self):
        attrs = ' '.join([
            f'name={self.name},',
            f'top_left={self.top_left},',
            f'grid={self.orientation.grid}',
        ])
        return f'Piece({attrs})'
            

class TreeNode:
    def __init__(self,
        piece_type,
        bbox_info,
        puzzle_grid_state,
        orientation,
        parent=None,
    ):
        self.piece_type = piece_type
        self.bbox_info = bbox_info
        self.puzzle_grid_state = puzzle_grid_state
        self.orientation = orientation
        self.parent = parent
        self.children = []
        
    def add_child(self, child):
        self.children.append(child)


def try_to_get_filled_grid_from_bbox_info(bboxes):
    bboxes = bboxes.copy()
    # Sort bounding boxes by top_left point
    bboxes.sort(key=lambda bbox: (bbox['top_left'].y, bbox['top_left'].x))
    
    # Initialize puzzle grid
    puzzle_grid_state = [
        [None for _ in range(6)]
        for _ in range(10)
    ]
    
    # Initialize tree with a blank root node
    root = TreeNode(None, None, puzzle_grid_state, None)
    nodes_to_check = [root]

    # Iterate through nodes_to_check
    while nodes_to_check:
        current_node = nodes_to_check.pop(0)
        print('-------------')
        print('current_node:', current_node.piece_type)

        puzzle_grid_state = current_node.puzzle_grid_state

        placed_pieces = set([
            piece_name 
            for row in puzzle_grid_state for piece_name in row
            if piece_name
        ])
        print('placed_pieces:', (','.join(placed_pieces) if placed_pieces else None))

        if len(placed_pieces) == 2:
            print('------------------------------------')
            print('puzzle_grid_state:\n' + '\n'.join([
                ''.join([(cell or '_') for cell in row])
                for row in puzzle_grid_state
            ]))
            print('bboxes:', *bboxes, sep='\n\t')
            print()

        # Find the next nearest piece
        current_piece = None
        current_piece = next(
            (bbox for bbox in bboxes if bbox['name'] not in placed_pieces),
            None,
        )
        print('current_piece:', current_piece)

        # If all pieces placed, this is a solution
        if current_piece is None:
            print()
            print('SOLUTION FOUND!')
            print()
            print('puzzle_grid_state:\n' + '\n'.join([
                ' '.join([(cell or '_') for cell in row])
                for row in puzzle_grid_state
            ]))
            break
        
        # Iterate over each possible orientation
        possible_orientations = get_orientations_for_piece(
            current_piece['name'],
            current_piece['height'],
            current_piece['width'],
        )
        for orientation in possible_orientations:
            print('\torientation:', orientation)
            
            # Check if the orientation is valid and update the puzzle grid
            if is_valid_orientation(puzzle_grid_state, current_piece, orientation):
                new_puzzle_grid_state = [row.copy() for row in current_node.puzzle_grid_state]
                update_puzzle_grid(new_puzzle_grid_state, current_piece, orientation)
                new_node = TreeNode(
                    current_piece['name'],
                    current_piece,
                    new_puzzle_grid_state,
                    orientation=orientation,
                    parent=current_node,
                )
                current_node.add_child(new_node)
                nodes_to_check.append(new_node)
            # else:

    # Walk up the tree to collect the result
    result_pieces = []
    while current_node.parent:
        result_pieces.append(Piece(
            current_node.piece_type,
            current_node.bbox_info['top_left'],
            current_node.orientation,
        ))
        current_node = current_node.parent
    result_pieces.reverse()

    return result_pieces  # Return the list of 12 pieces with specific orientation


def is_valid_orientation(puzzle_grid_state, piece, orientation):
    top_left = piece['top_left']
    # print('-- is_valid_orientation --',
    #     'top_left:', piece['top_left'],
    #     f'puzzle_grid[tl.y][tl.x]: {puzzle_grid_state[top_left_y][top_left_x]}',
    #     f'orientation.size: ({orientation.height}, {orientation.width})',
    #     sep='\n\t',
    # )
    for row in range(orientation.height):
        for col in range(orientation.width):
            if (
                orientation.grid[row][col] and
                (puzzle_grid_state[top_left.y + row][top_left.x + col] is not None)
            ):
                return False
    # print('\t\t', 'valid orientation!', sep='')
    return True


def update_puzzle_grid(puzzle_grid_state, piece, orientation):
    # print('-- update_puzzle_grid --')
    # print('\tpiece:', piece)
    # print('\tpuzzle_grid_state (before):\n' + '\n'.join([
    #     ''.join([(cell or '_') for cell in row])
    #     for row in puzzle_grid_state
    # ]))
    top_left = piece['top_left']
    for row in range(orientation.height):
        for col in range(orientation.width):
            if orientation.grid[row][col]:
                puzzle_grid_state[top_left.y + row][top_left.x + col] = piece['name']
    # print('\tpuzzle_grid_state (after):\n' + '\n'.join([
    #     ''.join([(cell or '_') for cell in row])
    #     for row in puzzle_grid_state
    # ]))
    


if __name__ == '__main__':

    bboxes = [
        {
            "name": "p",
            "top_left": [0, 0],
            "height": 3,
            "width": 2
        },
        {
            "name": "y",
            "top_left": [0, 1],
            "height": 2,
            "width": 4
        },
        {
            "name": "n",
            "top_left": [ 0, 4 ],
            "height": 4,
            "width": 2
        },
        {
            "name": "f",
            "top_left": [ 1, 1 ],
            "height": 3,
            "width": 3
        },
        {
            "name": "l",
            "top_left": [ 2, 4 ],
            "height": 4,
            "width": 2
        },
        {
            "name": "i",
            "top_left": [ 3, 0 ],
            "height": 5,
            "width": 1
        },
        {
            "name": "x",
            "top_left": [ 3, 2 ],
            "height": 3,
            "width": 3
        },
        {
            "name": "w",
            "top_left": [ 4, 1 ],
            "height": 3,
            "width": 3
        },
        {
            "name": "t",
            "top_left": [ 6, 1 ],
            "height": 3,
            "width": 3
        },
        {
            "name": "z",
            "top_left": [ 6, 3 ],
            "height": 3,
            "width": 3
        },
        {
            "name": "v",
            "top_left": [ 7, 3 ],
            "height": 3,
            "width": 3
        },
        {
            "name": "u",
            "top_left": [ 8, 0 ],
            "height": 2,
            "width": 3
        }
    ]
    for box in bboxes:
        y, x = box['top_left']
        box['top_left'] = Point(y=y, x=x)

    puzzle_grid = try_to_get_filled_grid_from_bbox_info(bboxes)
    print()
    print('----- puzzle_grid -----')
    print()
    for piece in puzzle_grid:
        print(piece)

