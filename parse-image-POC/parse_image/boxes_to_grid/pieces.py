from collections import defaultdict

EMPTY = 0
FILLED = 1
EMPTY_CHAR = ' '
FILLED_CHAR = '■'


# clockwise
def rotate_90(grid, n=1):
    n = n % 4
    for _ in range(n):
        grid = list(zip(*grid[::-1]))
    return grid

def grid_to_str(grid):
    return '\n'.join(
        ''.join(str(cell) for cell in row)
        for row in grid
    )

def dedupe_grids(grids):
    deduped = { grid_to_str(grid): grid for grid in grids }
    return list(deduped.values())


class Piece:
    def __init__(self, name, grid):
        self.name = name
        self.grid = grid
        self._variants_by_bbox_size = self._construct_all_orientations()
    
    height = property(lambda self: len(self.grid))
    width = property(lambda self: len(self.grid[0]))

    @property
    def all_variant_grids(self):
        return [
            variant_grid
            for list_of_grids in self._variants_by_bbox_size.values()
            for variant_grid in list_of_grids
        ]
    
    def get_variant_grids_by_bbox_size(self, height, width):
        return self._variants_by_bbox_size[(height, width)]
    
    @classmethod
    def _print_grid(cls, grid, prefix=''):
        for row in grid:
            row = [FILLED_CHAR if cell == FILLED else ' ' for cell in row]
            print((prefix or '') + ' '.join(row))

    def print_grid(self, prefix=''):
        self._print_grid(self.grid, prefix=prefix)
    
    def _construct_all_orientations(self):
        # this includes the original grid
        flipped_grid = [row[::-1] for row in self.grid]
        variant_grids = dedupe_grids([
            *[rotate_90(self.grid, n) for n in range(4)],
            *[rotate_90(flipped_grid, n) for n in range(4)],
        ])

        grids_by_bbox_size = defaultdict(list)
        for grid in variant_grids:
            height, width = len(grid), len(grid[0])
            grids_by_bbox_size[(height, width)].append(grid)
        return grids_by_bbox_size


    

PIECE_SHAPE_STRINGS = dict(
    f= '''
        |   ■ ■ |
        | ■ ■   |
        |   ■   |''',
    i= '''
        | ■ |
        | ■ |
        | ■ |
        | ■ |
        | ■ |''',
    l= '''
        | ■   |
        | ■   |
        | ■   |
        | ■ ■ |''',
    n= '''
        | ■   |
        | ■ ■ |
        |   ■ |
        |   ■ |''',
    p= '''
        | ■ ■ |
        | ■ ■ |
        | ■   |''',
    t= '''
        | ■ ■ ■ |
        |   ■   |
        |   ■   |''',
    u= '''
        | ■   ■ |
        | ■ ■ ■ |''',
    v= '''
        |     ■ |
        |     ■ |
        | ■ ■ ■ |''',
    w= '''
        |     ■ |
        |   ■ ■ |
        | ■ ■   |''',
    x= '''
        |   ■   |
        | ■ ■ ■ |
        |   ■   |''',
    y= '''
        |   ■ |
        | ■ ■ |
        |   ■ |
        |   ■ |''',
    z= '''
        | ■ ■   |
        |   ■   |
        |   ■ ■ |''',
)


def parse_piece_string_to_grid(piece_str):
    lines = piece_str.strip().split('\n')
    lines = [line.strip() for line in lines]
    assert len(set(len(line) for line in lines)) == 1, 'All lines must be the same length'

    grid = []

    for line in lines:
        first, *inner_str, last = line
        assert first == last == '|', 'Line must be bracketed with |'
        valid_chars = (FILLED_CHAR, EMPTY_CHAR)
        assert set(inner_str) == set(valid_chars), \
            f'Line must contain only {repr(valid_chars)}'

        row = []
        for i, c in enumerate(inner_str):
            if i % 2 == 0:
                assert c == ' ', 'Odd chars must be spaces'
            else:
                row.append(FILLED if c == FILLED_CHAR else EMPTY)

        grid.append(row)
    
    return grid
 

def parse_piece_strings(piece_strings):
    return {
        name: Piece(
            name=name,
            grid=parse_piece_string_to_grid(piece_str),
        )
        for name, piece_str in piece_strings.items()
    }


PIECES_BY_NAME = parse_piece_strings(PIECE_SHAPE_STRINGS)
PIECES = PIECES_BY_NAME.values()
