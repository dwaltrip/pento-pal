from parse_image.boxes_to_grid.map_boxes_to_grid import *
from parse_image.boxes_to_grid.pieces import parse_piece_string_to_grid


MAIN_GRID = SimpleNamespace(height=10, width=6)

class AlignedPieceBB:
    def __init__(self, name, top_left, grid):
        self.name = name
        self.top_left = top_left
        self.grid = grid 
        self.height = len(grid)
        self.width = len(grid[0])

        tl = self.top_left
        assert 0 <= tl.y < MAIN_GRID.height, f'Invalid top_y: {tl.y}'
        assert 0 <= tl.x < MAIN_GRID.width, f'Invalid top_x: {tl.x}'
    
    def clone(self):
        return self.__class__(
            name=self.name,
            top_left=self.top_left,
            grid=self.grid,
        )


Point = namedtuple('Point', ['y', 'x'])

if __name__ == '__main__':
    parse_piece = parse_piece_string_to_grid
    test_data = [
        AlignedPieceBB('p', top_left=Point(0, 0), grid=parse_piece('''
            | ■   |
            | ■ ■ |
            | ■ ■ |
        ''')),
        AlignedPieceBB('y', top_left=Point(0, 1), grid=parse_piece('''
            | ■ ■ ■ ■ |
            |     ■   |
        ''')),
        AlignedPieceBB('n', top_left=Point(0, 4), grid=parse_piece('''
            |   ■ |
            | ■ ■ |
            | ■   |
            | ■   |
        ''')),
        AlignedPieceBB('f', top_left=Point(1, 1), grid=parse_piece('''
            |   ■   |
            |   ■ ■ |
            | ■ ■   |
        ''')),
        AlignedPieceBB('l', top_left=Point(2, 4), grid=parse_piece('''
            |   ■ |
            |   ■ |
            |   ■ |
            | ■ ■ |
        ''')),
        AlignedPieceBB('i', top_left=Point(3, 0), grid=parse_piece('''
            | ■ |
            | ■ |
            | ■ |
            | ■ |
            | ■ |
        ''')),
        AlignedPieceBB('x', top_left=Point(3, 3), grid=parse_piece('''
            |   ■   |
            | ■ ■ ■ |
            |   ■   |
        ''')),
        AlignedPieceBB('w', top_left=Point(4, 1), grid=parse_piece('''
            | ■     |
            | ■ ■   |
            |   ■ ■ |
        ''')),
        AlignedPieceBB('t', top_left=Point(6, 1), grid=parse_piece('''
            | ■     |
            | ■ ■ ■ |
            | ■     |
        ''')),
        AlignedPieceBB('z', top_left=Point(6, 4), grid=parse_piece('''
            |   ■ ■ |
            |   ■   |
            | ■ ■   |
        ''')),
        AlignedPieceBB('v', top_left=Point(7, 4), grid=parse_piece('''
            |     ■ |
            |     ■ |
            | ■ ■ ■ |
        ''')),
        AlignedPieceBB('u', top_left=Point(8, 0), grid=parse_piece('''
            | ■   ■ |
            | ■ ■ ■ |
        ''')),

    ]
