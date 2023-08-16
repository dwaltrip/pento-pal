from collections import namedtuple
import random
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from settings import CLASS_MAPS
from utils.color import hex_to_rgb
from parse_image.boxes_to_grid.map_boxes_to_grid import (
    GridBoxForPiece,
    try_to_get_filled_grid_from_grid_boxes,
)
from parse_image.boxes_to_grid.pieces import (
    parse_piece_string_to_grid as parse_piece,
    Piece,
    FILLED,
)


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

    def print_grid(self, prefix=''):
        return Piece._print_grid(self.grid, prefix=prefix)  

    def to_dict(self):
        return dict(
            name=self.name,
            top_left=(self.top_left.y, self.top_left.x),
            height=self.height,
            width=self.width,
        )
    
    def clone(self):
        return self.__class__(
            name=self.name,
            top_left=self.top_left,
            grid=self.grid,
        )


def aligned_boxes_to_puzzle_grid(aligned_boxes):
    """ Convert a list of aligned boxes to a puzzle grid """
    rows, cols = MAIN_GRID.height, MAIN_GRID.width
    puzzle_grid = [[None] * cols for _ in range(rows)]

    for aligned_box in aligned_boxes:
        top_left = aligned_box.top_left
        for i, row in enumerate(aligned_box.grid):
            for j, cell in enumerate(row):
                if cell == FILLED:
                    y = top_left.y + i
                    x = top_left.x + j
                    puzzle_grid[y][x] = aligned_box.name
    return puzzle_grid


def build_viz_img_for_puzzle_grid(puzzle_grid):
    rows, cols = MAIN_GRID.height, MAIN_GRID.width
    cell_size = 50
    cell_padding = 5
    padding_color = '#8a7a7a'

    img_height = (cell_size + cell_padding) * rows + cell_padding
    img_width = (cell_size + cell_padding) * cols + cell_padding
    viz_img = np.full(
        (img_height, img_width, 3),
        hex_to_rgb(padding_color),
        dtype=np.uint8,
    )

    for i in range(MAIN_GRID.height):
        for j in range(MAIN_GRID.width):
            top_left = dict(
                y = i * (cell_size + cell_padding) + cell_padding,
                x = j * (cell_size + cell_padding) + cell_padding,
            )
            y = (top_left['y'], top_left['y'] + cell_size)
            x = (top_left['x'], top_left['x'] + cell_size)
            color = hex_to_rgb(CLASS_MAPS.name_to_color[puzzle_grid[i][j]])
            viz_img[y[0]:y[1], x[0]:x[1]] = color
    return viz_img


Point = namedtuple('Point', ['x', 'y'])

if __name__ == '__main__':
    aligned_boxes_SOLVED = [
        AlignedPieceBB('p', top_left=Point(0, 0), grid=parse_piece('''
            | ■   |
            | ■ ■ |
            | ■ ■ |
        ''')),
        AlignedPieceBB('y', top_left=Point(1, 0), grid=parse_piece('''
            | ■ ■ ■ ■ |
            |     ■   |
        ''')),
        AlignedPieceBB('n', top_left=Point(4, 0), grid=parse_piece('''
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
        AlignedPieceBB('l', top_left=Point(4, 2), grid=parse_piece('''
            |   ■ |
            |   ■ |
            |   ■ |
            | ■ ■ |
        ''')),
        AlignedPieceBB('i', top_left=Point(0, 3), grid=parse_piece('''
            | ■ |
            | ■ |
            | ■ |
            | ■ |
            | ■ |
        ''')),
        AlignedPieceBB('x', top_left=Point(2, 3), grid=parse_piece('''
            |   ■   |
            | ■ ■ ■ |
            |   ■   |
        ''')),
        AlignedPieceBB('w', top_left=Point(1, 4), grid=parse_piece('''
            | ■     |
            | ■ ■   |
            |   ■ ■ |
        ''')),
        AlignedPieceBB('t', top_left=Point(1, 6), grid=parse_piece('''
            | ■     |
            | ■ ■ ■ |
            | ■     |
        ''')),
        AlignedPieceBB('z', top_left=Point(3, 6), grid=parse_piece('''
            |   ■ ■ |
            |   ■   |
            | ■ ■   |
        ''')),
        AlignedPieceBB('v', top_left=Point(3, 7), grid=parse_piece('''
            |     ■ |
            |     ■ |
            | ■ ■ ■ |
        ''')),
        AlignedPieceBB('u', top_left=Point(0, 8), grid=parse_piece('''
            | ■   ■ |
            | ■ ■ ■ |
        ''')),
    ]

    # import json
    # print(json.dumps([x.to_dict() for x in aligned_boxes_SOLVED], indent=2))
    # assert False

    grid_boxes = [
        GridBoxForPiece(
            name=x.name,
            top_left=x.top_left,
            height=x.height,
            width=x.width,
        )
        for x in aligned_boxes_SOLVED
    ]
    try_to_get_filled_grid_from_grid_boxes(grid_boxes)

    # for box in grid_boxes:
    #     print(box.name)
    #     box.print_grid(prefix='\t')
    #     print()


    # for box in aligned_boxes:
    #     print(box.name)
    #     box.print_grid(prefix='\t')
    #     print() 

    # puzzle_grid = aligned_boxes_to_puzzle_grid(aligned_boxes)
    # print()
    # print(*[' '.join(row) for row in puzzle_grid], sep='\n')

    # viz_img = build_viz_img_for_puzzle_grid(puzzle_grid)
    # plt.imshow(viz_img)
    # plt.axis('off')
    # plt.show()
