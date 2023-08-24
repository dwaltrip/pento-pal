import os
import sys

from PIL import Image

from settings import AI_DATA_DIR
from utils.print_puzzle_grid import print_puzzle_grid
from parse_image.parser.parse_puzzle_solution import (
    parse_puzzle_solution
)


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print('Usage: python zzz_run_the_e2e_poc.py <image_path>')
    #     sys.exit(1)
    # image_path = sys.argv[1]

    image_dir = os.path.join(
        AI_DATA_DIR,
        'detect-grid-hard--2023-08-01',
        'images',
    )
    image_path = os.path.join(image_dir, 'IMG_2402.png')
    image = Image.open(image_path)

    puzzle_grid = parse_puzzle_solution(image)
    print_puzzle_grid(puzzle_grid)