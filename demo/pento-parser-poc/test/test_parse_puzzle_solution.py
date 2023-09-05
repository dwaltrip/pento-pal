import os
from pathlib import Path

from PIL import Image

from pento_parser_poc.parse_puzzle_solution import parse_puzzle_solution


IMAGE_DIR = os.path.join(
    Path(__file__).parent.absolute(),
    'data',
    'images',
)

def load_image(image_filename):
    return Image.open(os.path.join(IMAGE_DIR, image_filename))


def are_grids_equal(grid1, grid2):
    stringify_grid = lambda g: '\n'.join([' '.join(row) for row in g])
    return stringify_grid(grid1) == stringify_grid(grid2)


def test_parse_puzzle_solution():
    # TODO: This should be in data, paired with the image file
    # Just like the label files are for training
    expected_raw = [
        'f i i i i i',
        'f f f x u u',
        't f x x x u',
        't t t x u u',
        't y y y y w',
        'l l n y w w',
        'l n n w w z',
        'l n v z z z',
        'l n v z p p',
        'v v v p p p',
    ]
    expected_grid = [line.split() for line in expected_raw]

    image = load_image('IMG_2348.png')
    puzzle_grid = parse_puzzle_solution(image)

    assert are_grids_equal(puzzle_grid, expected_grid)


TEST_FUNCTIONS = [
    test_parse_puzzle_solution,
]


# TODO: Use a proper test runner
if __name__ == '__main__':
    success_count = 0
    failure_count = 0

    for test_function in TEST_FUNCTIONS:
        try:
            test_function()
            success_count += 1
        except AssertionError:
            failure_count += 1

    print(f'Success: {success_count}')
    print(f'Failure: {failure_count}')
