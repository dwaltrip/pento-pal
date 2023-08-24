from collections import defaultdict
import os
import sys

from PIL import Image
import rich
from rich.console import Console
from rich.style import Style

from settings import AI_DATA_DIR
from utils.print_puzzle_grid import print_puzzle_grid
from parse_image.parser.parse_puzzle_solution import (
    parse_puzzle_solution
)
from parse_image.parser.get_piece_bounding_boxes import get_piece_bounding_boxes
from parse_image.parser.errors import PieceDetectionError


def read_label_file(label_path):
    with open(label_path) as f:
        lines = f.read().strip().split('\n')
    label_rows = [line.strip().split(' ') for line in lines]
    return label_rows


def do_puzzle_grids_match(grid1, grid2):
    for row1, row2 in zip(grid1, grid2):
        for piece1, piece2 in zip(row1, row2):
            if piece1 != piece2:
                return False
    return True


def main():
    console = Console()

    image_dir = os.path.join(
        AI_DATA_DIR,
        'detect-grid-hard--2023-08-01',
        'images',
    )
    labels_dir = os.path.join(
        AI_DATA_DIR,
        'detect-grid-hard--2023-08-01',
        'labels_COMBINED',
    )

    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    print(
        f'Skipped {len(os.listdir(labels_dir)) - len(label_files)}',
        'non-txt files in labels_dir.',
    )

    outcome_counts = defaultdict(int)

    # raw_bb_counts = defaultdict(int)
    # bb_counts = defaultdict(int)

    for i, label_filename in enumerate(label_files):
        image_filename = label_filename.replace('.txt', '.png')

        label_path = os.path.join(labels_dir, label_filename)
        image_path = os.path.join(image_dir, image_filename)
        image = Image.open(image_path)

        # raw_count, count = piece_bounding_boxes = get_piece_bounding_boxes(image)
        # raw_bb_counts[raw_count] += 1
        # bb_counts[count] += 1

        if i % 20 == 0:
            percent = i / len(label_files) * 100
            print(f'--- {percent:.0f}% complete ---')

        print(f'[{i:3}] {image_filename}:', end=' ')

        grid_from_label = read_label_file(label_path)
        try:
            predicted_puzzle_grid = parse_puzzle_solution(image)
        except PieceDetectionError as err:
            counts_str = f"({err.data['count']}, {err.data['raw_count']})"
            console.print(f'error', style=Style(color='red'), end=' ')
            console.print(counts_str, style=Style(color='#888888'))
            outcome_counts['piece detection error'] += 1
            continue

        if not predicted_puzzle_grid:
            console.print('prediciton is None', style=Style(color='red'))
            outcome_counts['prediction is None'] += 1
            continue

        if do_puzzle_grids_match(grid_from_label, predicted_puzzle_grid):
            outcome_counts['success'] += 1
            console.print('success', style=Style(color='green'))
        else:
            outcome_counts['failure'] += 1
            console.print('failed', style=Style(color='#888888'))
            # image.show()
    

    # print('Raw counts:')
    # for k, v in raw_bb_counts.items():
    #     print(f'{k:3}: {v:3}')
    # print()
    # print('Counts:')
    # for k, v in bb_counts.items():
    #     print(f'{k:3}: {v:3}')
    # return

    print()
    print('----------------------------------------------')
    print(f'Dataset: {len(label_files)} images.')
    print()
    for k, v in outcome_counts.items():
        print(f'Counts - {k:25}: {v:3}')

    print()
    success_count = outcome_counts['success']
    success_rate = success_count / len(label_files) * 100
    print(f'Success rate:           {success_rate:.1f}%')
    print('----------------------------------------------')
    print()


if __name__ == '__main__':
    main()