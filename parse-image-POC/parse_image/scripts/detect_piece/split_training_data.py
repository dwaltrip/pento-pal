import os
import sys

import rich
from rich.console import Console
from rich.style import Style

from settings import AI_DATA_DIR
from tasks.detect_piece.split_training_data import (
    split_training_data,
    TrainingDataValidationError,
)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Incorrect args.')
        sys.exit(1)

    maybe_data_dir = sys.argv[1]
    data_dirs_to_try = [
        maybe_data_dir,
        os.path.join(AI_DATA_DIR, maybe_data_dir),
    ]

    data_dir = None
    for candidate_dir in data_dirs_to_try:
        if os.path.exists(candidate_dir):
            data_dir = candidate_dir
            break

    if data_dir is None:
        sep = '\n\t'
        print(f'data_dir not found. Tried:', sep.join(data_dirs_to_try), sep=sep)
        sys.exit(1)
    
    console = Console()
    try:
        split_training_data(
            data_dir=data_dir,
            percents=dict(train=0.8, val=0.2, test=0.0),
        )
    except TrainingDataValidationError as err:
        for i, (label_path, err_msg) in enumerate(err.invalid_label_files):
            if i > 0:
                print()
            console.print('Invalid file:', style=Style(color='red'), end=' ')
            console.print(label_path)
            console.print('\t' + '\n\t'.join(err_msg.split('\n')))
