import os
import sys

from settings import AI_DATA_DIR
from tasks.detect_piece.split_training_data import split_training_data


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
    
    split_training_data(
        data_dir=data_dir,
        percents=dict(train=0.8, val=0.2, test=0.0),
    )
