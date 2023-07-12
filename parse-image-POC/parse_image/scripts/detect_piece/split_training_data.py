import os
import sys

from tasks.detect_piece.split_training_data import split_training_data


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Incorrect args.')
        sys.exit(1)

    data_dir = sys.argv[1]
    if not os.path.exists(data_dir):
        print(f'{data_dir} not found.')
        sys.exit(1)

    split_training_data(
        data_dir=data_dir,
        percents=dict(train=0.8, val=0.2, test=0.0),
    )
