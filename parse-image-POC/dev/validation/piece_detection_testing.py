import os
from pathlib import Path

from PIL import Image
import rich
from rich.console import Console
from rich.style import Style

from settings import AI_DATA_DIR
from parse_image.validation.piece_detection import (
    is_label_file_valid,
    validate_label_files,
)


def get_files():
    data_dir = os.path.join(AI_DATA_DIR, 'piece-detect-2023-08-22')
    image_dirs = [
        os.path.join(data_dir, 'train', 'images'),
        os.path.join(data_dir, 'val', 'images'),
    ]
    label_dirs = [Path(path).with_stem('labels') for path in image_dirs]

    image_paths = sorted([
        image_path
        for image_dir in image_dirs
        for image_path in Path(image_dir).iterdir()
        if image_path.suffix in ['.jpg', '.png']
    ])
    label_paths = sorted([
        label_path
        for label_dir in label_dirs
        for label_path in Path(label_dir).iterdir() 
        if label_path.suffix == '.txt'
    ])
    assert len(image_paths) == len(label_paths), 'mismatched image/label counts'

    return image_paths, label_paths


def main():
    console = Console()
    image_paths, label_paths = get_files()
    for image_path, label_path in zip(image_paths, label_paths):
        image = Image.open(image_path)
        try:
            is_valid, err_msg = is_label_file_valid(label_path, image)
            if not is_valid:
                console.print('Validation error :(', style=Style(color='red'), end=' ')
                print('--', label_path)
                print(err_msg)
            else:
                console.print('Looks good!', style=Style(color='green'), end=' ')
                print('--', label_path)

        except Exception as err:
            console.print('Unexpected error.', style=Style(color='red'), end=' ')
            print('--', label_path)
            print(err) 


def main2():
    console = Console()
    image_paths, label_paths = get_files()
    invalid_files = validate_label_files(
        zip(label_paths, image_paths),
    )

    if len(invalid_files) == 0:
        print('No invalid files found.', end=' ')
        console.print('Success :)', style=Style(color='green'))
    else:
        for i, (label_path, err_msg) in enumerate(invalid_files):
            if i > 0:
                print()
            console.print('Invalid file:', style=Style(color='red'), end=' ')
            console.print(label_path)
            console.print('\t' + '\n\t'.join(err_msg.split('\n')))


if __name__ == '__main__':
    # main()
    main2()