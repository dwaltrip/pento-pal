import math
import os, os.path as path
from pathlib import Path
import random
import shutil

from PIL import Image


class TrainingDataValidationError(Exception):
    def __init__(self, message, invalid_label_files):
        super().__init__(message)
        self.invalid_label_files = invalid_label_files


def split_training_data(data_dir, percents, label_validator=None):
    assert sum(percents.values()) == 1, 'percents should sum to 100%'

    if not path.exists(data_dir):
        raise FileNotFoundError(f'{data_dir} not found.')

    src_images_dir = path.join(data_dir, 'images')
    src_labels_dir = path.join(data_dir, 'labels')

    train_dir = path.join(data_dir, 'train')
    val_dir = path.join(data_dir, 'val')
    test_dir = path.join(data_dir, 'test')

    image_files = [
        file
        for file in os.listdir(src_images_dir)
        if Path(file).suffix == '.png'
    ]
    label_files = [
        Path(image_file).with_suffix('.txt')
        for image_file in image_files
    ]

    if label_validator:
        invalid_files = label_validator(zip(
            [path.join(src_labels_dir, f) for f in label_files],
            [path.join(src_images_dir, f) for f in image_files],
        ))
        if len(invalid_files) > 0:
            raise TrainingDataValidationError(
                f'Invalid label files found in {data_dir}',
                invalid_label_files=invalid_files,
            )
        else:
            print('No invalid files found! Proceeding with split.')
    else:
        print('No `label_validator` provided. Proceeding with split.')
        
    for folder in [train_dir, val_dir, test_dir]:
        os.makedirs(path.join(folder, 'images'), exist_ok=True)
        os.makedirs(path.join(folder, 'labels'), exist_ok=True)

    def copy_dataset_item(image_filename, dest_dir):
        label_filename = Path(image_filename).with_suffix('.txt')

        shutil.move(
            path.join(src_images_dir, image_filename),
            path.join(dest_dir, 'images', image_filename),
        )
        shutil.move(
            path.join(src_labels_dir, label_filename),
            path.join(dest_dir, 'labels', label_filename),
        )

    train_imgs, val_imgs, test_imgs = split_list_by_percentage(
        image_files,
        [percents['train'], percents['val'], percents['test']],
    )

    for image_file in train_imgs:
        copy_dataset_item(image_file, train_dir)

    for image_file in val_imgs:
        copy_dataset_item(image_file, val_dir)

    for image_file in test_imgs:
        copy_dataset_item(image_file, test_dir)

    os.rmdir(src_images_dir)
    os.rmdir(src_labels_dir)


def split_list_by_percentage(lst, percentages, shuffle=True):
    assert sum(percentages) == 1.0, 'Percentages must add to 1'

    num_items = len(lst)
    split_sizes = [round(num_items * p) for p in percentages]
    split_sizes[-1] = num_items - sum(split_sizes[:-1])

    if shuffle:
        lst = lst.copy()
        random.shuffle(lst)

    sublists = []
    start = 0
    for list_size in split_sizes:
        sublist = lst[start : start + list_size]
        sublists.append(sublist)
        start += list_size

    return sublists
