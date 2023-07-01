import math
import os, os.path as path
from pathlib import Path
import random
import shutil

from PIL import Image


def split_data_into_train_val_test(data_dir, percents):
    assert sum(percents.values()) == 1, 'percents should sum to 100%'

    src_images_dir = path.join(data_dir, 'images')
    src_labels_dir = path.join(data_dir, 'labels')

    train_dir = path.join(data_dir, 'train')
    val_dir = path.join(data_dir, 'val')
    test_dir = path.join(data_dir, 'test')

    for dir in [train_dir, val_dir, test_dir]:
        os.makedirs(path.join(dir, 'images'), exist_ok=True)
        os.makedirs(path.join(dir, 'labels'), exist_ok=True)

    image_files = [
        file
        for file in os.listdir(src_images_dir)
        if Path(file).suffix == '.png'
    ]

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
