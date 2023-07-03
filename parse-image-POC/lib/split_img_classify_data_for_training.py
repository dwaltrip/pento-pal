from pathlib import Path
import os

from .split_list_by_percentage import split_list_by_percentage


def split_img_classify_data_for_training(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    class_names = [
        child.stem for child in Path(data_dir).iterdir()
        if 'DS_Store' not in child.stem
    ]

    for folder in [train_dir, val_dir, test_dir]:
        os.path.mkdir(folder)

        for class_name in class_names:
            os.path.mkdir(os.path.join(folder, class_name))


    image_files = []

    train_imgs, val_imgs, test_imgs = split_list_by_percentage(
        image_files,
        [percents['train'], percents['val'], percents['test']],
    )

    # WIP WIP
    # TODO: finish....
