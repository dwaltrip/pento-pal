from pathlib import Path
import os
import shutil

from split_obj_detect_data_for_training import split_list_by_percentage


def get_child_items(folder):
    return [
        child for child in Path(folder).iterdir()
        if 'DS_Store' not in child.stem
    ]


def is_png(file):
    return Path(file).suffix == '.png'


def split_img_classify_data_for_training(data_dir, percents):
    assert sum(percents.values()) == 1, 'percents should sum to 100%'

    src_class_dirs = get_child_items(data_dir)
    for src_class_dir in src_class_dirs:
        assert all(is_png(file) for file in get_child_items(src_class_dir)), \
            'src dirs should only contain png files'

    dest_data_dirs = {
        'train': os.path.join(data_dir, 'train'),
        'val': os.path.join(data_dir, 'val'),
        'test': os.path.join(data_dir, 'test'),
    }
    for dest_data_dir in dest_data_dirs.values():
        os.mkdir(dest_data_dir)

    for src_class_dir in src_class_dirs:
        class_name = src_class_dir.stem
        class_based_data_subdirs = {
            data_type: os.path.join(dir_path, class_name)
            for data_type, dir_path in dest_data_dirs.items()
        }

        image_files = [
            file.name for file in get_child_items(src_class_dir)
            if is_png(file)
        ]

        img_dict = {}
        (
            img_dict['train'], img_dict['val'], img_dict['test'],
        ) = split_list_by_percentage(
            image_files,
            [percents['train'], percents['val'], percents['test']],
        )

        for data_type, dest_dir in class_based_data_subdirs.items():
            os.mkdir(dest_dir)

            for image_file in img_dict[data_type]:
                shutil.move(
                    os.path.join(src_class_dir, image_file),
                    os.path.join(dest_dir, image_file),
                )

    for src_class_dir in src_class_dirs:
        ds_store = os.path.join(src_class_dir, '.DS_Store')
        if os.path.exists(ds_store):
            os.remove(ds_store)
        os.rmdir(src_class_dir)


if __name__ == '__main__':
    split_img_classify_data_for_training(
        '/Users/danielwaltrip/all-files/projects/ai-data/pento-exp-5--orientations',
        { 'train': 0.8, 'val': 0.2, 'test': 0 },
    )
