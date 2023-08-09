"""
Example usage:
    bash tools/run-script.sh parse_image/scripts/detect_piece/train_v2.py \
        testing-train-v2 \
        pento-exp-1 \
        weights/yolov8s.pt
"""
import os
import pathlib
import sys

from ultralytics import YOLO

from settings import AI_DATA_DIR, PROJECT_ROOT, RESULTS_ROOT
from parse_image.utils.misc import parse_script_args, write_yaml_file


CLASS_NAMES = ['F', 'I', 'L', 'N', 'P', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def build_dataset_paths(dirname):
    dirpath = os.path.join(AI_DATA_DIR, dirname)
    if not os.path.exists(dirpath):
        raise Exception(f'Dataset dir {dirpath} not found.')

    return dict(
        train=os.path.join(dirpath, 'train'),
        val=os.path.join(dirpath, 'val'),
    )

def setup_train_args(dataset_dir, results_dirname, other_args):
    yaml_path = os.path.join(
        PROJECT_ROOT,
        RESULTS_ROOT,
        '_datasets_',
        pathlib.Path(results_dirname).with_suffix('.yaml'),
    )
    yaml_data = dict(
        **build_dataset_paths(dataset_dir),
        nc=len(CLASS_NAMES),
        names=CLASS_NAMES,
    )
    write_yaml_file(yaml_path, yaml_data)

    return dict(
        data=yaml_path,
        # 'project', 'name' are required or the save_dir is in my old project.
        project=RESULTS_ROOT,
        name=results_dirname,
        **other_args,
    )


if __name__ == '__main__':
    args = parse_script_args([
        'run_name',
        'dataset_dir',
        'base_model',
    ])

    base_model_path = os.path.join(PROJECT_ROOT, args.base_model)
    train_args = setup_train_args(
        dataset_dir=args.dataset_dir,
        # TODO: throw an error if results_dirname alreaedy exists???
        results_dirname=args.run_name,

        other_args=dict(
            epochs=20,
            # batch_size=16,
            # device='mps',
        ),
    )

    model = YOLO(base_model_path)
    model.train(**train_args)
