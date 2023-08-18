"""
Example usage:
    bash tools/run-script.sh parse_image/detect_puzzle_box/train.py \
        testing-train-v2 \
        detect-puzzle-bb-and-kp--2023-08-17 \
        weights/yolov8s-pose.pt
"""
import os
import pathlib
import sys

from ultralytics import YOLO

from settings import AI_DATA_DIR, PROJECT_ROOT, RESULTS_ROOT
from parse_image.utils.misc import parse_script_args, write_yaml_file


CLASS_NAMES = ['PB']
KEYPOINTS = ['top-left', 'top-right', 'bot-right', 'bot-left']


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
        kpt_shape=[len(KEYPOINTS), 2],
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


# training_data_dir:
#   pentominoes/
#   detect-puzzle-bb-and-kp--2023-08-17
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
