"""
Example usage:
    bash tools/run-script.sh parse_image/detect_puzzle_box/train.py \
        testing-train-v2 \
        detect-puzzle-bb-and-kp--2023-08-17 \
        weights/yolov8s-pose.pt
"""
import os
from pathlib import Path
from ultralytics import YOLO

from settings import AI_DATA_DIR, PROJECT_ROOT, RESULTS_ROOT
from parse_image.utils.misc import parse_script_args, write_yaml_file


BASE_MODEL_WEIGHT_FILE_NAMES = dict(
    nano='yolov5n-pose.pt',
    small='yolov8s-pose.pt',
    medium='yolov8m-pose.pt',
    large='yolov8l-pose.pt',
)

PUZZLE_BOX_DETECT_CLASS_NAMES = ['puzzle-box']
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
        Path(results_dirname).with_suffix('.yaml'),
    )
    yaml_data = dict(
        **build_dataset_paths(dataset_dir),
        nc=len(PUZZLE_BOX_DETECT_CLASS_NAMES),
        names=PUZZLE_BOX_DETECT_CLASS_NAMES,
        kpt_shape=[len(KEYPOINTS), 2],
    )
    write_yaml_file(yaml_path, yaml_data)

    return dict(
        data=yaml_path,
        project=RESULTS_ROOT,
        name=results_dirname,
        **other_args,
    )


if __name__ == '__main__':
    args = parse_script_args([
        'run_name',
        'dataset_dir',
        'base_model_size',
    ])

    if args.base_model_size not in BASE_MODEL_WEIGHT_FILE_NAMES.keys():
        raise Exception(f'Invalid base model size: {args.base_model_size}')

    base_model_file = BASE_MODEL_WEIGHT_FILE_NAMES[args.base_model_size]
    print('Training with:', base_model_file)
    base_model_path = os.path.join(PROJECT_ROOT, 'weights', base_model_file)

    train_args = setup_train_args(
        dataset_dir=args.dataset_dir,
        # TODO: throw an error if results_dirname alreaedy exists???
        results_dirname=args.run_name,

        other_args=dict(
            epochs=20,
            # batch_size=8,
            # device='mps',
        ),
    )

    model = YOLO(base_model_path)
    model.train(**train_args)
