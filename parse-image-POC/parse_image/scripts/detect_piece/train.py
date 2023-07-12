import os
from ultralytics import YOLO

BASE_MODEL__SMALL = 'yolov8s.pt'
BASE_MODEL__MEDIUM = 'yolov8m.pt'
BASE_MODEL__LARGE = 'yolov8l.pt'


if __name__ == '__main__':
    base_model_path = os.path.join(
        'weights',
        BASE_MODEL__SMALL,
    )
    data_yaml = 'experiments/experiment-1.yaml'

    model = YOLO(base_model_path)
    model.train(
        data=data_yaml,
        epochs=20,
        # 'project', 'name' are required or the save_dir is in my old project.
        project='runs',
        name='piece-detect-train',
        # batch=8,
        # device='mps',
    )
