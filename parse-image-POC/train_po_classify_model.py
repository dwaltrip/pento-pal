import os
from ultralytics import YOLO

BASE_MODEL__SMALL = 'yolov8s-cls.pt'
BASE_MODEL__MEDIUM = 'yolov8m-cls.pt'
BASE_MODEL__LARGE = 'yolov8l-cls.pt'

# previous_weights_path = 'runs/classify-train-test/train/weights/best.pt'
previous_weights_path = 'zzzzzz'

DATA_DIR = '/Users/danielwaltrip/all-files/projects/ai-data'

if __name__ == '__main__':
    if os.path.isfile(previous_weights_path):
        model = YOLO(previous_weights_path)

    else:
        model = YOLO(f'weights/{BASE_MODEL__SMALL}')
        # model = YOLO(f'weights/{BASE_MODEL__MEDIUM}')
        # model = YOLO(f'weights/{BASE_MODEL__LARGE}')

        model.train(
            data=os.path.join(DATA_DIR, 'pento-exp-5--orientations'),
            epochs=10,
            # 'project', 'name' are required or the save_dir is in my old project.
            project='runs',
            name='classify-train-test',
            # batch=5,
            # device='mps',
        )
