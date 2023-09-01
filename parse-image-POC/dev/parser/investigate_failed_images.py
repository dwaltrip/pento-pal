from collections import defaultdict
import os
import sys

from PIL import Image

from settings import AI_DATA_DIR, CLASS_NAMES
from utils.iter_images_and_labels import iter_images_and_labels
from utils.print_puzzle_grid import print_puzzle_grid
from utils.read_label_file import read_puzzle_grid_label

from parse_image.parser.get_piece_bounding_boxes import get_piece_bounding_boxes
from parse_image.parser.errors import PieceDetectionError
from parse_image.parser.models import load_model
from parse_image.parser.logger import rich_print


CLASS_NAMES = [name.upper() for name in CLASS_NAMES]

# DETECTION_THRESHOLD = None
DETECTION_THRESHOLD = 0.5


def ltr_key(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (y1, x1, y2, x2)

def print_piece_bounding_boxes(image, model):
    results = model.predict(image, verbose=False)
    result = results[0]
    boxes = result.boxes

    if DETECTION_THRESHOLD:
        boxes = [
            box for box in boxes
            if box.conf.item() > 0.7
        ]

    counts = defaultdict(int)
    pred_confs = defaultdict(list)

    for name in CLASS_NAMES:
        counts[name] = 0
    
    for box in boxes:
        class_name = result.names[box.cls.item()].upper()
        counts[class_name] += 1
        pred_confs[class_name].append(box.conf.item())

    has_errors = False
    for name in CLASS_NAMES:
        if counts[name] != 1:
            has_errors = True
            break
    if not has_errors:
        rich_print('\t- no bounding box issues - ', color='green')
        return

    for name, count in sorted(counts.items(), key=lambda x: x[0]):
        if count == 1:
            continue

        if count != 1:
            rich_print(f'\t- error - ', end='', color='red')
        else:
            rich_print(f'\t-         ', end='')

        rich_print(f'{name}: {count}', end='')
        
        confs = pred_confs[name]
        if len(confs) > 0:
            conf_str = ', '.join([f'{c*100:0.0f}%' for c in confs])
            rich_print(f', conf: ({conf_str})')
        else:
            rich_print(', conf: n/a', color='#606060')


def investigate_images(target_images, data_dirname, piece_detect_model):
    image_dir = os.path.join(AI_DATA_DIR, data_dirname, 'images')
    labels_dir = os.path.join(AI_DATA_DIR, data_dirname, 'labels')

    target_files = [
        (image_path, label_path)
        for image_path, label_path in iter_images_and_labels(image_dir, labels_dir)
        if os.path.basename(image_path) in target_images
    ]

    for i, (image_path, label_path) in enumerate(target_files):
        image_filename = os.path.basename(image_path)

        print()
        print(f'[{i+1}/{len(target_files)}] {image_filename}')

        print_piece_bounding_boxes(image_path, piece_detect_model)


FAILED_IMAGES = [
    'IMG_2869.png', 'IMG_2399.png', 'IMG_2372.png', 'IMG_3247.png',
    'IMG_3251.png', 'IMG_3244.png', 'IMG_2947.png', 'IMG_3308.png',
    'IMG_3297.png', 'IMG_3042.png', 'IMG_3243.png', 'IMG_3041.png',
    'IMG_2810.png', 'IMG_2811.png', 'IMG_2379.png', 'IMG_2618.png',
    'IMG_2753.png',
]

# PIECE_DETECT_MODEL_NAME = 'detect-pieces--08-24--ds104-small-e90'
PIECE_DETECT_MODEL_NAME = 'detect-pieces--08-31--ds147-small-120'

if __name__ == '__main__':
    rich_print('Piece Detection Model:', PIECE_DETECT_MODEL_NAME)
    rich_print('Detection Threshold:', DETECTION_THRESHOLD)

    piece_detect_model = load_model(PIECE_DETECT_MODEL_NAME)

    investigate_images(
        FAILED_IMAGES,
        'detect-grid-hard--2023-08-01',
        piece_detect_model,
    )
    
