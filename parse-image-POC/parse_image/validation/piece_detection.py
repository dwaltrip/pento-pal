from collections import defaultdict
from pathlib import Path
import os

from PIL import Image

from settings import CLASS_NAMES
from parse_image.data.bounding_box import PieceBoundingBox


# TODO: possibly validate the aspect ratios of the bounding boxes
def is_label_file_valid(label_path, image):
    with open(label_path, 'r') as f:
        lines = f.read().strip().split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']

    if len(lines) == 0:
        return False, 'Label file was empty.'

    boxes = [
        PieceBoundingBox.from_yolo_label(line, image.width, image.height)
        for line in lines
    ]
    class_ids = [box.class_id for box in boxes]

    class_id_counts = defaultdict(int)
    for class_id in class_ids:
        class_id_counts[class_id] += 1

    errors = []

    for class_id, count in class_id_counts.items():
        cls_name = CLASS_NAMES[class_id].upper()
        if count > 1:
            errors.append( f'Class {cls_name} had {count} boxes.')
    
    class_names = [CLASS_NAMES[class_id] for class_id in class_ids]
    for class_name in CLASS_NAMES:
        if class_name not in class_names:
            errors.append(f'Class {class_name.upper()} was not present.')

    sep = '\n\t - '
    if len(errors) > 0:
        err_msg = 'Validation Errors:'  + sep  + sep.join(errors)
        return False, err_msg
    else:
        return True, None


def validate_label_files(label_file_image_pairs):
    invalid_label_files = []
    for label_path, image in label_file_image_pairs:
        if isinstance(image, (str, os.PathLike)):
            image = Image.open(image)

        try:
            is_valid, err_msg = is_label_file_valid(label_path, image)
            if not is_valid:
                invalid_label_files.append((
                    label_path,
                    err_msg,
                ))
        except Exception as err:
            invalid_label_files.append((
                label_path,
                f'Unexpected error: {err}',
            ))

    return invalid_label_files
