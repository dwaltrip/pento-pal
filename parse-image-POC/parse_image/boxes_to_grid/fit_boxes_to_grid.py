from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from settings import CLASS_NAMES, CLASS_MAPS
from parse_image.data.bounding_box import PieceBoundingBox
from parse_image.data.points import Point


def load_obj_detect_training_files(data_dir):
    images_dir = os.path.join(data_dir, 'images')
    labels_dir = os.path.join(data_dir, 'labels')

    image_files = [
        f for f in os.listdir(images_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]
    if not len(image_files):
        raise Exception('No images found.')

    training_files = []
    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, Path(image_file).with_suffix('.txt'))
        image = Image.open(image_path)
        boxes = read_label_file(label_path, image.width, image.height)
        training_files.append((image_path, boxes))
    return training_files


def read_label_file(file_path, img_width, img_height):
    with open(file_path, 'r') as file:
        lines = file.read().strip().split('\n')
    return [
        PieceBoundingBox.from_yolo_label(line, img_width, img_height)
        for line in lines
    ]
