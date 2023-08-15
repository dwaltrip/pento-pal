from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from settings import CLASS_NAMES, CLASS_MAPS


# TODO: It's probably non-idiomatic to put the y-coord first...
Point = namedtuple('Point', ['y', 'x'])

@dataclass
class BoundingBox:
    class_id: int
    top_left: Point
    bot_right: Point
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self.width = self.bot_right.x - self.top_left.x
        self.height = self.bot_right.y - self.top_left.y

    piece_type = property(lambda self: CLASS_NAMES[self.class_id])

    @classmethod
    def from_yolo(cls, yolo_box, img_width, img_height):
        class_id, x_center, y_center, width, height = yolo_box
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        top_left = Point(y=y_center - height / 2, x=x_center - width / 2)
        bot_right = Point(y=top_left.y + height, x=top_left.x + width)
        return BoundingBox(
            class_id=int(class_id),
            top_left=top_left,
            bot_right=bot_right,
        )


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
        label_file = Path(image_file).with_suffix('.txt')
        image_path = os.path.join(images_dir, image_file)
        label_path = os.path.join(labels_dir, label_file)

        image = Image.open(image_path)
        boxes = read_label_file(label_path, image.width, image.height)
        training_files.append((image, boxes))
    return training_files


def read_label_file(file_path, img_width, img_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    boxes = []
    for line in lines:
        boxes.append(BoundingBox.from_yolo(
            [float(val) for val in line.strip().split()],
            img_width,
            img_height,
        ))
    return boxes