from collections import namedtuple
from dataclasses import dataclass, field
import math
from pathlib import Path
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from settings import CLASS_NAMES, CLASS_MAPS
from parse_image.utils.geometry import dist_from_point_to_line
from parse_image.boxes_to_grid.pieces import PIECES_BY_NAME


Point = namedtuple('Point', ['x', 'y'])

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
    piece = property(lambda self: PIECES_BY_NAME[self.piece_type])

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

    # TODO: test this, does it work??
    def get_grid_shape(self):
        shapes = self.piece.grid_shapes()
        shape = shapes[0]
        if len(shapes) == 2:
            s1, s2 = shapes
            s1_err = abs((self.height / s1.height) - (self.width / s1.width))
            s2_err = abs((self.height / s2.height) - (self.width / s2.width))
            if s2_err < s1_err:
                shape = s2
        return shape


@dataclass
class PixelGrid:
    top_left: Point
    cell_size: float 
    rows: int
    cols: int

    def find_containing_cell(self, point):
        row = int((point.y - self.top_left.y) / self.cell_size)
        col = int((point.x - self.top_left.x) / self.cell_size)
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            raise ValueError(f'Point {point} not in grid. row={row}, col={col}')
        return row, col
    
    def cell_corners(self, cell):
        row, col = cell
        size = self.cell_size
        tl = Point(self.top_left.x + c*size, self.top_left.y + r*size)
        tr = Point(tl.x + size, tl.y)   
        bl = Point(tl.x       , tl.y + size)
        br = Point(tl.x + size, tl.y + size)
        return (tl, tr, br, bl)
    
    def __repr__(self):
        tl_str = f'({self.top_left.x:.2f}, {self.top_left.y:.2f})'
        return 'PixelGrid(' + ' '.join([
            f'top_left={tl_str},',
            f'cell_size={self.cell_size},',
            f'rows={self.rows},',
            f'cols={self.cols})',
        ])


def estimate_alignment_error(grid, bounding_boxes):
    dist = lambda p1, p2: math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    errors = []
    for box in bounding_boxes:
        # TODO: why did I think I need shape here?
        # I think all we need is box.top_left and box.bot_right
        shape = box.get_grid_shape()
        
        for point in (box.top_left, box.bot_right):
            cell = grid.find_containing_cell(point)
            min_dist = min(dist(point, c) for c in grid.cell_corners(cell))
            errors.append(min_dist)

    return sum(errors) / len(errors)


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