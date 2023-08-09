import os
from pathlib import Path
import sys

import cv2

def read_annotation_file(file_path, img_width, img_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        raw_box = (class_id, x_center, y_center, width, height)
        boxes.append(convert_to_pixel_coords(raw_box, img_width, img_height))

    return boxes
