from collections import namedtuple
from dataclasses import dataclass, field
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from settings import AI_DATA_DIR, GRID


Point = namedtuple('Point', ['x', 'y'])

@dataclass
class BoundingBox:
    top_left: Point
    bot_right: Point
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self.width = self.bot_right.x - self.top_left.x
        self.height = self.bot_right.y - self.top_left.y


def dewarp_rectangle(pil_image, corners, aspect_ratio):
    """
    De-warps a rectangular region in the given image.
    Params:
        pil_image: Input image (PIL Image)
        corners: List of four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        aspect_ratio: Target aspect ratio (width/height)
    Returns:
        De-warped image (and also cropped)
    """
    image = np.array(pil_image)
    # Detected points
    src_pts = np.array(corners, dtype=np.float32)

    # Calculate distances between corresponding points
    width_dist = max(cv2.norm(src_pts[0] - src_pts[1]), cv2.norm(src_pts[2] - src_pts[3]))
    height_dist = max(cv2.norm(src_pts[0] - src_pts[3]), cv2.norm(src_pts[1] - src_pts[2]))

    # Determine width and height based on the aspect ratio
    width = int(width_dist)
    height = int(width / aspect_ratio)

    # Idealized rectangle points
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    # Apply homography to the image
    dewwarped_image = cv2.warpPerspective(image, H, (width, height))

    return Image.fromarray(dewwarped_image)

# ----------------------


def read_label_file(label_path, img_height, img_width):
    with open(label_path, 'r') as file:
        lines = file.read().strip().split('\n')

    assert len(lines) == 1, f'Expected 1 line in label file, got {len(lines)}'
    line = lines[0]

    label_values = [float(val) for val in line.strip().split()]
    assert len(label_values) == 13, f'Expected 13 values in label file, got {len(label_values)}'
    (
        # puzzle box
        class_id, pb_x_center, pb_y_center, pb_width, pb_height,
        # 4 corners
        # tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y,
        # --------------------------------------------------------------------
        # NOTE: I accidentally flipped br and bl in the label files
        #   It's now fixed currently in parse_label_studio_json.py,
        #   so future datasets will have the correct ordering (clockwise).
        #   We will have to change this back for those.
        # -------------------------------------------------------------------
        tl_x, tl_y, tr_x, tr_y, bl_x, bl_y, br_x, br_y
    ) = label_values

    def point_from_yolo_coords(x, y):
        return Point(x=x*img_width, y=y*img_height)

    keypoints = [
        point_from_yolo_coords(x=tl_x, y=tl_y),
        point_from_yolo_coords(x=tr_x, y=tr_y),
        point_from_yolo_coords(x=br_x, y=br_y),
        point_from_yolo_coords(x=bl_x, y=bl_y),
    ]

    pb_top_left = point_from_yolo_coords(
        x=pb_x_center - pb_width / 2,
        y=pb_y_center - pb_height / 2,
    )
    pb_bot_right = point_from_yolo_coords(
        x=pb_x_center + pb_width / 2,
        y=pb_y_center + pb_height / 2,
    )
    pb_bb = BoundingBox(top_left=pb_top_left, bot_right=pb_bot_right)

    return pb_bb, keypoints


# ----------------------------------------------------------------

KEYPOINT_COLORS = [
    "#e15250", # "top-left"
    "#59d757", # "top-right"
    "#f5df36", # "bot-right"
    "#4a76d9", # "bot-left"
]


def display_dewarped_puzzle_using_keypoint_labels(data_dirname, image_filename):
    label_filename = Path(image_filename).with_suffix('.txt')

    data_dir = os.path.join(AI_DATA_DIR, data_dirname, 'train')
    image_path = os.path.join(data_dir, 'images', image_filename)
    label_path = os.path.join(data_dir, 'labels', label_filename)

    image = Image.open(image_path)
    (pb_bb, corner_keypoints) = read_label_file(label_path, image.height, image.width)

    size = 3
    for point, color in zip(corner_keypoints, KEYPOINT_COLORS):
        draw = ImageDraw.Draw(image)
        draw.ellipse(
            [point.x - size, point.y - size, point.x + size, point.y + size],
            fill=color,
            outline=color,
        )
    
    aspect_ratio = GRID.width / GRID.height
    dewarped_image = dewarp_rectangle(image, corner_keypoints, aspect_ratio)

    image.show()
    dewarped_image.show()


def main():
    data_dirname = 'detect-puzzle-box--2023-08-19--ts257'

    # image_filename = '7c9cee54-IMG_3041.png'
    # image_filename = '348be9b8-IMG_3047.png'
    image_filename = '57760362-IMG_2838.png'

    display_dewarped_puzzle_using_keypoint_labels(
        data_dirname,
        image_filename,
    )

if __name__ == '__main__':
    main()