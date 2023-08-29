from collections import namedtuple
from dataclasses import dataclass, field
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from settings import AI_DATA_DIR, GRID
from parse_image.detect_puzzle_box.labels import read_label_file, KEYPOINT_COLORS

# -----------------------------------------
# TODO: this doesn't work anymore..........
# test before recent changes
# -----------------------------------------

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
    dewarped_image = dewarp_rectangle(
        image,
        [(p.x, p.y) for p in corner_keypoints],
        aspect_ratio,
    )

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