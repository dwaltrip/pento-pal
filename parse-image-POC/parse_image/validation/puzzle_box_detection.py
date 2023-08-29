from collections import defaultdict
from pathlib import Path
import os

from PIL import Image

# from settings import CLASS_NAMES
from parse_image.detect_puzzle_box.labels import read_label_file


def are_corner_keypoints_valid(corner_keypoints):
    if len(corner_keypoints) != 4:
        return False, f'Expected 4 corner keypoints, got {len(corner_keypoints)}'

    top_left, top_right, bot_right, bot_left = corner_keypoints
    are_relative_positions_valid = (
        top_left.x < top_right.x and
        bot_left.x < bot_right.x and

        top_left.y < bot_left.y and
        top_right.y < bot_right.y
    )
    if not are_relative_positions_valid:
        return False, 'Corner keypoints were not in correct relative positions'

    return True, None
