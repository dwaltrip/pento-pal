from collections import namedtuple

import cv2
import numpy as np
from PIL import Image

from settings import NUM_CLASSES
from parse_image.data.bounding_box import PieceBoundingBox
from parse_image.data.points import Point
from parse_image.parser.errors import (
    AnalysisError,
    CornerDetectionError,
    PieceDetectionError,
)
from parse_image.parser.models import (
    load_corner_prediction_model,
    load_piece_detection_model,
)


DEBUG = False

DETECTION_THRESHOLD = 0.7


def get_puzzle_box_corners(image):
    model = load_corner_prediction_model()

    results = model.predict(image, verbose=False)
    result = results[0]
    boxes = [
        box for box in result.boxes
        if box.conf.item() > DETECTION_THRESHOLD
    ]

    if len(boxes) != 1:
        raise CornerDetectionError(f'Expected 1 box, got {len(boxes)}.')

    box = boxes[0]
    # TODO: we don't actually use puzzle_box... get rid of it?
    puzzle_box = box.xyxy[0].tolist()
    corners_from_keypoints = result.keypoints.squeeze(0).tolist()

    if len(corners_from_keypoints) != 4:
        raise CornerDetectionError(
            f'Expected 4 corners, got {len(corners_from_keypoints)}.',
        )

    return corners_from_keypoints


def dewarp_rectangle(pil_image, corners, aspect_ratio):
    """
    De-warps a rectangular region in the given image.
    Params:
        pil_image: Input image (PIL Image)
        corners: List of four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        aspect_ratio: Target aspect ratio (width/height)
    Returns:
        De-warped, cropped image
    """
    image = np.array(pil_image)
    # Detected corners from the image
    src_pts = np.array(corners, dtype=np.float32)

    width_dist = max(
        cv2.norm(src_pts[0] - src_pts[1]),
        cv2.norm(src_pts[2] - src_pts[3]),
    )
    height_dist = max(
        cv2.norm(src_pts[0] - src_pts[3]),
        cv2.norm(src_pts[1] - src_pts[2]),
    )

    # ----------------------------------------------------------------------
    # TODO: we aren't using the height_dist at all... not sure about that...
    # Need to ask GPT-4 to explain this code
    # ----------------------------------------------------------------------
    width = int(width_dist)
    height = int(width / aspect_ratio)

    # Idealized rectangle points
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Find homography and apply to the image
    H, _ = cv2.findHomography(src_pts, dst_pts)
    dewarped_image = cv2.warpPerspective(image, H, (width, height))

    return Image.fromarray(dewarped_image)


def get_piece_bounding_boxes(image):
    model = load_piece_detection_model()
    results = model.predict(image, verbose=False)
    result = results[0]

    boxes = [
        box for box in result.boxes
        if box.conf.item() > DETECTION_THRESHOLD
    ]

    if DEBUG:
        return len(result.boxes), len(boxes)

    # TODO: There are fallbacks we can attempt to do here.
    # E.g. For each piece type, take the one with the highest confidence.
    # Need to investigate these cases.
    if len(boxes) != NUM_CLASSES:
        msg = ' '.join([
            f'Incorrect numer of pieces detected.',
            f'Expected {NUM_CLASSES}, got {len(boxes)}.',
        ])
        data = dict(count=len(boxes), raw_count=len(result.boxes))
        raise PieceDetectionError(msg, data=data)

    return [PieceBoundingBox.from_prediction(box) for box in boxes]
