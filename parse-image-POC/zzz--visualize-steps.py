from collections import namedtuple
import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw

from parse_image.settings import GRID, AI_DATA_DIR
from parse_image.parser.get_puzzle_box_corners import get_puzzle_box_corners
from parse_image.parser.get_piece_bounding_boxes import get_piece_bounding_boxes
from parse_image.parser.straighten_rect import straighten_rect
from parse_image.parser.bounding_boxes_to_grid_boxes import (
    bounding_boxes_to_grid_boxes,
)
from parse_image.parser.get_puzzle_grid_from_piece_boxes import (
    get_puzzle_grid_from_piece_boxes
)

from parse_image.detect_puzzle_box.viz import draw_corners
from parse_image.detect_piece.viz import draw_bounding_boxes
from parse_image.detect_grid.common.viz import add_grid_lines

from parse_image.parser.bounding_boxes_to_grid_boxes import get_pixel_grid


StepViz = namedtuple('StepViz', ['title', 'image'])

DETECTION_THRESHOLD = 0.5

def make_visualizations_for_each_step(image):
    # Step 1
    puzzle_corners = get_puzzle_box_corners(
        image,
        conf_threshold=DETECTION_THRESHOLD,
    )
    image_with_detected_corners = image.copy()
    draw_corners(ImageDraw.Draw(image_with_detected_corners), puzzle_corners, size=7)

    # Step 2
    aspect_ratio = GRID.width / GRID.height
    normalized_image = straighten_rect(image, puzzle_corners, aspect_ratio)

    # Step 3
    piece_bounding_boxes = get_piece_bounding_boxes(
        normalized_image,
        conf_threshold=DETECTION_THRESHOLD,
    )
    image_with_detected_pieces = normalized_image.copy()
    draw_bounding_boxes(
        ImageDraw.Draw(image_with_detected_pieces),
        piece_bounding_boxes,
        width=4,
    )

    # Step 4 and 5
    piece_grid_boxes = bounding_boxes_to_grid_boxes(piece_bounding_boxes)

    # viz for step 4
    pixel_grid = get_pixel_grid(piece_bounding_boxes)
    image_with_pixel_grid = add_grid_lines(
        normalized_image,
        rows=pixel_grid.rows,
        cols=pixel_grid.cols,
        rect=dict(
            top_left=pixel_grid.top_left,
            height=pixel_grid.height,
            width=pixel_grid.width,
        ),
        color=(255, 0, 0),
        thickness=3,
    )

    # viz for step 5

    # Step 6
    puzzle_grid = get_puzzle_grid_from_piece_boxes(piece_grid_boxes)

    return [
        StepViz(
            title='Initial Image',
            image=image,
        ),
        StepViz(
            title='Keypoint detection of puzzle corners',
            image=image_with_detected_corners,
        ),
        StepViz(
            title='Straightened + Cropped Image',
            image=normalized_image,
        ),
        StepViz(
            title='Object detection of puzzle pieces',
            image=image_with_detected_pieces,
        ),
        StepViz(
            title='Divide puzzle into evenly spaced grid',
            image=image_with_pixel_grid,
        ),
        # StepViz(
    ]

# -----------------------------------------------------------------------------

def main():
    image_path = os.path.join(
        os.path.dirname(__file__),
        'example-images',
        'pento-1.png',
    )
    image = Image.open(image_path)

    visualizations = make_visualizations_for_each_step(image)

    num_images = len(visualizations)
    for i, viz in enumerate(visualizations):
        plt.subplot(1, num_images, i+1)
        plt.title(f'Step {i} - {viz.title}')
        plt.imshow(np.array(viz.image))
        plt.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
