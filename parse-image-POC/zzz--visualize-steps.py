from collections import namedtuple, defaultdict
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance

from parse_image.settings import GRID, AI_DATA_DIR, CLASS_NAMES, CLASS_MAPS
from parse_image.parser.get_puzzle_box_corners import get_puzzle_box_corners
from parse_image.parser.get_piece_bounding_boxes import get_piece_bounding_boxes
from parse_image.parser.straighten_rect import straighten_rect
from parse_image.parser.bounding_boxes_to_grid_boxes import (
    bounding_boxes_to_grid_boxes,
)
from parse_image.parser.get_puzzle_grid_from_piece_boxes import (
    get_puzzle_grid_from_piece_boxes
)

from parse_image.utils.color import hex_to_rgb
from parse_image.utils.draw import add_rect_with_alpha
from parse_image.data.points import Point, GridCoord
from parse_image.detect_puzzle_box.viz import draw_corners, draw_dot
from parse_image.detect_piece.viz import draw_bounding_boxes_with_alpha
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
    image_with_detected_pieces = draw_bounding_boxes_with_alpha(
        normalized_image,
        piece_bounding_boxes,
        fill_alpha=90,
        outline_alpha=150,
        width=3,
    )
    image_with_detected_pieces.show()

    # Step 4 and 5
    piece_grid_boxes = bounding_boxes_to_grid_boxes(piece_bounding_boxes)

    # viz for step 4
    pixel_grid = get_pixel_grid(piece_bounding_boxes)
    def add_grid_lines_wrapped(image, thickness=2):
        return add_grid_lines(
            # normalized_image,
            image,
            rows=pixel_grid.rows,
            cols=pixel_grid.cols,
            rect=dict(
                top_left=pixel_grid.top_left,
                height=pixel_grid.height,
                width=pixel_grid.width,
            ),
            color=(255, 0, 0),
            thickness=thickness,
        )
    image_with_pixel_grid = add_grid_lines_wrapped(normalized_image)

    # viz for step 5
    image_with_piece_grid_boxes = get_image_with_piece_grid_boxes(
        # add_grid_lines_wrapped(darken_image(normalized_image, factor=0.7)),
        darken_image(
            # add_grid_lines_wrapped(normalized_image, thickness=1),
            normalized_image,
            factor=0.7,
        ),
        # darken_image(normalized_image, 0.7),
        piece_grid_boxes,
        pixel_grid,
    )
    image_with_piece_grid_boxes = draw_grid_dots(
        image_with_piece_grid_boxes,
        pixel_grid,
    )
    image_with_piece_grid_boxes.show()

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
        StepViz(
            title='Assign each piece to a grid cell',
            image=image_with_piece_grid_boxes,
        ),
        # StepViz(
    ]


def get_image_with_piece_grid_boxes(image, piece_grid_boxes, pixel_grid):
    image = image.convert('RGBA')

    cell_h = pixel_grid.cell_size.height
    cell_w = pixel_grid.cell_size.width

    for box in piece_grid_boxes:
        tl = Point(
            x = pixel_grid.top_left.x + (box.top_left.col * cell_w),
            y = pixel_grid.top_left.y + (box.top_left.row * cell_h),
        )
        br = Point(
            x = tl.x + (box.width * cell_w),
            y = tl.y + (box.height * cell_h),
        )
        color = hex_to_rgb(CLASS_MAPS.name_to_color[box.piece_type])
        image = add_rect_with_alpha(
            image, 
            [tl, br],
            outline=(*color, 150),
            fill=(*color, 100),
            width=2,
        )

    return image


def draw_grid_dots(image, pixel_grid):
    cell_h = pixel_grid.cell_size.height
    cell_w = pixel_grid.cell_size.width
    draw = ImageDraw.Draw(image)
    for row in range(pixel_grid.rows + 1):
        for col in range(pixel_grid.cols + 1):
            point = Point(
                x = pixel_grid.top_left.x + (col * cell_w),
                y = pixel_grid.top_left.y + (row * cell_h),
            )
            draw_dot(draw, point, 4, (255, 0, 0, 128))
    return image


def darken_image(image, factor=0.5):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(factor)

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
