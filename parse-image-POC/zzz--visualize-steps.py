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
from parse_image.data.points import Point, GridCoord
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
    def add_grid_lines_wrapped(image):
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
            thickness=2,
        )
    image_with_pixel_grid = add_grid_lines_wrapped(normalized_image)

    # viz for step 5
    image_with_piece_grid_boxes = get_image_with_piece_grid_boxes(
        add_grid_lines_wrapped(darken_image(normalized_image, factor=0.7)),
        piece_grid_boxes,
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
        )
        # StepViz(
    ]


def get_image_with_piece_grid_boxes(image, piece_grid_boxes, pixel_grid):
    piece_types_per_grid_box = defaultdict(list)
    for box in piece_grid_boxes:
        tl, br = box.top_left, box.bot_right
        for col in range(tl.col, br.col):
            for row in range(tl.row, br.row):
                coord = GridCoord(col=col, row=row)
                piece_types_per_grid_box[coord].append(box.piece_type)
    
    cell_size = pixel_grid.cell_size
    cell_h, cell_w = cell_size.height, cell_size.width
    mini_grid = calc_square_grid_info(
        container=SimpleNamespace(height=cell_h, width=cell_w),
        rows=4,
        cols=3,
        pad=5,
    )

    def get_ith_mini_grid_square(container_top_left, i):
        rows = mini_grid.rows
        cols = mini_grid.cols
        square_size = mini_grid.square_size
        pad = mini_grid.pad
        offset = Point(
            x=container_top_left.x + mini_grid.offset.x,
            y=container_top_left.y + mini_grid.offset.y,
        )

        row = i // mini_grid.cols
        col = i % mini_grid.cols

        top_left = Point(
            x=offset.x + pad + (col * (square_size + pad)),
            y=offset.y + pad + (row * (square_size + pad)),
        )
        bot_right = Point(
            x=top_left.x + square_size,
            y=top_left.y + square_size,
        )
        return dict(top_left=top_left, bot_right=bot_right, size=square_size)

    draw = ImageDraw.Draw(image)
    def draw_piece_type_for_grid_cell(coord, class_num):
        cell_tl = Point(
            x=pixel_grid.top_left.x + (coord.col * cell_w),
            y=pixel_grid.top_left.y + (coord.row * cell_h),
        )
        cell_br = Point(x=cell_tl.x + cell_w, y=cell_tl.y + cell_h)

        color = hex_to_rgb(CLASS_MAPS.name_to_color[name])
        mini_cell = get_ith_mini_grid_square(cell_tl, class_num)
        draw.rectangle(
            [mini_cell['top_left'], mini_cell['bot_right']],
            outline=(220, 220, 220),
            fill=(*color, 128),
            width=1,
        )

    for coord, piece_types in piece_types_per_grid_box.items():
        piece_types = set(piece_types)
        for i in range(len(CLASS_NAMES)):
            name = CLASS_NAMES[i]
            if name in piece_types:
                draw_piece_type_for_grid_cell(coord, i)
    
    return image


def calc_square_grid_info(container, rows, cols, pad):
    con_h, con_w = container.height, container.width

    total_pad_x = (cols + 1) * pad
    total_pad_y = (rows + 1) * pad
    square_size = min(
        (con_w - total_pad_x) // cols,
        (con_h - total_pad_y) // rows,
    )

    grid_w = square_size * cols + total_pad_x
    grid_h = square_size * rows + total_pad_y
    offset = Point(
        x = (con_w - grid_w) // 2,
        y = (con_h - grid_h) // 2,
    )
    
    return SimpleNamespace(
        rows=rows,
        cols=cols,
        height=con_h,
        width=con_w,
        square_size=square_size,
        offset=offset,
        pad=pad,
    )


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
