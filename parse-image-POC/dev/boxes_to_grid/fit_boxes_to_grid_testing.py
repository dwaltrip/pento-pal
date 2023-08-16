from collections import namedtuple
from dataclasses import dataclass
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from settings import AI_DATA_DIR, CLASS_MAPS, CLASS_NAMES
from parse_image.boxes_to_grid.fit_boxes_to_grid import (
    load_obj_detect_training_files,
    PixelGrid,
    Point,
    estimate_alignment_error,
)
from parse_image.boxes_to_grid.pieces import (
    GridShape,
    FloatingRect,
)


def color_for_class_id(class_id):
    return CLASS_MAPS.name_to_color[CLASS_NAMES[class_id]]

def draw_bbox_points(boxes):
    for box in boxes:
        color = color_for_class_id(box.class_id)
        tl, br = box.top_left, box.bot_right
        plt.scatter(tl.x, -1 * tl.y, c=color)
        plt.scatter(br.x, -1 * br.y, c=color)
    plt.axis('equal')
    plt.show()
    return
    
    x_tl = [box.top_left.x for box in boxes]
    x_br = [box.bot_right.x for box in boxes]
    y_tl = [box.top_left.y for box in boxes]
    y_br = [box.bot_right.y for box in boxes]

    # plt.scatter(x, [-1 * y_coord for y_coord in y])
    plt.scatter(x_tl, [-1 * y for y in y_tl], c='red')
    plt.scatter(x_br, [-1 * y for y in y_br], c='blue')
    plt.axis('equal')
    plt.show()


# ----------------------------------------------------------


def zzz_estimate_grid_cell_size(box):
    shape = box.get_grid_shape()
    estimate = FloatingRect(
        height=box.height / shape.height,
        width=box.width / shape.width,
    )
    print('\testimate:', estimate)
    return (estimate.height + estimate.width) / 2.0
    # return [box.height / shape.height, box.width / shape.width]
    # return (box.height / shape.height + box.width / shape.width) / 2.0


def estimate_grid_cell_size(boxes):
    pass


# TODO:
#   What about skew?
#   Can we estimate from cell size vs total grid size?
def try_a_bunch_of_grids(boxes):
    min_x = min(box.top_left.x for box in boxes)
    max_x = max(box.bot_right.x for box in boxes)
    min_y = min(box.top_left.y for box in boxes)
    max_y = max(box.bot_right.y for box in boxes)

    ROWS = 10
    COLS = 6

    top_left_points = [
        Point(min_x + dx, max_y + dy)
        for dx in range(-2, 3) for dy in range(-2, 3)
    ]

    cs_min, cs_max = sorted([(max_x - min_x) / COLS, (max_y - min_y) / ROWS])
    cs_min -= 1
    cs_max += 1
    num_steps = int(cs_max -  cs_min) + 1
    step = (cs_max - cs_min) / int(cs_max - cs_min)
    cell_sizes = [cs_min + (i * step) for i in range(num_steps)]

    print('cell_sizes:', [round(x, 2) for x in cell_sizes])

    for top_left in top_left_points:
        for cell_size in cell_sizes:
            grid = PixelGrid(
                top_left=top_left,
                cell_size=cell_size,
                rows=ROWS,
                cols=COLS,
            )
            tl_str = f'({top_left.x:.1f}, {top_left.y:.1f})'
            print(f'\nGrid(top_left={tl_str}, cell_size={cell_size:.2f})')


def main2(training_files):
    image, boxes = training_files[0]
    for box in boxes:
        print(f'\npiece "{box.piece_type.upper()}"')
        est = zzz_estimate_grid_cell_size(box)
        print('\test:', round(est, 1))

# ----------------------------------------------------------

def main(training_files):
    image, boxes = training_files[0]
    draw = ImageDraw.Draw(image)

    for box in boxes:
        tl, br = box.top_left, box.bot_right
        draw.rectangle(
            [tl.x, tl.y, br.x, br.y],
            outline=color_for_class_id(box.class_id),
            width=2,
        )
    image.show()
    draw_bbox_points(boxes)


if __name__ == '__main__':
    training_files = load_obj_detect_training_files(
        data_dir=os.path.join(
            AI_DATA_DIR,
            'pentominoes',
            'ls-yolo-export--2023-07-26',
        )
    )

    # main(training_files)
    main2(training_files)