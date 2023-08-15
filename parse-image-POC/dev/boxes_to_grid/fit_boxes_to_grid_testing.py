from collections import namedtuple
import os

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from settings import AI_DATA_DIR, CLASS_MAPS, CLASS_NAMES
from parse_image.boxes_to_grid.fit_boxes_to_grid import (
    load_obj_detect_training_files,
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

Rect = namedtuple('Rect', ['height', 'width'])
fmt_rect = lambda r: f'{r.height:.0f} x {r.width:.0f}'

CLASS_BBOX_RECT = {
    'f': Rect(height=3, width=3),
    'i': Rect(height=5, width=1),
    'l': Rect(height=4, width=2),
    'n': Rect(height=4, width=2),
    'p': Rect(height=3, width=2),
    't': Rect(height=3, width=3),
    'u': Rect(height=3, width=2),
    'v': Rect(height=3, width=3),
    'w': Rect(height=3, width=3),
    'x': Rect(height=3, width=3),
    'y': Rect(height=4, width=2),
    'z': Rect(height=3, width=3),
}

def zzz_estimate_grid_cell_size(box):
    rect = CLASS_BBOX_RECT[CLASS_NAMES[box.class_id]]
    is_asymmetric = rect.height != rect.width
    
    estimate = Rect(
        height=box.height / float(rect.height),
        width=box.width / float(rect.width),
    )
    if is_asymmetric:
        estimate2 = Rect(
            height=box.height / float(rect.width),
            width=box.width / float(rect.height),
        )
        diff = abs(estimate.height - estimate.width)
        diff2 = abs(estimate2.height - estimate2.width)
        print(f'\testimate: {fmt_rect(estimate)} -- diff: {diff:.0f}')
        print(f'\testimate2: {fmt_rect(estimate2)} -- diff: {diff2:.0f}')
        if diff > diff2:
            estimate = estimate2
    else:
        print('\testimate:', fmt_rect(estimate))

    return (estimate.height + estimate.width) / 2.0


def main2(training_files):
    image, boxes = training_files[0]
    estimates = []
    for box in boxes:
        print(f'\npiece "{box.piece_type.upper()}"')
        estimates.append(zzz_estimate_grid_cell_size(box))
    print('\n\n')
    for est in estimates:
        print(round(est, 0))

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