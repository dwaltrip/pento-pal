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

def main(training_files):
    image_path, boxes = training_files[0]
    image = Image.open(image_path)
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
        data_dir=os.path.join(AI_DATA_DIR, 'ls-yolo-export--2023-07-26')
    )
    main(training_files)