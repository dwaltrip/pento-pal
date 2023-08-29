from collections import OrderedDict

from parse_image.data import BoundingBox, Point



KEYPOINT_COLORS_BY_NAME = OrderedDict([
    ("top-left", "#e15250"),
    ("top-right", "#59d757"),
    ("bot-right", "#f5df36"),
    ("bot-left", "#4a76d9"),
])

KEYPOINT_NAMES = list(KEYPOINT_COLORS_BY_NAME.keys())
KEYPOINT_COLORS = list(KEYPOINT_COLORS_BY_NAME.values())


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
        tl_x, tl_y,
        tr_x, tr_y,
        br_x, br_y,
        bl_x, bl_y,
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
    pb_bb = BoundingBox(
        class_id=class_id,
        top_left=pb_top_left,
        bot_right=pb_bot_right,
    )

    return pb_bb, keypoints
