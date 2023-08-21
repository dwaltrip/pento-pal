from collections import namedtuple
from dataclasses import dataclass
import os

from PIL import Image, ImageDraw

from settings import AI_DATA_DIR, CLASS_MAPS, GRID
from parse_image.boxes_to_grid.fit_boxes_to_grid import (
    load_obj_detect_training_files,
)



Point = namedtuple('Point', ['x', 'y'])
FloatingRect = namedtuple('FloatingRect', ['height', 'width'])


@dataclass
class PixelGrid:
    rows = GRID.height
    cols = GRID.width

    top_left: Point
    bot_right: Point

    height = property(lambda self: self.bot_right.y - self.top_left.y)
    width = property(lambda self: self.bot_right.x - self.top_left.x)

    cell_size = property(lambda self: FloatingRect(
        height=self.height / GRID.height,
        width=self.width / GRID.width,
    ))


@dataclass
class GridBox:
    top_left: Point
    height: int
    width: int
    piece_type: str

    bot_right = property(lambda self: Point(
        x=self.top_left.x + self.width,
        y=self.top_left.y + self.height,
    ))


def class_id_to_piece_type(class_id):
    return CLASS_MAPS.class_id_to_name[class_id]


def get_grid(boxes):
    return PixelGrid(
        top_left=Point(
            min(bb.top_left.x for bb in boxes),
            min(bb.top_left.y for bb in boxes),
        ),
        bot_right=Point(
            max(bb.bot_right.x for bb in boxes),
            max(bb.bot_right.y for bb in boxes),
        ),
    )


def bounding_boxes_to_grid_boxes(bounding_boxes):
    pixel_grid = get_grid(bounding_boxes)
    cell_size = pixel_grid.cell_size

    # Convert bounding boxes to grid boxes
    grid_boxes = []
    for bb in bounding_boxes:
        top_left = Point(
            int(round((bb.top_left.x - pixel_grid.top_left.x) / cell_size.width)),
            int(round((bb.top_left.y - pixel_grid.top_left.y) / cell_size.height)),
        )
        grid_box = GridBox(
            top_left=top_left,
            height=int(round(bb.height / cell_size.height)),
            width=int(round(bb.width / cell_size.width)),
            piece_type=class_id_to_piece_type(bb.class_id),
        )
        grid_boxes.append(grid_box)

    return grid_boxes

# ------------------------------------------------------------------------------

def draw_grid(image, pixel_grid):
    top_left, c_size = pixel_grid.top_left, pixel_grid.cell_size
    rows, cols = pixel_grid.rows, pixel_grid.cols

    draw = ImageDraw.Draw(image)
    draw_line = lambda x1, y1, x2, y2: draw.line([x1, y1, x2, y2], fill='red', width=2)

    # draw vertical lines
    for i in range(cols + 1):
        x = top_left.x + (i * c_size.width)
        draw_line(x, top_left.y, x, top_left.y + pixel_grid.height)
    # draw horizontal lines
    for i in range(rows + 1):
        y = top_left.y + (i * c_size.height)
        draw_line(top_left.x, y, top_left.x + pixel_grid.width, y)
    
    image.show()

def draw_rect(image, top_left, bot_right, color='red', width=2):
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        [top_left.x, top_left.y, bot_right.x, bot_right.y],
        outline=color,
        width=width,
    )

def main(image_path, boxes):
    image = Image.open(image_path)

    pixel_grid = get_grid(boxes)
    cell_size = pixel_grid.cell_size
    grid_boxes = sorted(
        bounding_boxes_to_grid_boxes(boxes),
        key=lambda gb: (gb.top_left.y, gb.top_left.x),
    )

    for grid_box in grid_boxes:
        tl = Point(
            pixel_grid.top_left.x + (grid_box.top_left.x * cell_size.width),
            pixel_grid.top_left.y + (grid_box.top_left.y * cell_size.height),
        )
        br = Point(
            tl.x + (grid_box.width * cell_size.width),
            tl.y + (grid_box.height * cell_size.height),
        )
        gb = grid_box
        print(f'Piece {gb.piece_type} --',
            f'top_left: (y={gb.top_left.y}, x={gb.top_left.x}),',
            f'height: {gb.height}, width: {gb.width},',
        )
        draw_rect(image, tl, br, color=CLASS_MAPS.name_to_color[grid_box.piece_type])
    image.show()
    draw_grid(Image.open(image_path), pixel_grid)

# ------------------------------------------------------------------------------

if __name__ == '__main__':
    training_data = load_obj_detect_training_files(
        data_dir=os.path.join(AI_DATA_DIR, 'ls-yolo-export--2023-07-26')
    )

    target_img = 'e5cc3dc8-IMG_2964.png'
    image_path, boxes = next((
        (image_path, boxes) for image_path, boxes in training_data
        if image_path.endswith(target_img)
    ))
    # image_path, boxes = training_data[0]

    main(image_path, boxes)