from collections import namedtuple
from dataclasses import dataclass, field

from pento_parser_poc.settings import CLASS_NAMES


# Using namedtuple allows for splatting, unlike dataclass.
# E.g. draw_rect(rect=[*top_left, *bot_right])
Point = namedtuple('Point', ['x', 'y'])


@dataclass(kw_only=True)
class BoundingBox:
    class_id: int
    top_left: Point
    bot_right: Point
    width: int = field(init=False)
    height: int = field(init=False)

    def __post_init__(self):
        self.width = self.bot_right.x - self.top_left.x
        self.height = self.bot_right.y - self.top_left.y

    @classmethod
    def from_prediction(cls, pred_box):
        x1, y1, x2, y2 = pred_box.xyxy[0].tolist()
        class_id = int(pred_box.cls.item())
        return cls(
            class_id=class_id,
            top_left=Point(x=x1, y=y1),
            bot_right=Point(x=x2, y=y2),
        )

    @classmethod
    def from_yolo_label(cls, label_line, img_width, img_height):
        raw_values = label_line.strip().split()
        class_id, *box_values = raw_values
        class_id = int(class_id)
        x_center, y_center, width, height = [float(val) for val in box_values]

        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height

        top_left = Point(
            x=x_center - width / 2,
            y=y_center - height / 2,
        )
        bot_right = Point(
            x=top_left.x + width,
            y=top_left.y + height,
        )
        return cls(class_id=class_id, top_left=top_left, bot_right=bot_right)


@dataclass(kw_only=True)
class PieceBoundingBox(BoundingBox):

    piece_type = property(lambda self: CLASS_NAMES[self.class_id])

