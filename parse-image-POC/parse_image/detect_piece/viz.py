from settings import CLASS_MAPS
from parse_image.utils.color import hex_to_rgb
from parse_image.utils.draw import add_rect_with_alpha


def draw_yolo_boxes(draw, boxes, width=2):
    for box in boxes:
        class_name = CLASS_MAPS.class_id_to_name[box.cls.item()]
        draw.rectangle(
            box.xyxy[0].tolist(),
            outline=CLASS_MAPS.name_to_color[class_name.lower()],
            width=width,
        )

def draw_bounding_boxes(draw, boxes, width=2):
    for box in boxes:
        draw.rectangle(
            [*box.top_left, *box.bot_right],
            outline=CLASS_MAPS.name_to_color[box.piece_type],
            width=width,
        )


def draw_bounding_boxes_with_alpha(
    image,
    boxes,
    outline_alpha=150,
    fill_alpha=80,
    width=2,
):
    image = image.convert('RGBA')
    for box in boxes:
        color = CLASS_MAPS.name_to_color[box.piece_type]
        if len(color) != 3:
            color = hex_to_rgb(color)

        image = add_rect_with_alpha(
            image,
            [*box.top_left, *box.bot_right],
            outline=(*color, outline_alpha),
            fill=(*color, fill_alpha),
            width=width,
        )
    return image
