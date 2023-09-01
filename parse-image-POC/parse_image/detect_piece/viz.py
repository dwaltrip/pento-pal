from settings import CLASS_MAPS


def draw_yolo_boxes(draw, boxes):
    for box in boxes:
        class_name = CLASS_MAPS.class_id_to_name[box.cls.item()]
        draw.rectangle(
            box.xyxy[0].tolist(),
            outline=CLASS_MAPS.name_to_color[class_name.lower()],
            width=2,
        )

def draw_bounding_boxes(draw, boxes):
    for box in boxes:
        draw.rectangle(
            [*box.top_left, *box.bot_right],
            outline=CLASS_MAPS.name_to_color[box.piece_type],
            width=2,
        )
