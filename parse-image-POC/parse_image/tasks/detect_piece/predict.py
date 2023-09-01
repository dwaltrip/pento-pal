import os
from PIL import Image, ImageDraw

from settings import CLASS_MAPS
from parse_image.detect_piece.viz import draw_yolo_boxes


def ltr_key(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (y1, x1, y2, x2)


def predict_image(model, image_path, conf_threshold=None):
    results = model.predict(image_path, verbose=False)
    result = results[0]
    print(f'num objects: {len(result.boxes)} --', end=' ')

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # boxes = sorted(result.boxes, key=ltr_key)
    boxes = sorted(result.boxes, key=lambda box: box.conf.item(), reverse=True)

    if conf_threshold:
        boxes = [box for box in boxes if box.conf.item() >= conf_threshold]

    for box in boxes:
        class_name = result.names[box.cls.item()].upper()
        conf = box.conf.item()
        print(f'{class_name}: {conf*100:0.0f}%', end=', ')
    print()

    draw_yolo_boxes(draw, boxes)
    image.show()


def predict_piece_detection(model, image_paths, conf_threshold=None):
    for image_path in image_paths:
        predict_image(model, image_path, conf_threshold=conf_threshold)
