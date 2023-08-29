import os
import sys

from PIL import Image, ImageDraw
from ultralytics import YOLO

from parse_image.detect_puzzle_box.viz import draw_corners


KEYPOINT_COLORS = [
    "#e15250", # "top-left"
    "#59d757", # "top-right"
    "#f5df36", # "bot-right"
    "#4a76d9", # "bot-left"
]

def predict_image(model, image_path):
    results = model.predict(image_path, verbose=False)
    result = results[0]
    print('num objects:', len(result.boxes))

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for box in result.boxes:
        class_name = result.names[box.cls.item()]
        conf = box.conf.item()

        if conf < 0.5:
            continue

        print(
            'class:', class_name,
            '-- conf:', round(conf, 3),
            '-- coords:', [int(c) for c in box.xyxy[0].tolist()],
        )
        draw.rectangle(
            box.xyxy[0].tolist(),
            # outline=class_color_map[class_name],
            outline='red',
            width=2,
        )

    keypoints = result.keypoints.squeeze(0).tolist()
    draw_corners(draw, keypoints)

    image.show()


def predict_puzzle_box_and_keypoints(model, image_paths):
    for image_path in image_paths:
        predict_image(model, image_path)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Incorrect args.")
        sys.exit(1)

    weights_path = sys.argv[1]

    test_images = [
        'example-images/pento-1.png',
        'example-images/pento-2.png',
        'example-images/pento-3.png',
        'example-images/pento-4.png',
        'example-images/pento-5.png',
    ]

    model = YOLO(weights_path)
    predict_puzzle_box_and_keypoints(model, test_images)
