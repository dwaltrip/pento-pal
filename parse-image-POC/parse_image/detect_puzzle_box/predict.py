import os
import sys

from PIL import Image, ImageDraw
from ultralytics import YOLO


def predict_image(model, image_path):
    results = model.predict(image_path)
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
    # use different colors to verify the keypoints are in the correct order
    for i, keypoint in enumerate(keypoints):
        draw_dot(draw, keypoint, 3, 'red')

    image.show()

def draw_dot(draw, point, size, fill):
    x, y = point
    top_left = (x - size, y - size)
    bot_right = (x + size, y + size)
    draw.ellipse((top_left, bot_right), fill=fill)


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
