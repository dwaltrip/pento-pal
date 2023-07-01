import os
from ultralytics import YOLO
from PIL import Image, ImageDraw

BASE_MODEL__MEDIUM = 'yolov8m.pt'
BASE_MODEL__LARGE = 'yolov8l.pt'

# previous_weights_path = 'runs/detect/train8/weights/best.pt'
previous_weights_path = 'zzzzzz'

class_color_map = {
  "F": "#f196f1",
  "U": "#efab2c",
  "I": "#000000",
  "L": "#87bce8",
  "N": "#f53f2a",
  "P": "#e4d6b2",
  "T": "#457ddf",
  "V": "#6b43c8",
  "W": "#d7d9db",
  "X": "#3aae45",
  "Y": "#e7ee40",
  "Z": "#9fdd92",
}
class_color_map = {
    key: (color + '{:02x}'.format(175))
    for key, color in class_color_map.items()
}

def ltr_key(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return (y1, x1, y2, x2)

def predict_image(model, image_path):
    results = model.predict(image_path)
    result = results[0]
    print('num objects:', len(result.boxes))

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for i, box in enumerate(sorted(result.boxes, key=ltr_key)):
        class_name = result.names[box.cls.item()]
        conf = box.conf.item()

        if conf < 0.7:
            continue

        print(
            'class:', class_name,
            '-- conf:', round(conf, 3),
            '-- coords:', [int(c) for c in box.xyxy[0].tolist()],
        )
        draw.rectangle(
            box.xyxy[0].tolist(),
            outline=class_color_map[class_name],
            width=2,
        )

    image.show()


if __name__ == '__main__':

    if os.path.isfile(previous_weights_path):
        model = YOLO(previous_weights_path)
    else:
        # model = YOLO(f'weights/{BASE_MODEL__MEDIUM}')
        model = YOLO(f'weights/{BASE_MODEL__LARGE}')
        model.train(
            data='experiments/experiment-1.yaml',
            device='mps',
            epochs=20,
            # batch=8,
        )

    test_images = [
        'example-images/pento-1.png',
        'example-images/pento-2.png',
        'example-images/pento-3.png',
        'example-images/pento-4.png',
        'example-images/pento-5.png',
    ]

    for image_path in test_images:
        predict_image(model, image_path)
