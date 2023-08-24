import os
from PIL import Image, ImageDraw

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
        class_name = result.names[box.cls.item()].upper()
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

def predict_piece_detection(model, image_paths):
    for image_path in image_paths:
        predict_image(model, image_path)
