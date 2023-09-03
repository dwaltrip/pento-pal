import os
from pathlib import Path

from PIL import Image, ImageDraw

from settings import AI_DATA_DIR, CLASS_MAPS, GRID
from parse_image.parser.get_puzzle_box_corners import get_puzzle_box_corners
from parse_image.parser.straighten_rect import straighten_rect
from parse_image.parser.models import load_piece_detection_model


ASPECT_RATIO = GRID.width / GRID.height


def visualize_bounding_boxes(model, image):
    results = model.predict(image, verbose=False)
    result = results[0]
    print(f'# objects: {len(result.boxes)} --', end=' ')

    draw = ImageDraw.Draw(image)

    for box in sorted(result.boxes, key=lambda box: box.conf.item(), reverse=True):
        class_name = result.names[box.cls.item()].upper()
        conf = box.conf.item()

        if conf < 0.5: continue
        print(f'{class_name}: {conf*100:0.0f}%', end=', ')

        draw.rectangle(
            box.xyxy[0].tolist(),
            outline=CLASS_MAPS.name_to_color[class_name.lower()],
            width=2,
        )
    print()
    print(','.join(
        sorted(list(set(
            [result.names[box.cls.item()].upper() for box in result.boxes]
        )))
    ))
    print(','.join(
        sorted(list(
            [result.names[box.cls.item()].upper() for box in result.boxes]
        ))
    ))
    print()


def investigate_images(model, image_paths):
    for image_path in image_paths:
        image = Image.open(image_path)
        # image.show()

        puzzle_corners = get_puzzle_box_corners(image)
        # pc = puzzle_corners
        # puzzle_corners = (pc[0], pc[1], pc[3], pc[2])
        normalized_image = straighten_rect(image, puzzle_corners, ASPECT_RATIO)

        print(f'Image: {Path(image_path).name}', end=' -- ')
        visualize_bounding_boxes(model, normalized_image)

        # if 'IMG_3041' in image_path:
        #     normalized_image.show()


FAILED_IMAGES = [
    'IMG_2869.png',
    'IMG_2399.png',
    'IMG_2372.png',
    'IMG_3247.png',
    'IMG_3251.png',
    'IMG_3244.png',
    'IMG_2947.png',
    'IMG_3308.png',
    'IMG_3297.png',
    'IMG_3042.png',
    'IMG_3243.png',
    'IMG_3041.png',
    'IMG_2810.png',
    'IMG_2811.png',
    'IMG_2379.png',
    'IMG_2618.png',
    'IMG_2753.png',
]
DATA_DIRNAME = 'detect-grid-hard--2023-08-01'


def main():
    image_dir = os.path.join(AI_DATA_DIR, DATA_DIRNAME, 'images')
    image_paths = [
        os.path.join(image_dir, filename) for filename in os.listdir(image_dir)
        if filename in FAILED_IMAGES
    ]

    model = load_piece_detection_model()
    # foo_baz(model, image_paths[:3])
    investigate_images(model, image_paths)


if __name__ == '__main__':
    main()
