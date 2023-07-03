import os
from pathlib import Path
import sys

import cv2

def read_annotation_file(file_path, img_width, img_height):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    boxes = []
    for line in lines:
        class_id, x_center, y_center, width, height = map(float, line.strip().split())
        raw_box = (class_id, x_center, y_center, width, height)
        boxes.append(convert_to_pixel_coords(raw_box, img_width, img_height))

    return boxes

def convert_to_pixel_coords(box, img_width, img_height):
    class_id, x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height
    return (class_id, x_center, y_center, width, height)

def crop_image(image, box):
    class_id, x_center, y_center, width, height = box
    top_left = (
        int(x_center - width / 2),
        int(y_center - height / 2),
    )
    bottom_right = (
        int(x_center + width / 2),
        int(y_center + height / 2),
    )
    return image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]


def obj_detect_labels_to_cropped_pieces(images_dir, labels_dir, output_dir):
    if os.path.exists(output_dir) and any(Path(output_dir).iterdir()):
        raise Exception('output_dir is not empty.')
    else:
        os.makedirs(output_dir, exist_ok=True)

    image_files = [
        f for f in os.listdir(images_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ]

    if not len(image_files):
        raise Exception('No images found.')

    prepped_cropped_images = []

    for image_file in image_files:
        image_path = os.path.join(images_dir, image_file)
        annotation_path = os.path.join(
            labels_dir,
            Path(image_file).with_suffix('.txt'),
        )

        # if not os.path.exists(annotation_path):
        #     print('label file not found')
        #     pass

        image = cv2.imread(image_path)
        img_height, img_width = image.shape[:2]

        boxes = read_annotation_file(annotation_path, img_width, img_height)

        class_counts = {}

        for box in boxes:
            class_id, _, _, _, _ = box
            class_id = int(class_id)
            class_dir = os.path.join(output_dir, f'class_{class_id}')

            if not os.path.exists(class_dir):
                os.mkdir(class_dir)

            base_filename = f'{Path(image_file).stem}_c{class_id}'
            if class_id in class_counts:
                class_counts[class_id] += 1
                output_filename = f'{base_filename}_#{class_counts[class_id]}.png'
            else:
                class_counts[class_id] = 0
                output_filename = f'{base_filename}.png'

            cropped_img = crop_image(image, box)
            output_path = os.path.join(class_dir, output_filename)
            prepped_cropped_images.append((cropped_img, output_path))


    for cropped_img, output_path in prepped_cropped_images:
        cv2.imwrite(output_path, cropped_img)


if __name__ == '__main__':
    obj_detect_labels_to_cropped_pieces(
        images_dir='/Users/danielwaltrip/all-files/projects/ai-data/pento-exp-5/train/images',
        labels_dir='/Users/danielwaltrip/all-files/projects/ai-data/pento-exp-5/train/labels',
        output_dir='/Users/danielwaltrip/all-files/projects/ai-data/pento-exp-5/CROP_TEST',
    )
