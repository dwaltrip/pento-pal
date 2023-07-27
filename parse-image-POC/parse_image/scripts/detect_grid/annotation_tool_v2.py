import os
import subprocess

import cv2
import numpy as np
from PIL import ImageColor
from pathlib import Path

from parse_image.scripts.detect_grid.config import (
    IMAGE_DIR,
    LABEL_DIR,
    CLASS_NAMES,
    CLASS_MAPS,
)
from parse_image.scripts.detect_grid.dataset import is_image


def get_class_color(name):
    return CLASS_MAPS.name_to_color[name]

def hex_to_cv2_color(hex_color):
    r, g, b = ImageColor.getrgb(hex_color)
    bgr = (b, g, r)
    return bgr


def validate_labels(label_path):
    with open(label_path, 'r') as f:
        lines = f.read().splitlines()
    labels = [line for line in lines if line and line[0] != '#']
    flat_labels = ''.join(labels)
    return all(flat_labels.count(c) == 5 for c in CLASS_NAMES)


def visualize_labels(label_path):
    num_rows = 10
    num_cols = 6
    size = 50 # pixels
    padding = 4 # pixels
    padding_color = '#808080'

    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.read().strip().splitlines()]
    labels = [line.split(' ') for line in lines if line and line[0] != '#']

    img_size = (
        (size + padding) * num_rows + padding,
        (size + padding) * num_cols + padding,
        3,
    )
    vis_img = np.full(img_size, hex_to_cv2_color(padding_color), dtype=np.uint8)

    for i in range(num_rows):
        for j in range(num_cols):
            top_left = dict(
                y = i * (size + padding) + padding,
                x = j * (size + padding) + padding,
            )
            y = (top_left['y'], top_left['y'] + size)
            x = (top_left['x'], top_left['x'] + size)
            
            hex_color = get_class_color(labels[i][j])
            vis_img[y[0]:y[1], x[0]:x[1]] = hex_to_cv2_color(hex_color)

    cv2.namedWindow('label-verification', cv2.WINDOW_NORMAL)
    cv2.imshow('label-verification', vis_img)
    cv2.waitKey(1)


def normalize_label_file(label_path):
    with open(label_path, 'r') as f: 
        content = f.read().strip()

    def keep_line(line):
        line = line.strip()
        # drop blank lines
        if not line:
            return False
        # drop comments
        return line[0] != '#' if line else False

    def normalize_line(line):
        # Separate every char by a single space
        return ' '.join(line.strip().replace(' ', ''))

    raw_lines = content.split('\n')
    cleaned_lines = [normalize_line(line) for line in raw_lines]
    cleaned_content  = '\n'.join(
        line for line in cleaned_lines if keep_line(line)
    )

    with open(label_path, 'w') as file:
        file.write(cleaned_content)


def label_images(image_dir, label_dir):
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(os.listdir(image_dir)):
        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path):
            normalize_label_file(label_path)
            if validate_labels(label_path):
                print('Skipping... Valid existing labels found:', label_filename)
                continue

        img_path = os.path.join(image_dir, filename)

        # Load an image, create a window, and display image
        img = cv2.imread(img_path)
        cv2.namedWindow('training-img', cv2.WINDOW_NORMAL)
        cv2.imshow('training-img', img)
        cv2.waitKey(1)

        while True:
            subprocess.call(['vim', label_path])
            normalize_label_file(label_path)

            if not validate_labels(label_path):
                with open(label_path, 'r+') as f:
                    content = f.read()
                    f.seek(0, 0)
                    error_msg = '\n'.join([
                        '# ERROR: Invalid label file!',
                        '# It should have exactly 5 occurrences of each class.',
                        '# Please correct the labels.',
                        '# --------------------------',
                    ])
                    f.write(error_msg + '\n' + content)
                continue

            visualize_labels(label_path)
            user_input = input(' '.join([
                'If visual verification is correct, press ENTER.',
                'If not, press any other key to edit labels: ',
            ]))
            if user_input == '':
                break

        cv2.destroyAllWindows()
        print('Saved labels for ' + filename)

    print('Finished labeling all images')


if __name__ == '__main__':
    label_images(IMAGE_DIR, LABEL_DIR)
