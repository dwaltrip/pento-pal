import os
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image, ImageColor
from pathlib import Path
import torchvision.transforms

from settings import CLASS_NAMES, CLASS_MAPS, GRID
from parse_image.utils.misc import (
    generate_perlin_noise,
    is_image,
)


LABEL_VIZ = SimpleNamespace(
    rows=GRID.height,
    cols=GRID.width,
    square_size=60, # pixels
    padding=5, # pixels
    padding_color='#808080',
    scale_factor=0.9, # account for the bg of the training images
)
def get_img_size(rows, cols, square_size, padding):
    return SimpleNamespace(
        height=(square_size+padding)*rows + padding,
        width=(square_size+padding)*cols + padding,
    )
VIZ_IMG_SIZE = get_img_size(
    rows=LABEL_VIZ.rows,
    cols=LABEL_VIZ.cols,
    square_size=LABEL_VIZ.square_size,
    padding=LABEL_VIZ.padding,
)


def get_class_color(name):
    return CLASS_MAPS.name_to_color[name]

def hex_to_cv2_color(hex_color):
    r, g, b = ImageColor.getrgb(hex_color)
    bgr = (b, g, r)
    return bgr

def pad_img(img, target_size, fill=(0,0,0)):
    h, w = img.shape[:2]
    pad_height = max(target_size[0] - h, 0)
    pad_width = max(target_size[1] - w, 0)

    pad_top = pad_height // 2
    pad_bot = pad_height - pad_top

    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padding = (pad_left, pad_top, pad_right, pad_bot)
    pad_fn = torchvision.transforms.Pad(padding, fill=fill)
    return np.array(pad_fn(Image.fromarray(img)))

def combine_images(img1, img2, padding=20, bg_color='#000'):
    height1, height2 = img1.shape[0], img2.shape[0]

    # Calculate the padding for top and bottom to center the image
    diff = abs(height1 - height2)
    pad_top = diff // 2
    pad_bot = diff - pad_top

    # Create padding between the images
    color = hex_to_cv2_color(bg_color)
    horiz_padding_dim = (max(height1, height2), padding, 3)
    horiz_padding_img = np.full(horiz_padding_dim, color, dtype=np.uint8)

    # Pad the smaller image so it matches size of larger image
    if height1 < height2:
        img1 = pad_img(img1, img2.shape[:2], fill=color)
    else:
        img2 = pad_img(img2, img1.shape[:2], fill=color)

    # Concatenate the images horizontally
    return np.hstack((img1, horiz_padding_img, img2))


def validate_labels(label_path_or_content):
    if os.path.exists(label_path_or_content):
        with open(label_path_or_content, 'r') as f:
            lines = f.read().splitlines()
    else:
        lines = label_path_or_content.splitlines()

    labels = [line for line in lines if line and line[0] != '#']
    flat_labels = ''.join(labels)
    return all(flat_labels.count(c) == 5 for c in CLASS_NAMES)


def make_textured_color_block(size, base_color): 
    block_shape = (size, size)
    texture = generate_perlin_noise(block_shape, scale=0.02)
    texture_bgr = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)
    plain_block = np.full((*block_shape, 3), base_color, dtype=texture.dtype)

    # blend the Perlin noise texture with the base color
    alpha = 0.95
    return cv2.addWeighted(plain_block, alpha, texture_bgr, 1 - alpha, 0)


def visualize_labels(label_path):
    with open(label_path, 'r') as f:
        lines = [line.strip() for line in f.read().strip().splitlines()]

    labels = [line.split(' ') for line in lines if line and line[0] != '#']

    img_size = (VIZ_IMG_SIZE.height, VIZ_IMG_SIZE.width, 3)
    viz_img = np.full(
        img_size,
        hex_to_cv2_color(LABEL_VIZ.padding_color),
        dtype=np.uint8,
    )

    size = LABEL_VIZ.square_size
    padding = LABEL_VIZ.padding

    for i in range(LABEL_VIZ.rows):
        for j in range(LABEL_VIZ.cols):
            top_left = dict(
                y = i * (size + padding) + padding,
                x = j * (size + padding) + padding,
            )
            y = (top_left['y'], top_left['y'] + size)
            x = (top_left['x'], top_left['x'] + size)

            hex_color = get_class_color(labels[i][j])
            color_block = make_textured_color_block(
                size=size,
                base_color=hex_to_cv2_color(hex_color),
            )
            viz_img[y[0]:y[1], x[0]:x[1]] = color_block

    s = LABEL_VIZ.scale_factor
    scaled_size = (int(VIZ_IMG_SIZE.width * s), int(VIZ_IMG_SIZE.height * s))
    return cv2.resize(viz_img, scaled_size)


def normalize_label_file_content(content):
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

    raw_lines = content.strip().split('\n')
    cleaned_lines = [normalize_line(line) for line in raw_lines]
    cleaned_content = '\n'.join(
        line for line in cleaned_lines if keep_line(line)
    )
    return cleaned_content


def normalize_label_file(label_path):
    with open(label_path, 'r') as f: 
        content = f.read()
    with open(label_path, 'w') as file:
        file.write(normalize_label_file_content(content))


def label_images(image_dir, label_dir):
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(os.listdir(image_dir)):
        if not is_image(filename):
            continue

        label_filename = os.path.splitext(filename)[0] + '.txt'
        label_path = os.path.join(label_dir, label_filename)

        if os.path.exists(label_path):
            normalize_label_file(label_path)
            if validate_labels(label_path):
                print('Skipping... Valid existing labels found:', label_filename)
                continue

        img_path = os.path.join(image_dir, filename)

        # Load and resize the image, then display it
        img = cv2.imread(img_path)
        img = cv2.resize(img, (VIZ_IMG_SIZE.width, VIZ_IMG_SIZE.height))

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

            label_viz_img = visualize_labels(label_path)
            cv2.destroyAllWindows()
            combined_img = combine_images(img, label_viz_img, bg_color='#ba8')

            cv2.namedWindow('original-plus-label-viz', cv2.WINDOW_NORMAL)
            cv2.imshow('original-plus-label-viz', combined_img)
            cv2.waitKey(1)

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
    if len(sys.argv) != 3:
        print('Usage: python annotation_tool_v2.py <image_dir> <label_dir>')
        sys.exit(1)

    image_dir = sys.argv[1]
    label_dir = sys.argv[2]
    print('image_dir:', image_dir)
    print('label_dir:', label_dir)
    label_images(image_dir, label_dir)
