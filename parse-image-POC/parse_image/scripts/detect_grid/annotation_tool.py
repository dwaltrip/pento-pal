import os
import subprocess
from PIL import Image
from pathlib import Path

import cv2

from parse_image.scripts.detect_grid.config import IMAGE_DIR, LABEL_DIR
from parse_image.scripts.detect_grid.dataset import is_image


classes = ['f', 'i', 'l', 'n', 'p', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def label_images(image_dir, label_dir):
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    for i, filename in enumerate(os.listdir(image_dir)):
        if is_image(filename):
            label_filename = os.path.splitext(filename)[0] + '.txt'
            label_path = os.path.join(label_dir, label_filename)

            if os.path.exists(label_path):
                raise Exception(f'Label file already exists for: {filename}')

            img_path = os.path.join(image_dir, filename)

            # Load an image, create a window, and display image
            img = cv2.imread(img_path)
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', img)
            cv2.waitKey(1)

            with open(label_path, 'w') as f:
                f.write('# Enter labels for image: ' + filename + '\n')
                f.write('# Format: 10 lines with 6 class labels each\n')
                f.write(f'# Classes: {', '.join(classes)}\n')

            subprocess.call(['vim', label_path])

            # Close all OpenCV windows
            cv2.destroyAllWindows()

            print('Saved labels for ' + filename)

    print('Finished labeling all images')


if __name__ == '__main__':
    label_images(IMAGE_DIR, LABEL_DIR)
