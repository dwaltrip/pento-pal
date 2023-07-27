import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

from parse_image.scripts.detect_grid.config import IMAGE_DIR, LABEL_DIR

def label_images(image_dir, label_dir):
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = mpimg.imread(os.path.join(image_dir, filename))
            imgplot = plt.imshow(img)
            plt.show()

            with open('temp.txt', 'w') as f:
                f.write('# Enter labels for image: ' + filename + '\n')
                f.write('# Format: 10 lines with 6 integers (0-11) each\n')

            subprocess.call(['vim', 'temp.txt'])

            with open('temp.txt', 'r') as f:
                labels = f.read()

            label_filename = os.path.splitext(filename)[0] + '.txt'
            with open(os.path.join(label_dir, label_filename), 'w') as f:
                f.write(labels)

            print('Saved labels for ' + filename)

    print('Finished labeling all images')

if __name__ == "__main__":
    label_images(IMAGE_DIR, LABEL_DIR)
