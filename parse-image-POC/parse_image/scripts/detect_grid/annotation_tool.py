import os
import subprocess
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

def label_images(image_dir, label_dir):
    # Ensure the label directory exists
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    # Loop through all images in the image directory
    for filename in os.listdir(image_dir):
        # Only process files with image extensions
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Load and display the image
            img = mpimg.imread(os.path.join(image_dir, filename))
            imgplot = plt.imshow(img)
            plt.show()

            # Create a temporary file for the user to input labels
            with open('temp.txt', 'w') as f:
                f.write('# Enter labels for image: ' + filename + '\n')
                f.write('# Format: 10 lines with 6 integers (0-11) each\n')

            # Open the temporary file in vim and wait for the user to finish editing
            subprocess.call(['vim', 'temp.txt'])

            # Load the edited labels from the temporary file
            with open('temp.txt', 'r') as f:
                labels = f.read()

            # Save the labels to a new file with the same name as the image file (but with .txt extension)
            label_filename = os.path.splitext(filename)[0] + '.txt'
            with open(os.path.join(label_dir, label_filename), 'w') as f:
                f.write(labels)

            print('Saved labels for ' + filename)

    print('Finished labeling all images')

# TODO: receive the path as script arg

# Call the function with your image and label directories
label_images('/path/to/your/images', '/path/to/your/labels')
