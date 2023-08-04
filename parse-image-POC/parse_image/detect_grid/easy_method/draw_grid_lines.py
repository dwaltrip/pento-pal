import math

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch

def add_grid_lines(image_tensor, color=(255, 0, 0), thickness=1):
    # Convert to (0, 255) range and to a numpy array
    image = (image_tensor.permute(1,2,0) * 255).byte().numpy()

    # Define the number of rows and cols
    rows, cols = 10, 10

    # Calculate the step sizes
    row_step = image.shape[0] // rows
    col_step = image.shape[1] // cols

    # Hack to make cv2.line happy. It was getting angry about the image type,
    #   even though we already converted to a numpy array.
    image = image.copy()

    # Draw the grid
    for i in range(rows):
        cv2.line(
            image,
            (0, i * row_step),
            (image.shape[1], i * row_step),
            color=color,
            thickness=thickness,
        )
    for i in range(cols):
        cv2.line(
            image,
            (i * col_step, 0),
            (i * col_step, image.shape[0]),
            color=color,
            thickness=thickness,
        )

    return image


def show_images(images):
    if images[0].shape[0] == 3:
        def permute(x):
            if isinstance(x, np.ndarray):
                return x.transpose(1,2,0)
            elif isinstance(x, torch.Tensor):
                return x.permute(1,2,0)
            else:
                raise ValueError(f'Unexpected type: {type(x)}')
        images = [permute(img) for img in images]

    if isinstance(images[0], torch.Tensor):
        images = [img.numpy() for img in images]

    num_images = len(images)
    rows = int(math.floor(math.sqrt(num_images)))
    cols = int(math.ceil(num_images / rows))
    fig, axs = plt.subplots(
        rows,
        cols,
        figsize=(3*num_images, 3), 
        gridspec_kw={'hspace': 0.4}
    )
    axs = axs.ravel() # Flatten the axs array to make indexing easier

    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].set_title(f'Image {i+1}', fontsize=12)
        axs[i].axis('off') # Removes numbers from axes
    plt.show()


def add_grid_lines_and_show(images):
    images_with_grid = [add_grid_lines(img) for img in images]
    show_images(images_with_grid)
