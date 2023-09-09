import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from parse_image.detect_grid.common.viz import add_grid_lines


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
    images_with_grid = [
        add_grid_lines(img, rows=10, cols=10)
        for img in images
    ]
    show_images(images_with_grid)
