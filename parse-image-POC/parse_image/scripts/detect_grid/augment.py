import os
from collections import defaultdict

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F

from settings import PROJECT_ROOT


def rotate_tensor_image(t_img, angle):
    return F.affine(
        t_img,
        angle=angle,
        translate=[0, 0],
        scale=1.0,
        shear=[0, 0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )


rand_angle = lambda x: torch.randint(-x, x, (1,)).item()
# ROTATION_AMOUNTS = [0, rand_angle(), rand_angle()]
ROTATION_AMOUNTS = [0, 20, -20]

def get_augmentations(base_img, base_label):
    def rotate(t_img, label, amount):
        assert amount <= 30, 'Assumes small rotation, else labels will be wrong'
        # Because it is a small rotation, we use the original label
        return (rotate_tensor_image(t_img, amount), label)

    # `dims` for the image is 1 more than label, due to color dimension.
    # Image dims are: [C, H, W]

    base_item = (base_img, base_label)

    all_possible_flips = [(*base_item, 'BASE'), *get_flips(*base_item)]

    flips_plus_rotations = [
        (*rotate(aug_img, aug_label, rot_amt), f'{desc} + r{rot_amt}')
        for (aug_img, aug_label, desc) in all_possible_flips
        for rot_amt in ROTATION_AMOUNTS
    ]
    return flips_plus_rotations


def get_flips(t_img, label):
    # `dims` for the image is 1 more than label, due to color dimension.
    # Image dims are: [C, H, W]
    return [
        # Horizontal flip
        (torch.flip(t_img, dims=[2]), torch.flip(label, dims=[1]), 'Horiz'),
        # Vertical flip
        (torch.flip(t_img, dims=[1]), torch.flip(label, dims=[0]), 'Vert'),
        # Horizontal + Vertical flip
        (torch.flip(t_img, dims=[1,2]), torch.flip(label, dims=[0,1]), 'Horiz + Vert'),
    ]


if __name__ == '__main__':
    # img = torch.randint(0, 20, (2, 2, 4))
    # label = torch.randint(60, 99, (2, 2))

    # print('img:\n', img)
    # print('label:\n', label)

    image_dir = os.path.join(PROJECT_ROOT, 'example-images')
    filenames = [f'pento-{i}.png' for i in range(1,6)] 
    # filenames = [f'pento-1.png']
    example_images = [os.path.join(image_dir, f) for f in filenames]

    img = F.to_tensor(Image.open(example_images[0]))
    label = torch.randint(60, 99, (2, 2))

    def show_images_OLD(images, titles):
        num_images = len(images)
        cols = max(6, num_images // 2)
        rows = num_images // cols 
        rows += num_images % cols

        fig = plt.figure(figsize=(3*num_images, 3))
        for i, (img, title) in enumerate(zip(images, titles)):
            ax = fig.add_subplot(rows, cols, i+1)
            # ax = fig.add_subplot(1, len(images), i+1)
            ax.set_title(title, fontsize=10)
            ax.imshow(img)
        plt.show()
    

    def show_images(images, titles):
        num_images = len(images)
        cols = max(6, num_images // 2)
        rows = num_images // cols 
        rows += num_images % cols

        fig, axs = plt.subplots(rows, cols, figsize=(3*num_images, 3), 
                                gridspec_kw={'hspace': 0.5}) # Increase spacing between rows
        axs = axs.ravel() # Flatten the axs array to make indexing easier
        for i, (img, title) in enumerate(zip(images, titles)):
            axs[i].set_title(title, fontsize=10)
            axs[i].imshow(img)
            axs[i].axis('off') # Removes numbers from axes
        plt.show()


    aug_images = []
    aug_titles = []
    for i, (aug_img, aug_label, desc) in enumerate(get_augmentations(img, label)):
        title = f'#{i+1}, {desc}'

        aug_images.append(aug_img.permute(1,2,0))
        aug_titles.append(title)
        # aug_images_and_titles.append((aug_img.permute(1,2,0), title))
        print()
        print(f'--- {title} ---')
        # print('aug_img:\n', aug_img)
        print('aug_label:\n', aug_label)

        # plt.imshow(aug_img.permute(1,2,0))
        # plt.title(title)
        # plt.show()

    show_images(aug_images, aug_titles)
