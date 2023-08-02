import torch
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F


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
ROTATION_AMOUNTS = [0, 15, -15]

def get_image_augmentations(base_img, base_label):
    def rotate(t_img, amount):
        assert amount < 25, 'Assumes small rotation, else labels will be wrong'
        # Because it is a small rotation, we use the original label
        return (rotate_tensor_image(t_img, amount), label)

    # `dims` for the image is 1 more than label, due to color dimension.
    # Image dims are: [C, H, W]

    base_item = (base_img, base_label)

    all_possible_flips = [(*base_item, 'BASE'), *get_flips(*base_item)]
    return all_possible_flips

    flips_plus_rotations = [
        (*rotate(img, rot_amt), f'{desc} + r{rot_amt}')
        for (img, label, desc) in all_possible_flips
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
    img = torch.randint(0, 20, (2, 2, 4))
    label = torch.randint(60, 99, (2, 2))

    print('img:\n', img)
    print('label:\n', label)

    for (aug_img, aug_label, desc) in get_image_augmentations(img, label):
        print()
        print(f'--- {desc} ---')
        print('aug_img:\n', aug_img)
        print('aug_label:\n', aug_label)
