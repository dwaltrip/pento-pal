import os

from torchvision import transforms
from PIL import Image


def resize_and_pad(image, target_size):
    # Resize
    w, h = image.size
    if h > w:
        new_h, new_w = target_size, int(target_size * w / h)
    else:
        new_h, new_w = int(target_size * h / w), target_size
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    # Calculate padding
    left_padding = (target_size - new_w) // 2
    right_padding = target_size - new_w - left_padding
    top_padding = (target_size - new_h) // 2
    bottom_padding = target_size - new_h - top_padding
    
    # Pad
    padding = (left_padding, top_padding, right_padding, bottom_padding)  # left, top, right, bottom
    padded_image = transforms.Pad(padding)(resized_image)

    return padded_image


# for testing purposes
if __name__ == '__main__':
    from parse_image.scripts.detect_grid.config import IMAGE_SIDE_LEN

    image_path = os.path.join(
        '/Users/danielwaltrip/all-files/projects/ai-data',
        'pento-custom-model-EXPERIMENT-1/images',
        '00c5d3cb-IMG_2459.png',
    )
    # Use the function
    original = Image.open(image_path)

    prepped = resize_and_pad(original, target_size=IMAGE_SIDE_LEN)

    original.show()
    prepped.show()
