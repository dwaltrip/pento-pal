import os

from torchvision import transforms
from PIL import Image


# TODO: rename to `resize_as_square` or something
def resize_and_pad(image, side_len):
    """ Resize to a square and pad """
    # Resize
    w, h = image.size
    if h > w:
        new_h, new_w = side_len, int(side_len * w / h)
    else:
        new_h, new_w = int(side_len * h / w),side_len 
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    # Calculate padding
    left_padding = (side_len - new_w) // 2
    right_padding = side_len - new_w - left_padding
    top_padding = (side_len - new_h) // 2
    bottom_padding = side_len - new_h - top_padding
    
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

    prepped = resize_and_pad(original, side_len=IMAGE_SIDE_LEN)

    original.show()
    prepped.show()
