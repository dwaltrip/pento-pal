from PIL import Image
from torchvision import transforms


def resize_as_square(image, side_len):
    """Rectangular images will be padded not cropped"""
    w, h = image.size
    if h > w:
        new_h, new_w = side_len, int(side_len * w / h)
    else:
        new_h, new_w = int(side_len * h / w),side_len 
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)

    left_padding = (side_len - new_w) // 2
    right_padding = side_len - new_w - left_padding
    top_padding = (side_len - new_h) // 2
    bottom_padding = side_len - new_h - top_padding
    
    padding = (left_padding, top_padding, right_padding, bottom_padding)
    padded_image = transforms.Pad(padding)(resized_image)

    return padded_image
