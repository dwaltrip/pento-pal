from PIL import Image, ImageDraw
import numpy as np
import torch
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms


to_tensor = transforms.ToTensor()

IMAGE_SIDE_LEN = 224
IMAGE_SIZE = (3, IMAGE_SIDE_LEN, IMAGE_SIDE_LEN)


def tensor_to_PIL(img_t):
    return Image.fromarray((img_t * 255).byte().permute(1, 2, 0).numpy())

def resize_and_pad(image, target_size, pad_color=(0, 0, 0)):
    if isinstance(image, torch.Tensor):
        image = tensor_to_PIL(image)

    # Resize
    w, h = image.size
    if h > w:
        new_h, new_w = target_size, int(target_size * w / h)
    else:
        new_h, new_w = int(target_size * h / w), target_size
    resized_image = image.resize((new_w, new_h), Image.BILINEAR)
    # return resized_image

    # Calculate padding
    left_padding = (target_size - new_w) // 2
    right_padding = target_size - new_w - left_padding
    top_padding = (target_size - new_h) // 2
    bottom_padding = target_size - new_h - top_padding

    # Pad
    padding = (left_padding, top_padding, right_padding, bottom_padding)  # left, top, right, bottom
    padded_image = transforms.Pad(padding, fill=pad_color)(resized_image)

    return padded_image


def predict_image(model, image):
    image_data = to_tensor(image).unsqueeze(0).numpy()
    results = model(torch.from_numpy(image_data))


if __name__ == '__main__':
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    image_path = '/Users/danielwaltrip/Desktop/test-image.png'

    raw_image = Image.open(image_path)
    raw_image_t = to_tensor(raw_image)
    raw_image_t = raw_image_t[:3] # remove alpha channel
    img_mean = tuple((raw_image_t.mean(dim=(1,2)) * 255).byte().tolist())

    # image = resize_and_pad(raw_image_t, IMAGE_SIDE_LEN)
    image = resize_and_pad(raw_image_t, IMAGE_SIDE_LEN, pad_color=img_mean)
    image.show()

