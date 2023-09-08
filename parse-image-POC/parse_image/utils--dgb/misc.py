import cv2
import numpy as np
from noise import snoise2
import torch


IMAGE_EXTS = ['.png']

def is_image(filename):
    return any([filename.endswith(ext) for ext in IMAGE_EXTS])


def get_output_shape(model, image_size):
    with torch.no_grad():
        dummy_tensor = torch.zeros((1,) + image_size)  # create a dummy tensor
        output = model(dummy_tensor)  # pass the tensor through the model

    return output.shape[1:]


def generate_perlin_noise(size, scale=0.1):
    noise = np.zeros(size)
    
    for i in range(size[0]):
        for j in range(size[1]):
            noise[i][j] = snoise2(i * scale, j * scale)

    # Scale the noise values to the range 0-255
    noise_texture = cv2.normalize(
        noise, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
    )
    return noise_texture


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
