import cv2
import numpy as np
from noise import snoise2


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
