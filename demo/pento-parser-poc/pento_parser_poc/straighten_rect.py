import cv2
import numpy as np
from PIL import Image


def straighten_rect(pil_image, corners, aspect_ratio, padding=50):
    """
    De-warps a rectangular region in the given image.
    Params:
        pil_image: Input image (PIL Image)
        corners: List of four corner points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        aspect_ratio: Target aspect ratio (width/height)
    Returns:
        De-warped, cropped image
    """
    image = np.array(pil_image)
    # Detected corners from the image
    src_pts = np.array(corners, dtype=np.float32)

    width_dist = max(
        cv2.norm(src_pts[0] - src_pts[1]),
        cv2.norm(src_pts[2] - src_pts[3]),
    )
    # Calculating width and height based on original aspect ratio
    width = int(width_dist)
    height = int(width / aspect_ratio)

    pad = padding
    padded_dims = (width + 2 * pad, height + 2 * pad)

    # Idealized rectangle points with padding
    dst_pts = np.array([
        [pad, pad],
        [pad + width, pad],
        [pad + width, pad + height],
        [pad, pad + height]
    ], dtype=np.float32)

    # Find homography and apply to the image
    H, _ = cv2.findHomography(src_pts, dst_pts)
    dewarped_image = cv2.warpPerspective(image, H, padded_dims)

    return Image.fromarray(dewarped_image)