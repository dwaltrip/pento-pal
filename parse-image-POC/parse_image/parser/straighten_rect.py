import cv2
import numpy as np
from PIL import Image


def straighten_rect(pil_image, corners, aspect_ratio):
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
    height_dist = max(
        cv2.norm(src_pts[0] - src_pts[3]),
        cv2.norm(src_pts[1] - src_pts[2]),
    )

    # ----------------------------------------------------------------------
    # TODO: we aren't using the height_dist at all... not sure about that...
    # Need to ask GPT-4 to explain this code
    # ----------------------------------------------------------------------
    width = int(width_dist)
    height = int(width / aspect_ratio)

    # Idealized rectangle points
    dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    # Find homography and apply to the image
    H, _ = cv2.findHomography(src_pts, dst_pts)
    dewarped_image = cv2.warpPerspective(image, H, (width, height))

    return Image.fromarray(dewarped_image)