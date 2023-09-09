import cv2
import numpy as np
import PIL
from PIL import Image
import torch

from parse_image.data.points import Point


def add_grid_lines(
    src_image,
    rows,
    cols,
    color=(255, 0, 0),
    thickness=1,
    rect=None,
):
    is_pil_image = isinstance(src_image, PIL.Image.Image)
    if is_pil_image:
        image = np.array(src_image)
    elif isinstance(src_image, torch.Tensor):
        # Convert to (0, 255) range and to a numpy array
        image = (src_image.permute(1,2,0) * 255).byte().numpy()
    else:
        raise ValueError(f'Unexpected type: {type(src_image)}')

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    _r, _g, _b = color
    color = (_b, _g, _r) # cv2 uses BGR instead of RGB

    if not rect:
        rect = dict(
            top_left = Point(0, 0),
            height = image.shape[0],
            width = image.shape[1],
        )
    rect['height'] = int(rect['height'])
    rect['width'] = int(rect['width'])

    cell_height = rect['height'] / rows
    cell_width = rect['width'] / cols
    top_left = Point(*map(int, rect['top_left']))
    bot_right = Point(
        x=top_left.x + rect['width'],
        y=top_left.y + rect['height'],
    )

    # Hack to make cv2.line happy. It was getting angry about the image type,
    #   even though we already converted to a numpy array.
    image = image.copy()

    for i in range(rows+1):
        y = int(round(top_left.y + (i * cell_height)))
        start = (top_left.x, y)
        end = (bot_right.x, y)
        cv2.line(image, start, end, color=color, thickness=thickness)

    for i in range(cols+1):
        x = int(round(top_left.x + (i * cell_width)))
        start = (x, top_left.y)
        end = (x, bot_right.y)
        cv2.line(image, start, end, color=color, thickness=thickness)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if is_pil_image:
        return Image.fromarray(image)
    else:
        return image
