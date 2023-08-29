from parse_image.detect_puzzle_box.labels import KEYPOINT_COLORS


def draw_corners(draw, corners, size=3):
    for i, keypoint in enumerate(corners):
        draw_dot(draw, keypoint, size, KEYPOINT_COLORS[i])


def draw_dot(draw, point, size, fill):
    x, y = point
    top_left = (x - size, y - size)
    bot_right = (x + size, y + size)
    draw.ellipse((top_left, bot_right), fill=fill)
