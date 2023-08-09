from PIL import ImageColor


def hex_to_bgr(hex_color):
    r, g, b = ImageColor.getrgb(hex_color)
    bgr = (b, g, r)
    return bgr

def hex_to_rgb(hex_color):
    return ImageColor.getrgb(hex_color)