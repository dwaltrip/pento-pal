from PIL import Image, ImageDraw

    
def add_rect_with_alpha(image, rect, outline, fill, width=2):
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        rect,
        outline=outline,
        fill=fill,
        width=width,
    )
    return Image.alpha_composite(image, overlay)
