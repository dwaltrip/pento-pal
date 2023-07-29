from PIL import Image, ImageDraw

def draw_bboxes(image, bboxes, color_map=None):
    draw = ImageDraw.Draw(image)

    for (box, name) in bboxes:
        if color_map:
            color = color_map.get(name)
        else:
            color = 'blue'

        draw.rectangle(
            box.xyxy[0].tolist(),
            outline=class_color_map[class_name],
            width=2,
        )


if __name__ == '__main__':
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    from torchvision.transforms import ToTensor
    import numpy as np

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    image_path = '/Users/danielwaltrip/Desktop/test-image.png'
    image = Image.open(image_path)

    image_data = (ToTensor()(image)).unsqueeze(0).numpy()
    results = model(torch.from_numpy(image_data))
    print('dir(results):', dir(results))
    result = results[0]

    def ltr_key(box):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        return (y1, x1, y2, x2)

    # draw = ImageDraw.Draw(image)

    # for box in sorted(result.boxes, key=ltr_key):
    #     class_name = result.names[box.cls.item()]
    #     conf = box.conf.item()

    #     if conf < 0.7:
    #         continue

    #     print(
    #         'class:', class_name,
    #         '-- conf:', round(conf, 3),
    #         '-- coords:', [int(c) for c in box.xyxy[0].tolist()],
    #     )
    #     draw.rectangle(
    #         box.xyxy[0].tolist(),
    #         outline=class_color_map[class_name],
    #         width=2,
    #     )

    # image.show()
