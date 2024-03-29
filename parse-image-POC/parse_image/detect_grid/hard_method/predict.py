import os

import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms

from settings import PROJECT_ROOT

from settings import AI_DATA_DIR, CLASS_MAPS
from parse_image.utils.resize_as_square import resize_as_square
from parse_image.detect_grid.hard_method.config import (
    IMAGE_SIDE_LEN,
    TRAINED_MODEL_SAVE_PATH,
    IMAGE_DIR,
)
from parse_image.detect_grid.hard_method.model import get_custom_model


def prep_image(img_path):
    img = Image.open(img_path)
    img = resize_as_square(img, side_len=IMAGE_SIDE_LEN)
    # remove alpha channel, if present. add batch dimension
    img_t = transforms.ToTensor()(img)[:3].unsqueeze(0)
    return img_t, img


def predict_grid(model, img_path, idx, debug=False):
    img_t, img = prep_image(img_path)
    model.eval()
    with torch.no_grad():
        raw_output = model(img_t)
    output = raw_output.squeeze(0)
    print('output.shape:', output.shape)

    if debug:
        def pp_preds(preds):
            print(', '.join([f'{x:6.2f}' for x in preds]))

        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                preds = [x.item() for x in output[i][j]]
                print('-+-' * 10)
                pp_preds(preds)
                pp_preds(sorted(preds))
        print('-+-' * 10)

    result = output.argmax(dim=-1)

    def result_to_cls_names(raw_result):
        return [
            [CLASS_MAPS.class_id_to_name[pred.item()] for pred in row]
            for row in raw_result
        ]

    result_w_class_names = result_to_cls_names(result)

    print('--- result ---')
    if debug:
        print(result)
    print('\n'.join(str(row) for row in result_w_class_names))

    plt.imshow(img)
    plt.title(f'Image {idx+1}')
    plt.show()
    # img.show()


if __name__ == '__main__':
    print('--- TRAINED_MODEL_SAVE_PATH:', TRAINED_MODEL_SAVE_PATH)
    state_dict = torch.load(TRAINED_MODEL_SAVE_PATH)
    model = get_custom_model()
    model.load_state_dict(state_dict)

    image_dir = os.path.join(PROJECT_ROOT, 'example-images')

    example_images = [
        os.path.join(image_dir, f)
        for f in [f'pento-{i}.png' for i in range(1,6)]
    ]
    training_data_images = [
        'b8ffe330-IMG_2358.png',
        '70398230-IMG_2422.png',
        '302458e1-IMG_2457.png'
        '2530f2a6-IMG_2965.png',
        'ec882312-IMG_2447.png',
    ]

    example_images = [
        # os.path.join(AI_DATA_DIR, 'detect-grid-hard--2023-08-01/images/IMG_2926.png'),
        os.path.join(AI_DATA_DIR, 'detect-grid-hard--2023-08-01/images/IMG_2991.png'),
        os.path.join(AI_DATA_DIR, 'detect-grid-hard--2023-08-01/images/IMG_2772.png'),
    ]
    images = [os.path.join(IMAGE_DIR, f) for f in example_images]
    # images = [os.path.join(IMAGE_DIR, f) for f in training_images]
    # images = [os.path.join(image_dir, 'pento-5.png')]

    for i, img_path in enumerate(images):
        # Image.open(img_path).show()
        print(img_path)
        predict_grid(model, img_path, i, debug=False)
