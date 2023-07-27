import os

from PIL import Image
import torch
from torchvision import transforms

from settings import PROJECT_ROOT

from parse_image.scripts.detect_grid.config import (
    CLASS_NAMES,
    IMAGE_SIDE_LEN,
    TRAINED_MODEL_SAVE_PATH,
)
from parse_image.scripts.detect_grid.model import get_custom_model
from parse_image.scripts.detect_grid.prep_images import resize_and_pad


def prep_image(img_path):
    img = Image.open(img_path)
    img = resize_and_pad(img, target_size=IMAGE_SIDE_LEN)
    img_t = transforms.ToTensor()(img).unsqueeze(0)
    return img_t, img


def predict_grid(model, img_path):
    img_t, img = prep_image(img_path)
    output = model(img_t)
    result = output.argmax(dim=-1).squeeze(0)

    def result_to_cls_names(raw_result):
        label_to_name = lambda x: CLASS_NAMES[x.item()]
        return [
            [label_to_name(label) for label in row]
            for row in raw_result
        ]

    result_w_class_names = result_to_cls_names(result)

    print('--- result ---')
    print(result)
    print('\n'.join(str(row) for row in result_w_class_names))

    img.show()


if __name__ == '__main__':
    state_dict = torch.load(TRAINED_MODEL_SAVE_PATH)
    model = get_custom_model() # instance of `nn.Module`
    model.load_state_dict(state_dict)
    
    image_dir = os.path.join(PROJECT_ROOT, 'example-images')
    img_path = os.path.join(image_dir, 'pento-5.png')

    predict_grid(model, img_path)
