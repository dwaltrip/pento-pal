import math
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import utils
from torchvision.transforms import ToTensor
from torchvision.models import resnet50, ResNet50_Weights

from settings import RESNET_IMAGE_SIDE_LEN, AI_DATA_DIR
from parse_image.utils.resize_and_pad import resize_and_pad

to_tensor = ToTensor()


class ResnetBackboneWithLayerOutputs(nn.Module):
    def __init__(self):
        super().__init__()
        vanilla_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        LAST_LAYERS_TO_REMOVE = 3
        modules_with_names = list(
            vanilla_resnet.named_children()
        )[:-LAST_LAYERS_TO_REMOVE]
    
        self.module_names_by_idx = {
            i: name for i, (name, _) in enumerate(modules_with_names)
        }
        backbone_modules = [module for name, module in modules_with_names]
        self.resnet_backbone = nn.Sequential(*backbone_modules)

    def forward(self, x):
        layer_outputs = []
        for i, module in enumerate(self.resnet_backbone):
            x = module(x)
            layer_outputs.append((x, self.module_names_by_idx[i]))
        return x, layer_outputs


def get_model():
    return ResnetBackboneWithLayerOutputs()

def get_model_OLd():
    vanilla_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet_modules = list(vanilla_resnet.children())
    resnet_backbone = nn.Sequential(*resnet_modules[:-3])
    return resnet_backbone
    # _final_conv_layer = resnet_backbone[-1][-1].conv3
    # return resnet_backbone, _final_conv_layer


def prep_image(image_path):
    img = Image.open(image_path)
    img = resize_and_pad(img, side_len=RESNET_IMAGE_SIDE_LEN)
    img = to_tensor(img)
    # remove alpha channel, if there is one
    img = img[:3]
    return img

def show_img(img):
    plt.imshow(img.permute(1, 2, 0))
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    img_path = os.path.join(
        AI_DATA_DIR,
        'detect-grid-easy--2023-08-03/images',
        'IMG_2359.png',
    )
    img = prep_image(img_path)
    batched_img = img.unsqueeze(0)
    # show_img(img)

    # model, _final_conv_layer = get_model()
    model = get_model()
    output, layer_outputs = model(batched_img)
    print('output.shape:', output.shape)
    for layer_output, name in layer_outputs:
        print(f'layer {name} output shape:', layer_output.shape)
    assert False

    # print('feature shape:', output[0, 0, :, :].shape)
    # I chose these randomly.. not really sure what I'm doing lol
    features_indices_of_interest = [
        0, 1,
        10, 11,
        20, 21,
        100, 200, 300, 400,
        550, 650, 750, 850,
        1000, 1010,
    ]
    # features_indices_of_interest = range(100)
    features_indices_of_interest = range(100, 200)
    num_features = len(features_indices_of_interest)

    nrows = int(math.ceil(math.log(num_features, 2)))
    ncols = int(math.ceil(num_features / nrows))
    print('nrows:', nrows, 'ncols:', ncols)

    for i, feature_idx in enumerate(features_indices_of_interest):
        feature = output[0, feature_idx, :, :]
        ax = plt.subplot(nrows, ncols, i+1)
        ax.axis('off')
        ax.imshow(feature.detach().numpy(), cmap='viridis')
    plt.show()
