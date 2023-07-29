import json

import torch
from PIL import Image

import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
import numpy as np
import requests
import matplotlib.pyplot as plt


def predict_images(model, image_urls, device='cpu'):
    print(f'Using {device} for inference')

    utils = torch.hub.load(
        'NVIDIA/DeepLearningExamples:torchhub',
        'nvidia_convnets_processing_utils',
    )

    model.eval().to(device) 
    batch = torch.cat(
        [utils.prepare_input_from_uri(uri) for uri in image_urls]
    ).to(device)

    with torch.no_grad():
        output = torch.nn.functional.softmax(model(batch), dim=1)
        
    results = utils.pick_n_best(predictions=output, n=5)
    return results


if __name__ == '__main__':
    uris = [
        'http://images.cocodataset.org/test-stuff2017/000000024309.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000028117.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000006149.jpg',
        'http://images.cocodataset.org/test-stuff2017/000000004954.jpg',
    ]

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    # model = torch.hub.load(
    #     'NVIDIA/DeepLearningExamples:torchhub',
    #     'nvidia_resnet50',
    #     pretrained=True,
    # )

    results = predict_images(model, uris)

    for uri, result in zip(uris, results):
        img = Image.open(requests.get(uri, stream=True).raw)
        img.thumbnail((256,256), Image.ANTIALIAS)
        plt.imshow(img)
        plt.show()
        print(result)

