import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from parse_image.scripts.detect_grid.config import (
    IMAGE_SIZE,
    PRETRAINED_MODEL_SAVE_PATH,
)
from parse_image.scripts.detect_grid.utils import get_output_shape


class GridDetection(nn.Module):
    def __init__(self, backbone, grid_predictor):
        super().__init__()
        self.backbone = backbone
        self.grid_predictor = grid_predictor

    def forward(self, x):
        x = self.backbone(x) # Feature extraction
        return self.grid_predictor(x)


class GridPredictor(nn.Module):
    def __init__(self, in_features, hidden_layer_size, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(in_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 6*10*num_classes)

    def forward(self, x):
        # x = F.adaptive_avg_pool2d(x, (6, 10))  # Resize to grid size
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 10, 6, self.num_classes)


def load_pretrained_model(model_path):
    if os.path.exists(model_path):
        backbone = torch.load(model_path)
    else:
        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        # Remove the last layer, a fully connected layer.
        backbone = nn.Sequential(*list(backbone.children())[:-1])

        output_size = get_output_shape(backbone, IMAGE_SIZE)
        num_elems = output_size.numel()
        backbone.output_num_elems = num_elems

        torch.save(backbone, model_path)
    return backbone


def get_custom_model(num_classes=12, hidden_layer_size=256):
    backbone = load_pretrained_model(PRETRAINED_MODEL_SAVE_PATH)
    in_features = backbone.output_num_elems
    print('GridPredictor -- in_features:', in_features)
    grid_predictor = GridPredictor(
        in_features,
        hidden_layer_size,
        num_classes,
    )
    model = GridDetection(backbone, grid_predictor)
    return model
