import torch
from torch import nn
import torch.nn.functional as F
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn


class GridPredictor(nn.Module):
    def __init__(self, in_features, hidden_layer_size, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.fc1 = nn.Linear(in_features, hidden_layer_size)
        self.fc2 = nn.Linear(hidden_layer_size, 6*10*num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 6, 10, self.num_classes)


def get_custom_model(num_classes=12, hidden_layer_size=256):
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = GridPredictor(
        in_features,
        hidden_layer_size,
        num_classes,
    )
    return model
