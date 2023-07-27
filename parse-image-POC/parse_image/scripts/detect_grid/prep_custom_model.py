import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch import nn

# Load a pre-trained model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# Get the number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features

# Define the new custom head
class GridPredictor(nn.Module):
    def __init__(self, in_features, hidden_layer, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_layer)
        self.fc2 = nn.Linear(hidden_layer, 6*10*num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x.view(-1, 6, 10, num_classes)

# Replace the pre-trained head with the new one
num_classes = 12
model.roi_heads.box_predictor = GridPredictor(
    in_features,
    hidden_layer=256,
    num_classes=num_classes,
)

# Save this modified model for future use
torch.save(model.state_dict(), 'modified_model.pth')
