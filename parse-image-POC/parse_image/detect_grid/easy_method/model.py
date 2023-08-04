import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from settings import NUM_CLASSES, GRID
from parse_image.utils.misc import get_output_shape, count_params


RESNET_IMAGE_SHAPE = (3, 224, 224)
# -----------------------------------------------------------------
# RESNET_IMAGE_SHAPE = (3, 199, 199) # TEST THIS! does it break????
# -----------------------------------------------------------------

# From `ResNet._forward_impl`.
# Leaves out the final `flatten` and `fc` layers.
# This is a list of named children that produce the same output as the
#   actual ResNet model class, when they are called sequentially.
# This lets us extract the backbone from the full ResNet model.
RESNET_BACKBONE = [
    'conv1',
    'bn1',
    'relu',
    'maxpool',
    'layer1',
    'layer2',
    'layer3',
    'layer4',
    'avgpool',
]

class GridDetectionModel(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


class GridDetectionHead(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.out_features = out_features

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Linear(256, out_features)
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.linear_relu_stack(x)
        return x.view(-1, self.out_features)


def load_pretrained_model(backbone_save_path):
    if os.path.exists(backbone_save_path):
        backbone = torch.load(backbone_save_path)
    else:
        vanilla_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_modules = list(vanilla_resnet.children())
        module_names = [name for name, _ in vanilla_resnet.named_children()]

        IMPLEMENTATION_HAS_CHANGED = (
            module_names[len(RESNET_BACKBONE)] != RESNET_BACKBONE
        )
        assert not IMPLEMENTATION_HAS_CHANGED, 'ResNet implementation has changed!'

        print('--- vanilla_resnet.childrden[-1] ---', resnet_modules[-1])
        print('--- vanilla_resnet.childrden[-2] ---', resnet_modules[-2])
        # Remove the last 2 layers, which are avgpool and fc
        backbone = nn.Sequential(*list(vanilla_resnet.children())[:-2])

        output_size = get_output_shape(backbone, RESNET_IMAGE_SHAPE)
        num_elems = output_size.numel()
        backbone.output_num_elems = num_elems

        print('Saving backbone to:', backbone_save_path)
        torch.save(backbone, backbone_save_path)
    return backbone


def get_custom_model(save_path):
    backbone = load_pretrained_model(save_path)
    in_features = backbone.output_num_elems
    out_features = GRID.height * GRID.width * NUM_CLASSES
    model = GridDetectionModel(
        backbone=backbone,
        head=GridDetectionHead(in_features, out_features)
    )
    print('count_params(model.head):', count_params(model.head))
    return model
