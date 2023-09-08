import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from settings import NUM_CLASSES, GRID
from parse_image.utils.slice_tensor import slice_tensor
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


class ResizeFeatureMap(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=False):
        super().__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    def __repr__(self):
        attrs = ['size', 'mode']
        attr_str = ', '.join(f'{attr}={getattr(self, attr)}' for attr in attrs)
        return f'{self.__class__.__name__}({attr_str})'


class ExtractSubGrid(nn.Module):
    """Extract a sub-grid from a feature map."""
    def __init__(self, y_slice=None, x_slice=None):
        super().__init__()
        self.y_slice = y_slice
        self.x_slice = x_slice
        self.slice_specs = []

        if self.y_slice:
            assert 'dim' in self.y_slice, 'must specify "dim"'
            self.slice_specs.append(self.y_slice)
        if self.x_slice:
            assert 'dim' in self.x_slice, 'must specify "dim"'
            self.slice_specs.append(self.x_slice)

    def forward(self, x):
        return slice_tensor(x, self.slice_specs)

    def __repr__(self):
        attrs = ['y_slice', 'x_slice']
        attr_str = ', '.join(f'{attr}={getattr(self, attr)}' for attr in attrs)
        return f'{self.__class__.__name__}({attr_str})'


class GridDetectionHead(nn.Module):

    def __init__(self, grid_size, num_classes, feature_map_shape):
        super().__init__()
        backbone_out_filters = feature_map_shape[0]
        fm_h, fm_w = feature_map_shape[-2:]
        assert fm_h == fm_w, 'feature map should be square'
        assert fm_h >= grid_size, f'feature map height ({fm_h}) is not >= grid height ({grid_size})'

        # GRID_CONV_REDUCTION_FACTOR = 8
        kernel_size = 3
        grid_conv = dict(
            # reduction_factor=16,
            reduction_factor=8,
            # reduction_factor=4,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        grid_conv_out_filters = backbone_out_filters // grid_conv['reduction_factor']
        fc_in = 10 * 6 * grid_conv_out_filters
        # TODO: don't hardcode this
        fc_out = 10 * 6 * num_classes
        self.out_features = fc_out

        # TODO: shape log everything in a cleaner fashion
        self.layers = nn.Sequential(
            ResizeFeatureMap(grid_size),
            # ShapeLogger('ResizeFeatureMap'),
            nn.Conv2d(
                backbone_out_filters,
                grid_conv_out_filters,
                kernel_size=grid_conv['kernel_size'],
                padding=grid_conv['padding'],
            ),
            # nn.Conv2d(
            #     grid_conv_out_filters,
            #     grid_conv_out_filters // 2,
            #     kernel_size=grid_conv['kernel_size'],
            #     padding=grid_conv['padding'],
            # ),
            # ShapeLogger('Conv2d'),
            # Thesee values (2, -2) come from this calc: 10 - 6 / 2
            # TODO: don't hardcode this
            ExtractSubGrid(x_slice=dict(dim=-1, start=2, end=-2)),
            # ShapeLogger('ExtractSubGrid'),
            nn.Flatten(start_dim=1),
            # ShapeLogger('Flatten'),
            nn.Linear(fc_in, fc_out),
            # nn.Linear(in_features=int(fc_in / 2), out_features=fc_out),
        )

    def forward(self, x):
        x = self.layers(x)
        return x.view(-1, 10, 6, 12)
        # return x.view(-1, self.out_features)


def load_backbone(backbone_weights_path):
    if os.path.exists(backbone_weights_path):
        backbone = torch.load(backbone_weights_path)
    else:
        vanilla_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        resnet_modules = list(vanilla_resnet.children())
        module_names = [name for name, _ in vanilla_resnet.named_children()]

        IMPLEMENTATION_HAS_CHANGED = (
            module_names[:len(RESNET_BACKBONE)] != RESNET_BACKBONE
        )
        assert not IMPLEMENTATION_HAS_CHANGED, 'ResNet implementation has changed!'

        print('--- vanilla_resnet.childrden[-1] ---', resnet_modules[-1])
        print('--- vanilla_resnet.childrden[-2] ---', resnet_modules[-2])
        # Remove the last 2 layers, which are avgpool and fc
        backbone = nn.Sequential(*resnet_modules[:-3])
        output_shape = get_output_shape(backbone, RESNET_IMAGE_SHAPE)

        out_h, out_w = output_shape[-2:]
        assert out_h >= GRID.height, f'kernel height ({out_h}) is not >= grid height ({GRID.height})'
        assert out_w >= GRID.width, f'kernel width ({out_w}) is not >= grid width ({GRID.width})'

        backbone.output_shape = output_shape 
        backbone.output_num_elems = output_shape.numel()

        print('Saving backbone weights to:', backbone_weights_path)
        torch.save(backbone, backbone_weights_path)
    return backbone


def get_custom_model(backbone_weights_path):
    backbone = load_backbone(backbone_weights_path)
    # in_features = backbone.output_num_elems
    # out_features = GRID.height * GRID.width * NUM_CLASSES

    head = GridDetectionHead(
        grid_size=GRID.height,
        num_classes=NUM_CLASSES,
        feature_map_shape=backbone.output_shape,
    )
    model = GridDetectionModel(backbone=backbone, head=head)
    print('count_params(model.head):', count_params(model.head))
    return model


class ShapeLogger(nn.Module):
    def __init__(self, prev_layer_name):
        super().__init__()
        self.prev_layer_name = prev_layer_name
    def forward(self, x):
        print(f'[ShapeLogger] {self.prev_layer_name} out shape:', x.shape)
        return x

def add_shape_logging_to_sequential_module(module):
    layers = []
    for i, layer in enumerate(module):
        if i == 0:
            layers.append(layer)
        else:
            layers.extend([ShapeLogger(layer.__class__.__name__), layer])
    return nn.Sequential(*layers)