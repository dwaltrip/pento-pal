import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from torch_receptive_field import receptive_field, receptive_field_for_unit
from torchscan import summary



RESNET_IMAGE_SHAPE = (3, 224, 224)
OTHER_IMAGE_SHAPE = (3, 256, 256)

def get_model():
    vanilla_resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet_modules = list(vanilla_resnet.children())
    resnet_backbone = nn.Sequential(*resnet_modules[:-3])
    return resnet_backbone


# using torch_receptive_field, meh
def main1():
    model = get_model()
    receptive_field_dict = receptive_field(model, RESNET_IMAGE_SHAPE)
    receptive_field_for_unit(receptive_field_dict, "2", (1,1))

# using torch-scan
def main2():
    model = get_model()
    summary(model, RESNET_IMAGE_SHAPE, receptive_field=True)


if __name__ == '__main__':
    # main1()
    # main2()

    import torchinfo
    torchinfo.summary(get_model(), input_size=(1, *RESNET_IMAGE_SHAPE))

