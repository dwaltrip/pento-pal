import torch


def get_output_shape(model, image_size):
    """
    Determine the shape of the feature map and output channels produced by a model.
    Args:
        model (nn.Module): the model
        image_size (tuple): the size of the input image tensor (C, H, W)
    Returns:
        tuple: the shape of the output feature map (excluding the batch dimension)
        int: output channels of the last layer of the model
    """
    with torch.no_grad():
        dummy_tensor = torch.zeros((1,) + image_size)  # create a dummy tensor
        output = model(dummy_tensor)  # pass the tensor through the model

    return output.shape[1:]
